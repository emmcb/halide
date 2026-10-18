// Exercise the Vulkan runtime with a mock driver, without requiring a GPU.
#include "common.h"
#include "vulkan.cpp"

using namespace Halide::Runtime::Internal;
using namespace Halide::Runtime::Internal::Vulkan;

namespace {

struct TestMemory {
    uint8_t *bytes;
    VkMemoryPropertyFlags properties;
    bool mapped;
};

struct TestBuffer {
    size_t size;
    TestMemory *memory;
    size_t offset;
};

VkPhysicalDeviceMemoryProperties memory_properties;
VulkanMemoryAllocator *allocator;
VkDevice test_device = (VkDevice)1;
bool reject_unified_allocation;
int copies, maps, unmaps, barriers, flushes, invalidates, waits;
int submissions, pools_created, pools_destroyed;
VkResult submit_result = VK_SUCCESS;
VkResult wait_result = VK_SUCCESS;
VkResult end_result = VK_SUCCESS;

struct TestCommand {
    TestBuffer *src;
    TestBuffer *dst;
    VkBufferCopy copy;
};

struct TestCommands {
    TestCommand commands[512];
    int count;
    bool recording;
    bool ended;
    bool pending;
};

struct TestPool {
    TestCommands *commands;
};

TestCommands *pending_commands;
VkResult map_result = VK_SUCCESS;

void *allocate_zeroed(size_t size) {
    void *ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
}

#define check(condition) HALIDE_CHECK(nullptr, condition)

void reset_counts() {
    copies = maps = unmaps = barriers = flushes = invalidates = waits = 0;
    submissions = pools_created = pools_destroyed = 0;
}

VKAPI_ATTR void VKAPI_CALL get_properties(VkPhysicalDevice, VkPhysicalDeviceProperties *properties) {
    *properties = {};
    properties->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    properties->limits.minStorageBufferOffsetAlignment = 32;
    properties->limits.minUniformBufferOffsetAlignment = 32;
    properties->limits.maxUniformBufferRange = 65536;
    properties->limits.maxPushConstantsSize = 128;
}

VKAPI_ATTR void VKAPI_CALL get_memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *properties) {
    *properties = memory_properties;
    properties->memoryHeapCount = 2;
    properties->memoryHeaps[0] = {1ULL << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    properties->memoryHeaps[1] = {1ULL << 30, 0};
    for (uint32_t i = 0; i < properties->memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = properties->memoryTypes[i].propertyFlags;
        check(!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
              (flags & (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)));
        properties->memoryTypes[i].heapIndex = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 0 : 1;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice, const VkMemoryAllocateInfo *info, const VkAllocationCallbacks *, VkDeviceMemory *out) {
    if (reject_unified_allocation &&
        (memory_properties.memoryTypes[info->memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        reject_unified_allocation = false;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    auto *memory = (TestMemory *)malloc(sizeof(TestMemory));
    memory->bytes = (uint8_t *)malloc(info->allocationSize);
    memset(memory->bytes, 0xa5, info->allocationSize);
    memory->properties = memory_properties.memoryTypes[info->memoryTypeIndex].propertyFlags;
    memory->mapped = false;
    *out = (VkDeviceMemory)memory;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory handle, const VkAllocationCallbacks *) {
    auto *memory = (TestMemory *)handle;
    check(!memory->mapped);
    free(memory->bytes);
    free(memory);
}

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo *info, const VkAllocationCallbacks *, VkBuffer *out) {
    auto *buffer = (TestBuffer *)malloc(sizeof(TestBuffer));
    *buffer = {size_t(info->size), nullptr, 0};
    *out = (VkBuffer)buffer;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks *) {
    free((void *)buffer);
}

VKAPI_ATTR void VKAPI_CALL get_requirements(VkDevice, VkBuffer handle, VkMemoryRequirements *requirements) {
    auto *buffer = (TestBuffer *)handle;
    *requirements = {(buffer->size + 31) & ~size_t(31), 32, (1U << memory_properties.memoryTypeCount) - 1};
}

VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice, VkBuffer handle, VkDeviceMemory memory, VkDeviceSize offset) {
    auto *buffer = (TestBuffer *)handle;
    buffer->memory = (TestMemory *)memory;
    buffer->offset = offset;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice, VkDeviceMemory handle, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags, void **out) {
    maps++;
    if (map_result != VK_SUCCESS) {
        return map_result;
    }
    auto *memory = (TestMemory *)handle;
    check(memory->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    check(!memory->mapped);
    if (size == VK_WHOLE_SIZE) {
        check(offset == 0 && !pending_commands);
    }
    memory->mapped = true;
    *out = memory->bytes + offset;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL unmap_memory(VkDevice, VkDeviceMemory handle) {
    unmaps++;
    auto *memory = (TestMemory *)handle;
    check(memory->mapped);
    memory->mapped = false;
}

void check_cache_range(uint32_t count, const VkMappedMemoryRange *range) {
    check(count == 1 && range->offset == 0 && range->size == VK_WHOLE_SIZE);
    check(((TestMemory *)range->memory)->mapped);
}

VKAPI_ATTR VkResult VKAPI_CALL flush_memory(VkDevice, uint32_t count, const VkMappedMemoryRange *range) {
    flushes++;
    check_cache_range(count, range);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL invalidate_memory(VkDevice, uint32_t count, const VkMappedMemoryRange *range) {
    invalidates++;
    check_cache_range(count, range);
    check(barriers > 0 && waits > 0);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL create_pool(VkDevice, const VkCommandPoolCreateInfo *, const VkAllocationCallbacks *, VkCommandPool *out) {
    pools_created++;
    *out = (VkCommandPool)allocate_zeroed(sizeof(TestPool));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_pool(VkDevice, VkCommandPool handle, const VkAllocationCallbacks *) {
    pools_destroyed++;
    auto *pool = (TestPool *)handle;
    if (pool->commands) {
        check(!pool->commands->pending);
        free(pool->commands);
    }
    free(pool);
}

VKAPI_ATTR VkResult VKAPI_CALL reset_pool(VkDevice, VkCommandPool handle, VkCommandPoolResetFlags) {
    auto *pool = (TestPool *)handle;
    check(!pool->commands || !pool->commands->pending);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_commands(VkDevice, const VkCommandBufferAllocateInfo *info, VkCommandBuffer *out) {
    auto *pool = (TestPool *)info->commandPool;
    check(!pool->commands);
    pool->commands = (TestCommands *)allocate_zeroed(sizeof(TestCommands));
    *out = (VkCommandBuffer)pool->commands;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL free_commands(VkDevice, VkCommandPool handle, uint32_t count, const VkCommandBuffer *commands) {
    auto *pool = (TestPool *)handle;
    check(count == 1 && (TestCommands *)commands[0] == pool->commands && !pool->commands->pending);
    free(pool->commands);
    pool->commands = nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL begin_commands(VkCommandBuffer handle, const VkCommandBufferBeginInfo *) {
    auto *commands = (TestCommands *)handle;
    check(!commands->recording && !commands->ended);
    commands->recording = true;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL end_commands(VkCommandBuffer handle) {
    auto *commands = (TestCommands *)handle;
    check(commands->recording && !commands->ended);
    commands->recording = false;
    commands->ended = true;
    return end_result;
}

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue, uint32_t count, const VkSubmitInfo *info, VkFence) {
    submissions++;
    if (submit_result != VK_SUCCESS) {
        return submit_result;
    }
    check(count == 1 && info->commandBufferCount == 1);
    auto *commands = (TestCommands *)info->pCommandBuffers[0];
    check(commands->ended && !commands->pending && !pending_commands);
    commands->pending = true;
    pending_commands = commands;
    for (int i = 0; i < commands->count; i++) {
        TestCommand &command = commands->commands[i];
        memcpy(command.dst->memory->bytes + command.dst->offset + command.copy.dstOffset,
               command.src->memory->bytes + command.src->offset + command.copy.srcOffset, command.copy.size);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkQueue) {
    waits++;
    if (wait_result == VK_SUCCESS && pending_commands) {
        pending_commands->pending = false;
        pending_commands = nullptr;
    }
    return wait_result;
}

VKAPI_ATTR void VKAPI_CALL pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                            VkDependencyFlags, uint32_t count, const VkMemoryBarrier *barrier,
                                            uint32_t, const VkBufferMemoryBarrier *, uint32_t, const VkImageMemoryBarrier *) {
    barriers++;
    check(count == 1);
    if (dst == VK_PIPELINE_STAGE_HOST_BIT) {
        check(src == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        check(barrier->srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT && barrier->dstAccessMask == VK_ACCESS_HOST_READ_BIT);
    }
}

VKAPI_ATTR void VKAPI_CALL copy_buffer(VkCommandBuffer handle, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy *regions) {
    copies++;
    auto *commands = (TestCommands *)handle;
    check(commands->recording);
    for (uint32_t i = 0; i < count; i++) {
        check(commands->count < 128);
        commands->commands[commands->count++] = {(TestBuffer *)src, (TestBuffer *)dst, regions[i]};
    }
}

VKAPI_ATTR void VKAPI_CALL fill_buffer(VkCommandBuffer, VkBuffer handle, VkDeviceSize offset, VkDeviceSize size, uint32_t value) {
    auto *buffer = (TestBuffer *)handle;
    if (size == VK_WHOLE_SIZE) {
        size = buffer->size - offset;
    }
    uint32_t *data = (uint32_t *)(buffer->memory->bytes + buffer->offset + offset);
    for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
        data[i] = value;
    }
}

int acquire(void *, halide_vulkan_memory_allocator **out_allocator, VkInstance *instance, VkDevice *device,
            VkPhysicalDevice *physical_device, VkQueue *queue, uint32_t *family, VkDebugUtilsMessengerEXT *messenger, bool) {
    *out_allocator = (halide_vulkan_memory_allocator *)allocator;
    *instance = (VkInstance)1;
    *device = test_device;
    *physical_device = (VkPhysicalDevice)1;
    *queue = (VkQueue)1;
    *family = 0;
    *messenger = VK_NULL_HANDLE;
    return halide_error_code_success;
}

int release(void *, VkInstance, VkDevice, VkQueue, VkDebugUtilsMessengerEXT) {
    return halide_error_code_success;
}

void ignore_error(void *, const char *) {
}

void test_transfers(bool coherent, bool unified, bool reject_allocation = false, bool cached_memory = false) {
    const bool visible = unified && !reject_allocation;
    const bool mapped = visible && coherent;
    reject_unified_allocation = reject_allocation;
    memory_properties = {};
    memory_properties.memoryTypeCount = 2;
    memory_properties.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory_properties.memoryTypes[coherent ? 1 : 0].propertyFlags =
        (unified ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0) | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        (coherent ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (!coherent) {
        memory_properties.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    if (cached_memory) {
        // Prefer the cached coherent type over an earlier non-coherent type.
        memory_properties.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        memory_properties.memoryTypes[1].propertyFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    allocator = VulkanMemoryAllocator::create(nullptr, VulkanMemoryAllocator::default_config(), (VkDevice)1, (VkPhysicalDevice)1,
                                              {vk_system_malloc, vk_system_free});
    check(allocator != nullptr);

    // Allocate a prefix in the same block to exercise suballocation offsets.
    halide_dimension_t prefix_dim = {0, 8, 1, 0};
    halide_buffer_t prefix = {};
    prefix.type = halide_type_t(halide_type_int, 32);
    prefix.dimensions = 1;
    prefix.dim = &prefix_dim;
    check(halide_vulkan_device_malloc(nullptr, &prefix) == 0);
    check(!reject_unified_allocation);

    int32_t host[24];
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 8; x++) {
            host[y * 8 + x] = 100 * y + x;
        }
    }
    halide_dimension_t dimensions[2] = {{0, 5, 1, 0}, {0, 3, 8, 0}};
    halide_buffer_t buffer = {};
    buffer.type = prefix.type;
    buffer.host = (uint8_t *)host;
    buffer.dimensions = 2;
    buffer.dim = dimensions;
    check(halide_vulkan_device_malloc(nullptr, &buffer) == 0);
    auto *region = (MemoryRegion *)buffer.device;
    check(region->allocation.offset > 0);
    const auto *allocation = allocator->allocation_for(nullptr, region);
    check(bool(allocation->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == visible);
    const size_t allocated_regions = allocator->regions_allocated();
    auto *device_buffer = (TestBuffer *)*(VkBuffer *)region->handle;
    auto *device_data = (int32_t *)(device_buffer->memory->bytes + device_buffer->offset);
    const int32_t gap_value = device_data[5];

    reset_counts();
    check(halide_vulkan_copy_to_device(nullptr, &buffer) == 0);
    check((copies == 0) == mapped);
    check(maps == 1 && unmaps == 1 && barriers == 0);
    check(flushes == 0);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 8 && y * 8 + x < 21; x++) {
            check(device_data[y * 8 + x] == (x < 5 ? 100 * y + x : gap_value));
        }
    }
    if (mapped) {
        check(allocator->regions_allocated() == allocated_regions);
    }
    for (int &value : host) {
        value = -1;
    }
    reset_counts();
    check(halide_vulkan_copy_to_host(nullptr, &buffer) == 0);
    check((copies == 0) == mapped);
    check(barriers == (mapped ? 1 : 0));
    check(invalidates == 0);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 8; x++) {
            check(host[y * 8 + x] == (x < 5 ? 100 * y + x : -1));
        }
    }

    halide_buffer_t host_buffer = buffer;
    host_buffer.device = 0;
    host_buffer.device_interface = nullptr;
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &host_buffer, &vulkan_device_interface, &buffer) == 0);
    check((copies == 0) == mapped);
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &buffer, nullptr, &host_buffer) == 0);
    check((copies == 0) == mapped);

    // Device copies preserve multidimensional strides.
    int32_t destination_host[24];
    for (int32_t &value : destination_host) {
        value = -1;
    }
    halide_dimension_t destination_dims[2] = {{0, 5, 1, 0}, {0, 3, 7, 0}};
    halide_buffer_t destination = buffer;
    destination.device = 0;
    destination.device_interface = nullptr;
    destination.host = (uint8_t *)destination_host;
    destination.dim = destination_dims;
    check(halide_vulkan_device_malloc(nullptr, &destination) == 0);
    check(halide_vulkan_copy_to_device(nullptr, &destination) == 0);
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &buffer, &vulkan_device_interface, &destination) == 0);
    check(copies == 3 && submissions == 1 && waits == 2 && maps == 0);
    check(halide_vulkan_copy_to_host(nullptr, &destination) == 0);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 7; x++) {
            check(destination_host[y * 7 + x] == (x < 5 ? 100 * y + x : -1));
        }
    }
    check(halide_vulkan_device_free(nullptr, &destination) == 0);

    if (mapped) {
        // A cropped device buffer copied to/from a host buffer with different
        // minima and strides must leave the surrounding pixels untouched.
        MemoryRegion *crop_region = allocator->create_crop(nullptr, region, {9});
        check(crop_region != nullptr);
        halide_dimension_t crop_dims[2] = {{1, 3, 1, 0}, {1, 2, 8, 0}};
        halide_buffer_t crop = buffer;
        crop.device = (uint64_t)crop_region;
        crop.dim = crop_dims;
        crop.host = (uint8_t *)(host + 9);
        int32_t small_host[8] = {777, 778, -1, -1, 779, 780, -1, -1};
        halide_dimension_t small_dims[2] = {{2, 2, 1, 0}, {1, 2, 4, 0}};
        halide_buffer_t small = {};
        small.type = buffer.type;
        small.host = (uint8_t *)small_host;
        small.dimensions = 2;
        small.dim = small_dims;
        reset_counts();
        check(halide_vulkan_buffer_copy(nullptr, &small, &vulkan_device_interface, &crop) == 0);
        check(copies == 0 && maps == 1 && unmaps == 1);
        small_host[0] = small_host[1] = small_host[4] = small_host[5] = -1;
        check(halide_vulkan_buffer_copy(nullptr, &crop, nullptr, &small) == 0);
        check(copies == 0 && maps == 2 && unmaps == 2 && barriers == 1);
        check(small_host[0] == 777 && small_host[1] == 778 && small_host[4] == 779 && small_host[5] == 780);
        check(small_host[2] == -1 && small_host[3] == -1 && small_host[6] == -1 && small_host[7] == -1);
        check(halide_vulkan_copy_to_host(nullptr, &buffer) == 0);
        check(host[9] == 101 && host[10] == 777 && host[11] == 778 && host[12] == 104);
        check(host[17] == 201 && host[18] == 779 && host[19] == 780 && host[20] == 204);
        check(allocator->destroy_crop(nullptr, crop_region) == 0);

        reset_counts();
        map_result = VK_ERROR_MEMORY_MAP_FAILED;
        check(halide_vulkan_copy_to_device(nullptr, &buffer) == halide_error_code_device_buffer_copy_failed);
        check(maps == 1 && unmaps == 0 && copies == 0);
        map_result = VK_SUCCESS;
    }
    check(halide_vulkan_device_free(nullptr, &buffer) == 0);
    check(halide_vulkan_device_free(nullptr, &prefix) == 0);
    // Collect cached blocks through the allocator's callback context before
    // destroying the allocator itself.
    allocator->collect(nullptr);
    check(allocator->blocks_allocated() == 0);
    check(VulkanMemoryAllocator::destroy(nullptr, allocator) == 0);
    allocator = nullptr;
}

}  // namespace

int main(int, char **) {
    vkGetPhysicalDeviceProperties = get_properties;
    vkGetPhysicalDeviceMemoryProperties = get_memory_properties;
    vkAllocateMemory = allocate_memory;
    vkFreeMemory = free_memory;
    vkCreateBuffer = create_buffer;
    vkDestroyBuffer = destroy_buffer;
    vkGetBufferMemoryRequirements = get_requirements;
    vkBindBufferMemory = bind_memory;
    vkMapMemory = map_memory;
    vkUnmapMemory = unmap_memory;
    vkFlushMappedMemoryRanges = flush_memory;
    vkInvalidateMappedMemoryRanges = invalidate_memory;
    vkCreateCommandPool = create_pool;
    vkDestroyCommandPool = destroy_pool;
    vkResetCommandPool = reset_pool;
    vkAllocateCommandBuffers = allocate_commands;
    vkFreeCommandBuffers = free_commands;
    vkBeginCommandBuffer = begin_commands;
    vkEndCommandBuffer = end_commands;
    vkQueueSubmit = submit;
    vkQueueWaitIdle = wait_idle;
    vkCmdPipelineBarrier = pipeline_barrier;
    vkCmdCopyBuffer = copy_buffer;
    vkCmdFillBuffer = fill_buffer;
    Halide::Runtime::Internal::Vulkan::acquire_context = acquire;
    Halide::Runtime::Internal::Vulkan::release_context = release;
    halide_set_error_handler(ignore_error);
    halide_reuse_device_allocations(nullptr, false);

    test_transfers(true, true);
    test_transfers(false, true);
    test_transfers(true, false);
    test_transfers(true, true, true);
    test_transfers(true, true, false, true);
    halide_print(nullptr, "Success!\n");
    return 0;
}
