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
VkQueue test_queue = (VkQueue)1;
bool reject_unified_allocation;
int copies, maps, unmaps, barriers, host_barriers, flushes, invalidates, waits;
int submissions, recorded_dispatches, executed_dispatches, pools_created, pools_destroyed;
int descriptor_sets, descriptor_pools, descriptor_pools_created, descriptor_pools_reset, pipelines, pipelines_created;
VkResult pipeline_result = VK_SUCCESS;
VkResult submit_result = VK_SUCCESS;
VkResult wait_result = VK_SUCCESS;
VkResult end_result = VK_SUCCESS;
VkResult descriptor_result = VK_SUCCESS;

struct TestDescriptors {
    TestBuffer *uniform;
    size_t uniform_offset;
    TestBuffer *storage;
    TestBuffer *input;
};

struct TestDescriptorPool {
    TestDescriptors *sets[64];
    uint32_t count;
};

struct TestPipeline {
    uint32_t threads;
};

struct TestDispatch {
    TestDescriptors *descriptors;
    TestPipeline *pipeline;
    uint32_t constants[32];
    TestDescriptors descriptor_snapshot;
};

struct TestCommand {
    TestDispatch dispatch;
    TestBuffer *src;
    TestBuffer *dst;
    VkBufferCopy copy;
};

struct TestCommands {
    TestCommand commands[512];
    int count;
    TestDescriptors *descriptors;
    TestPipeline *pipeline;
    uint32_t constants[32];
    TestDescriptors pushed_descriptors;
    bool recording;
    bool ended;
    bool pending;
};

struct TestPool {
    TestCommands *commands;
};

struct TestDescriptorTemplate {
    uint32_t count;
    bool uniform;
};

int descriptor_templates, template_updates, descriptor_pushes;

TestCommands *pending_commands;
VkResult map_result = VK_SUCCESS;

void *allocate_zeroed(size_t size) {
    void *ptr = malloc(size);
    memset(ptr, 0, size);
    return ptr;
}

#define check(condition) HALIDE_CHECK(nullptr, condition)

void reset_counts() {
    copies = maps = unmaps = barriers = host_barriers = flushes = invalidates = waits = 0;
    submissions = recorded_dispatches = executed_dispatches = pools_created = pools_destroyed = 0;
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
        if (command.src) {
            memcpy(command.dst->memory->bytes + command.dst->offset + command.copy.dstOffset,
                   command.src->memory->bytes + command.src->offset + command.copy.srcOffset, command.copy.size);
            continue;
        }
        TestDispatch &dispatch = command.dispatch;
        TestBuffer *uniform = dispatch.descriptors->uniform;
        TestBuffer *storage = dispatch.descriptors->storage;
        auto *args = uniform ? (int32_t *)(uniform->memory->bytes + uniform->offset + dispatch.descriptors->uniform_offset) : (int32_t *)dispatch.constants;
        auto *output = (int32_t *)(storage->memory->bytes + storage->offset) + args[dispatch.descriptors->input ? 2 : 1];
        auto *input = output;
        if (dispatch.descriptors->input) {
            TestBuffer *input_buffer = dispatch.descriptors->input;
            input = (int32_t *)(input_buffer->memory->bytes + input_buffer->offset) + args[1];
        }
        // Model a dependent kernel with scalar arguments and a cropped offset.
        *output = 2 * *input + args[0] + dispatch.pipeline->threads;
        executed_dispatches++;
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

VKAPI_ATTR VkResult VKAPI_CALL create_descriptor_pool(VkDevice, const VkDescriptorPoolCreateInfo *info, const VkAllocationCallbacks *, VkDescriptorPool *out) {
    check(info->maxSets == 64);
    for (uint32_t i = 0; i < info->poolSizeCount; i++) {
        check(info->pPoolSizes[i].descriptorCount >= 64);
    }
    descriptor_pools++;
    descriptor_pools_created++;
    *out = (VkDescriptorPool)allocate_zeroed(sizeof(TestDescriptorPool));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL reset_descriptor_pool(VkDevice, VkDescriptorPool handle, VkDescriptorPoolResetFlags) {
    check(!pending_commands);
    auto *pool = (TestDescriptorPool *)handle;
    descriptor_pools_reset++;
    for (uint32_t i = 0; i < pool->count; i++) {
        free(pool->sets[i]);
        descriptor_sets--;
    }
    pool->count = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_descriptor_pool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks *) {
    reset_descriptor_pool(device, pool, 0);
    descriptor_pools--;
    free((void *)pool);
}

VKAPI_ATTR VkResult VKAPI_CALL allocate_descriptors(VkDevice, const VkDescriptorSetAllocateInfo *info, VkDescriptorSet *out) {
    check(info->descriptorSetCount == 1);
    if (descriptor_result != VK_SUCCESS) {
        return descriptor_result;
    }
    auto *pool = (TestDescriptorPool *)info->descriptorPool;
    check(pool->count < 64);
    auto *set = (TestDescriptors *)allocate_zeroed(sizeof(TestDescriptors));
    pool->sets[pool->count++] = set;
    descriptor_sets++;
    *out = (VkDescriptorSet)set;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL update_descriptors(VkDevice, uint32_t count, const VkWriteDescriptorSet *writes, uint32_t, const VkCopyDescriptorSet *) {
    check(count >= 1 && count <= 3);
    const bool has_uniform = writes[0].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    const uint32_t storage_count = count - has_uniform;
    for (uint32_t i = 0; i < count; i++) {
        auto *descriptors = (TestDescriptors *)writes[i].dstSet;
        auto *buffer = (TestBuffer *)writes[i].pBufferInfo->buffer;
        if (writes[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            descriptors->uniform = buffer;
            descriptors->uniform_offset = writes[i].pBufferInfo->offset;
            check(descriptors->uniform_offset % 32 == 0);
        } else if (storage_count == 2 && writes[i].dstBinding == uint32_t(has_uniform)) {
            descriptors->input = buffer;
        } else {
            descriptors->storage = buffer;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL create_descriptor_template(VkDevice, const VkDescriptorUpdateTemplateCreateInfo *info, const VkAllocationCallbacks *, VkDescriptorUpdateTemplate *out) {
    auto *entry = (TestDescriptorTemplate *)malloc(sizeof(TestDescriptorTemplate));
    *entry = {info->descriptorUpdateEntryCount, info->pDescriptorUpdateEntries[0].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    for (uint32_t i = 0; i < entry->count; i++) {
        check(info->pDescriptorUpdateEntries[i].offset == i * sizeof(VkDescriptorBufferInfo));
    }
    *out = (VkDescriptorUpdateTemplate)entry;
    descriptor_templates++;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_descriptor_template(VkDevice, VkDescriptorUpdateTemplate handle, const VkAllocationCallbacks *) {
    descriptor_templates--;
    free((void *)handle);
}

VKAPI_ATTR void VKAPI_CALL update_with_template(VkDevice device, VkDescriptorSet set, VkDescriptorUpdateTemplate handle, const void *data) {
    auto *entry = (TestDescriptorTemplate *)handle;
    VkWriteDescriptorSet writes[3] = {};
    check(entry->count <= 3);
    for (uint32_t i = 0; i < entry->count; i++) {
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = entry->uniform && i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = (const VkDescriptorBufferInfo *)data + i;
    }
    template_updates++;
    update_descriptors(device, entry->count, writes, 0, nullptr);
}

VKAPI_ATTR void VKAPI_CALL push_descriptors(VkCommandBuffer handle, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t count, const VkWriteDescriptorSet *src) {
    auto *commands = (TestCommands *)handle;
    check(commands->recording && count <= 3);
    VkWriteDescriptorSet writes[3];
    commands->pushed_descriptors = {};
    for (uint32_t i = 0; i < count; i++) {
        writes[i] = src[i];
        writes[i].dstSet = (VkDescriptorSet)&commands->pushed_descriptors;
    }
    update_descriptors(test_device, count, writes, 0, nullptr);
    commands->descriptors = &commands->pushed_descriptors;
    descriptor_pushes++;
}

VKAPI_ATTR void VKAPI_CALL push_with_template(VkCommandBuffer handle, VkDescriptorUpdateTemplate entry, VkPipelineLayout, uint32_t, const void *data) {
    auto *commands = (TestCommands *)handle;
    commands->pushed_descriptors = {};
    update_with_template(test_device, (VkDescriptorSet)&commands->pushed_descriptors, entry, data);
    commands->descriptors = &commands->pushed_descriptors;
    descriptor_pushes++;
}

VKAPI_ATTR VkResult VKAPI_CALL create_descriptor_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo *info, const VkAllocationCallbacks *, VkDescriptorSetLayout *out) {
    check(info->bindingCount >= 1);
    if (info->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) {
        check(allocator->dispatch_extensions() & VulkanPushDescriptors);
        check(info->bindingCount <= 32);
    }
    *out = (VkDescriptorSetLayout)1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL create_pipeline_layout(VkDevice, const VkPipelineLayoutCreateInfo *info, const VkAllocationCallbacks *, VkPipelineLayout *out) {
    check(info->setLayoutCount == 1 && info->pushConstantRangeCount <= 1);
    if (info->pushConstantRangeCount) {
        check(info->pPushConstantRanges->size == 8 && info->pPushConstantRanges->offset == 0);
        check(info->pPushConstantRanges->stageFlags == VK_SHADER_STAGE_COMPUTE_BIT);
    }
    *out = (VkPipelineLayout)1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, uint32_t count, const VkComputePipelineCreateInfo *info, const VkAllocationCallbacks *, VkPipeline *out) {
    check(count == 1);
    pipelines_created++;
    if (pipeline_result != VK_SUCCESS) {
        *out = VK_NULL_HANDLE;
        return pipeline_result;
    }
    pipelines++;
    auto *pipeline = (TestPipeline *)allocate_zeroed(sizeof(TestPipeline));
    if (info->stage.pSpecializationInfo) {
        pipeline->threads = ((const uint32_t *)info->stage.pSpecializationInfo->pData)[0];
    }
    *out = (VkPipeline)pipeline;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline handle, const VkAllocationCallbacks *) {
    check(!pending_commands);
    pipelines--;
    free((void *)handle);
}

VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer handle, VkPipelineBindPoint, VkPipeline pipeline) {
    auto *commands = (TestCommands *)handle;
    check(commands->recording);
    commands->pipeline = (TestPipeline *)pipeline;
}

VKAPI_ATTR void VKAPI_CALL bind_descriptors(VkCommandBuffer handle, VkPipelineBindPoint, VkPipelineLayout, uint32_t, uint32_t count, const VkDescriptorSet *sets, uint32_t, const uint32_t *) {
    check(count == 1);
    auto *commands = (TestCommands *)handle;
    check(commands->recording);
    commands->descriptors = (TestDescriptors *)sets[0];
}

VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer handle, uint32_t, uint32_t, uint32_t) {
    auto *commands = (TestCommands *)handle;
    check(commands->recording && commands->count < 128);
    commands->commands[commands->count++] = {{commands->descriptors, commands->pipeline}, nullptr, nullptr, {}};
    memcpy(commands->commands[commands->count - 1].dispatch.constants, commands->constants, sizeof(commands->constants));
    if (commands->descriptors == &commands->pushed_descriptors) {
        auto &dispatch = commands->commands[commands->count - 1].dispatch;
        dispatch.descriptor_snapshot = commands->pushed_descriptors;
        dispatch.descriptors = &dispatch.descriptor_snapshot;
    }
    recorded_dispatches++;
}

VKAPI_ATTR void VKAPI_CALL push_constants(VkCommandBuffer handle, VkPipelineLayout, VkShaderStageFlags, uint32_t offset, uint32_t size, const void *data) {
    auto *commands = (TestCommands *)handle;
    check(commands->recording && offset == 0 && size <= 128 && size % 4 == 0);
    memcpy(commands->constants, data, size);
}

VKAPI_ATTR void VKAPI_CALL pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                            VkDependencyFlags, uint32_t count, const VkMemoryBarrier *barrier,
                                            uint32_t, const VkBufferMemoryBarrier *, uint32_t, const VkImageMemoryBarrier *) {
    barriers++;
    check(count == 1);
    if (dst == VK_PIPELINE_STAGE_HOST_BIT) {
        check(src == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        check(barrier->srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT && barrier->dstAccessMask == VK_ACCESS_HOST_READ_BIT);
    } else if (dst == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        check(src == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && barrier->srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT);
        check(barrier->dstAccessMask & VK_ACCESS_SHADER_READ_BIT);
        check(barrier->dstAccessMask & VK_ACCESS_SHADER_WRITE_BIT);
    } else if (dst == VK_PIPELINE_STAGE_TRANSFER_BIT) {
        check(src == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && barrier->srcAccessMask == VK_ACCESS_MEMORY_WRITE_BIT);
        check(barrier->dstAccessMask == (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT));
    } else {
        check(src == (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT));
        check(dst == VK_PIPELINE_STAGE_ALL_COMMANDS_BIT || dst == (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT));
        check(barrier->srcAccessMask == (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) && (barrier->dstAccessMask & VK_ACCESS_MEMORY_READ_BIT));
    }
    if (dst & VK_PIPELINE_STAGE_HOST_BIT) {
        check(barrier->dstAccessMask & VK_ACCESS_HOST_READ_BIT);
        host_barriers++;
    } else {
        check(!(barrier->dstAccessMask & VK_ACCESS_HOST_READ_BIT));
    }
}

VKAPI_ATTR void VKAPI_CALL copy_buffer(VkCommandBuffer handle, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy *regions) {
    copies++;
    auto *commands = (TestCommands *)handle;
    check(commands->recording);
    for (uint32_t i = 0; i < count; i++) {
        check(commands->count < 128);
        commands->commands[commands->count++] = {{}, (TestBuffer *)src, (TestBuffer *)dst, regions[i]};
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
    *queue = test_queue;
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
    if (mapped) {
        check(submissions == 0 && waits == 0);
    }
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &buffer, nullptr, &host_buffer) == 0);
    check((copies == 0) == mapped);
    if (mapped) {
        // The previous readback covers all allocations, including direct host
        // writes, until another GPU operation writes memory on this queue.
        check(submissions == 0 && waits == 0 && host_barriers == 0);
    }

    // Device copies preserve multidimensional strides in a deferred batch.
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
    check(copies == 3 && submissions == 0 && waits == 0 && maps == 0);
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
        check(copies == 0 && maps == 2 && unmaps == 2 && barriers == 0);
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
    check(halide_vulkan_device_release(nullptr) == 0);
    check(dispatch_arenas == nullptr);
    // Collect cached blocks through the allocator's callback context before
    // destroying the allocator itself.
    allocator->collect(nullptr);
    check(allocator->blocks_allocated() == 0);
    check(VulkanMemoryAllocator::destroy(nullptr, allocator) == 0);
    allocator = nullptr;
}

void test_pipeline_cache() {
    VulkanSpecializationConstant constants[] = {{1, 4, "x"}, {2, 4, "y"}, {3, 4, "z"}, {4, 4, "shared"}};
    VulkanSharedMemoryAllocation allocation = {4, 4, 0, "shared"};
    VulkanShaderBinding binding = {};
    binding.entry_point_name = const_cast<char *>("test");
    binding.specialization_constants = constants;
    binding.specialization_constants_count = 4;
    binding.shared_memory_allocations = &allocation;
    binding.shared_memory_allocations_count = 1;
    binding.dispatch_data.local_size_binding = {{1, 2, 3}};
    VulkanDispatchData data = {};
    data.local_size[0] = data.local_size[1] = data.local_size[2] = 1;
    data.shared_mem_bytes = 16;
    auto setup = [&](VulkanShaderBinding &b, VkPipeline *out) {
        return vk_setup_compute_pipeline(nullptr, allocator, &b, &data, (VkShaderModule)1, (VkPipelineLayout)1, out);
    };
    const int created_before = pipelines_created;
    VkPipeline original = VK_NULL_HANDLE, pipeline = VK_NULL_HANDLE;
    check(setup(binding, &original) == 0 && original != VK_NULL_HANDLE);
    check(setup(binding, &pipeline) == 0 && pipeline == original);
    data.global_size[0] = 17;
    check(setup(binding, &pipeline) == 0 && pipeline == original);
    check(pipelines_created == created_before + 1);

    // Each local dimension and dynamic shared memory affect specialization.
    for (int dim = 0; dim < 3; dim++) {
        data.local_size[dim] = 2;
        check(setup(binding, &pipeline) == 0 && pipeline != original);
        data.local_size[dim] = 1;
        check(setup(binding, &pipeline) == 0 && pipeline == original);
    }
    data.shared_mem_bytes = 32;
    check(setup(binding, &pipeline) == 0 && pipeline != original);
    data.shared_mem_bytes = 16;
    check(setup(binding, &pipeline) == 0 && pipeline == original);
    check(pipelines_created == created_before + 5 && pipelines == 5);

    // A failed creation must leave existing variants reusable and allow retry.
    pipeline_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    data.shared_mem_bytes = 64;
    check(setup(binding, &pipeline) != 0 && pipelines == 5);
    data.shared_mem_bytes = 16;
    check(setup(binding, &pipeline) == 0 && pipeline == original);
    pipeline_result = VK_SUCCESS;
    data.shared_mem_bytes = 64;
    check(setup(binding, &pipeline) == 0 && pipeline != original && pipelines == 6);

    // Equal specialization values in another binding must not share pipelines.
    VulkanShaderBinding other = binding;
    other.entry_point_name = const_cast<char *>("other");
    other.specialized_pipelines = nullptr;
    VkPipeline separate = VK_NULL_HANDLE;
    check(setup(other, &separate) == 0 && separate != pipeline);
    vk_destroy_shader_binding_pipelines(nullptr, allocator, &other);

    // A shared-memory-only binding can use both the default and specialized
    // pipeline, without overwriting or destroying either cached variant.
    vk_destroy_shader_binding_pipelines(nullptr, allocator, &binding);
    check(pipelines == 0 && binding.specialized_pipelines == nullptr);
    binding.dispatch_data.local_size_binding = {};
    data.shared_mem_bytes = 0;
    check(setup(binding, &original) == 0 && original == binding.compute_pipeline);
    data.shared_mem_bytes = 16;
    check(setup(binding, &pipeline) == 0 && pipeline != original);
    data.shared_mem_bytes = 0;
    check(setup(binding, &pipeline) == 0 && pipeline == original);
    vk_destroy_shader_binding_pipelines(nullptr, allocator, &binding);
    check(pipelines == 0 && binding.compute_pipeline == VK_NULL_HANDLE && binding.specialized_pipelines == nullptr);
}

void test_scalar_argument_alignment() {
    uint8_t narrow = 7;
    uint64_t wide = 0x123456789abcdef0ULL;
    uint32_t logical_bool = 1;
    MemoryRegion region = {};
    region.indexing.offset = 13;
    halide_buffer_t buffer = {};
    buffer.device = (uint64_t)&region;
    size_t sizes[] = {sizeof(narrow), sizeof(halide_buffer_t *), sizeof(wide), sizeof(logical_bool), 0};
    void *args[] = {&narrow, &buffer, &wide, &logical_bool};
    int8_t is_buffer[] = {0, 1, 0, 0};
    uint8_t packed[24] = {};
    check(vk_estimate_scalar_uniform_buffer_size(nullptr, sizes, args, is_buffer) == sizeof(packed));
    vk_pack_scalar_arguments(nullptr, packed, sizeof(packed), sizes, args, is_buffer);

    uint64_t packed_wide;
    uint32_t packed_bool;
    int32_t packed_offset;
    memcpy(&packed_wide, packed + 8, sizeof(packed_wide));
    memcpy(&packed_bool, packed + 16, sizeof(packed_bool));
    memcpy(&packed_offset, packed + 20, sizeof(packed_offset));
    check(packed[0] == narrow && packed_wide == wide && packed_bool == logical_bool && packed_offset == 13);
}

void test_argument_pages() {
    VulkanDispatchArena arena = {nullptr, allocator, (VkQueue)1, nullptr, nullptr, false};
    VkDescriptorBufferInfo first = {}, second = {}, slice = {};
    uint8_t *a = nullptr, *b = nullptr, *mapped = nullptr;
    int maps_before = maps;
    check(vk_allocate_argument_slice(nullptr, &arena, 40000, &first, &a) == 0);
    check(vk_allocate_argument_slice(nullptr, &arena, 40000, &second, &b) == 0);
    check(first.buffer != second.buffer && maps == maps_before + 2);
    memset(a, 0x12, 40000);
    memset(b, 0x34, 40000);
    check(vk_allocate_argument_slice(nullptr, &arena, 8, &slice, &mapped) == 0);
    check(slice.offset == 40000 && mapped == b + slice.offset);
    check(a[39999] == 0x12 && b[39999] == 0x34);
    vk_reset_dispatch_arena(nullptr, &arena);
    check(vk_allocate_argument_slice(nullptr, &arena, 40000, &slice, &mapped) == 0);
    check(mapped == b && slice.offset == 0 && maps == maps_before + 2);

    for (auto *page = arena.pages; page; page = page->next) {
        page->used = page->capacity;
    }
    auto *old_pages = arena.pages;
    map_result = VK_ERROR_MEMORY_MAP_FAILED;
    check(vk_allocate_argument_slice(nullptr, &arena, 65536, &slice, &mapped) != 0);
    check(arena.pages == old_pages);
    map_result = VK_SUCCESS;
    check(vk_allocate_argument_slice(nullptr, &arena, 65536, &slice, &mapped) == 0);
    check(slice.range == 65536 && arena.pages != old_pages);
    check(vk_allocate_argument_slice(nullptr, &arena, 65537, &slice, &mapped) != 0);
    while (arena.pages) {
        auto *page = arena.pages;
        arena.pages = page->next;
        vk_destroy_argument_page(nullptr, allocator, page);
    }
}

void test_command_grouping(bool specialized, bool unified, bool coherent = true, bool cached_memory = false, bool push = false) {
    const bool mapped = unified && coherent;
    halide_reuse_device_allocations(nullptr, unified);
    static uintptr_t next_device = 2;
    test_device = (VkDevice)next_device++;
    memory_properties = {};
    memory_properties.memoryTypeCount = coherent ? 2 : 3;
    memory_properties.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    memory_properties.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | (coherent ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_HOST_CACHED_BIT) |
                                                     (unified ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0);
    // Scalar argument and staging buffers need coherent host memory when
    // storage allocations fall back to a non-coherent device-local type.
    memory_properties.memoryTypes[2].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!coherent && unified) {
        // The only device-local type is host-visible but non-coherent, so it
        // must use staging rather than the direct mapped path.
        memory_properties.memoryTypeCount = 2;
        memory_properties.memoryTypes[0].propertyFlags = memory_properties.memoryTypes[1].propertyFlags;
        memory_properties.memoryTypes[1].propertyFlags = memory_properties.memoryTypes[2].propertyFlags;
    }
    if (cached_memory) {
        memory_properties.memoryTypes[1].propertyFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    allocator = VulkanMemoryAllocator::create(nullptr, VulkanMemoryAllocator::default_config(), test_device, (VkPhysicalDevice)1,
                                              {vk_system_malloc, vk_system_free});
    check(allocator != nullptr);

    test_pipeline_cache();
    test_argument_pages();

    // Seed the compilation cache directly: this test models dispatch execution
    // without requiring a SPIR-V compiler or a real Vulkan driver.
    VulkanSpecializationConstant constants[3] = {{1, 4, "x"}, {2, 4, "y"}, {3, 4, "z"}};
    VulkanShaderBinding binding = {};
    binding.entry_point_name = const_cast<char *>("test");
    binding.uniform_buffer_count = push ? 0 : 1;
    binding.push_constant_bytes = push ? 8 : 0;
    binding.storage_buffer_count = 1;
    if (specialized) {
        binding.specialization_constants = constants;
        binding.specialization_constants_count = 3;
        binding.dispatch_data.local_size_binding = {{1, 2, 3}};
    }
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VulkanCompiledShaderModule module = {};
    module.shader_module = (VkShaderModule)1;
    module.descriptor_set_layouts = &descriptor_layout;
    module.shader_count = 1;
    module.shader_bindings = &binding;
    auto *module_ptr = &module;
    VulkanCompilationCacheEntry entry = {allocator, &module_ptr, 1};
    VulkanCompilationCacheEntry *cached = nullptr;
    void *state = nullptr;
    check(compilation_cache.kernel_state_setup(nullptr, &state, test_device, cached, [](VulkanCompilationCacheEntry *e) { return e; }, &entry));
    check(cached == &entry);

    int32_t host_a[4] = {}, host_b[4] = {};
    halide_dimension_t dim = {0, 4, 1, 0};
    halide_buffer_t a = {}, b = {};
    a.type = b.type = halide_type_t(halide_type_int, 32);
    a.dimensions = b.dimensions = 1;
    a.dim = b.dim = &dim;
    a.host = (uint8_t *)host_a;
    b.host = (uint8_t *)host_b;
    check(halide_vulkan_device_malloc(nullptr, &a) == 0);
    check(halide_vulkan_device_malloc(nullptr, &b) == 0);
    // Distinct allocations in the same memory block are independent targets.
    check(a.device != b.device);
    check(allocator->allocation_for(nullptr, (MemoryRegion *)a.device)->memory ==
          allocator->allocation_for(nullptr, (MemoryRegion *)b.device)->memory);
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    check(halide_vulkan_copy_to_device(nullptr, &b) == 0);

    auto run = [&](halide_buffer_t &buffer, int32_t scalar, int threads = 1) {
        size_t sizes[] = {sizeof(scalar), sizeof(halide_buffer_t *), 0};
        void *args[] = {&scalar, &buffer};
        int8_t is_buffer[] = {0, 1};
        return halide_vulkan_run(nullptr, state, "test", 1, 1, 1, threads, 1, 1, 0, sizes, args, is_buffer);
    };
    const int t = specialized ? 1 : 0;
    const int created_before = pipelines_created;
    reset_counts();
    check(run(a, 1) == 0);
    check(run(b, 2, 2) == 0);
    check(run(a, 3, 3) == 0);
    check(recorded_dispatches == 3 && executed_dispatches == 0 && submissions == 0 && waits == 0);
    check(pools_created == 1 && pools_destroyed == 0 && descriptor_sets == 3);
    check(descriptor_pools == 1 && maps == (push ? 0 : 1) && unmaps == 0);
    check(halide_vulkan_device_sync(nullptr, &a) == 0);
    check(submissions == 1 && waits == 1 && executed_dispatches == 3);
    check(pools_destroyed == 1 && descriptor_sets == 0);
    check(descriptor_pools == 1);  // Retained for the next command batch.
    check(pipelines == (specialized ? 3 : 1));
    check(pipelines_created == created_before + (specialized ? 3 : 1));
    check(halide_vulkan_device_sync(nullptr, &a) == 0);
    check(submissions == 1 && waits == 2);
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_a[0] == 5 + 5 * t && host_b[0] == 2 + 2 * t);

    // Reuse a variant after synchronization without recompiling it.
    const int maps_before_reuse = maps, unmaps_before_reuse = unmaps;
    check(run(a, 0) == 0);
    check(pipelines_created == created_before + (specialized ? 3 : 1));
    check(halide_vulkan_device_sync(nullptr, &a) == 0);
    check(maps == maps_before_reuse && unmaps == unmaps_before_reuse);
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0);

    // Upload an independent allocation while A is queued. The upload must
    // finish synchronously while the kernel batch remains unsubmitted.
    const int32_t old_a = host_a[0];
    reset_counts();
    check(run(a, 0) == 0);
    host_b[0] = 11;
    check(halide_vulkan_copy_to_device(nullptr, &b) == 0);
    check(executed_dispatches == 0 && descriptor_sets == 1);
    check(submissions == (mapped ? 0 : 1) && waits == (mapped ? 0 : 1));
    check(copies == (mapped ? 0 : 1) && maps == 1 && unmaps == 1);
    check(flushes == 0);
    check(run(b, 4) == 0);
    check(halide_vulkan_device_sync(nullptr, &b) == 0);
    check(executed_dispatches == 2 && submissions == (mapped ? 1 : 2));
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_a[0] == 2 * old_a + t && host_b[0] == 26 + t);

    // A readback flushes pending work, including cropped buffer offsets.
    halide_buffer_t crop = a;
    crop.device = (uint64_t)allocator->create_crop(nullptr, (MemoryRegion *)a.device, {1});
    check(crop.device != 0);
    reset_counts();
    check(run(crop, 7) == 0);
    check(halide_vulkan_device_release_crop(nullptr, &crop) == 0);
    check(submissions == 0 && waits == 0);
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
    check(executed_dispatches == 1 && host_a[1] == 7 + t);

    // A host write must execute earlier dispatches before overwriting memory.
    reset_counts();
    check(run(a, 9) == 0);
    host_a[0] = 20;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    check(executed_dispatches == 1);
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
    check(host_a[0] == 20);

    reset_counts();
    check(run(a, 2) == 0);
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) == 0);
    check(run(b, 3) == 0);
    check(executed_dispatches == 0 && submissions == 0 && waits == 0 && pools_created == 1);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(executed_dispatches == 2 && host_b[0] == 87 + 3 * t);

    // Copy-only batches must flush before host writes to either endpoint.
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) == 0);
    host_a[0] = 21;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    check(submissions == (mapped ? 1 : 2) && descriptor_sets == 0);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_b[0] == 42 + t);
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) == 0);
    host_b[0] = 22;
    check(halide_vulkan_copy_to_device(nullptr, &b) == 0);
    check(submissions == (mapped ? 1 : 2));
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_b[0] == 22);

    // A released crop remains tracked by its owner until its copy completes.
    crop = a;
    crop.device = (uint64_t)allocator->create_crop(nullptr, (MemoryRegion *)a.device, {1});
    check(crop.device != 0);
    halide_dimension_t one_element = {0, 1, 1, 0};
    crop.dim = &one_element;
    halide_buffer_t dst_crop = b;
    dst_crop.device = (uint64_t)allocator->create_crop(nullptr, (MemoryRegion *)b.device, {2});
    check(dst_crop.device != 0);
    dst_crop.dim = &one_element;
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &crop, &vulkan_device_interface, &dst_crop) == 0);
    check(halide_vulkan_device_release_crop(nullptr, &crop) == 0);
    check(halide_vulkan_device_release_crop(nullptr, &dst_crop) == 0);
    check(submissions == 0 && waits == 0);
    host_a[1] = 23;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    check(submissions == (mapped ? 1 : 2));
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_b[2] == 7 + t && host_b[1] == 7 + t && host_b[3] == 0);
    host_a[1] = 7 + t;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    host_a[0] = 20;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);

    // Back-to-back copies retain freed intermediates and use one submission.
    halide_buffer_t copy_intermediate = a;
    copy_intermediate.device = 0;
    copy_intermediate.device_interface = nullptr;
    check(halide_vulkan_device_malloc(nullptr, &copy_intermediate) == 0);
    const size_t regions_before_free = allocator->regions_allocated();
    reset_counts();
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &copy_intermediate) == 0);
    check(halide_vulkan_buffer_copy(nullptr, &copy_intermediate, &vulkan_device_interface, &b) == 0);
    check(halide_vulkan_device_free(nullptr, &copy_intermediate) == 0);
    check(copy_intermediate.device == 0 && allocator->regions_allocated() == regions_before_free);
    check(submissions == 0 && waits == 0 && pools_created == 1);
    check(halide_vulkan_device_sync(nullptr, &b) == 0);
    check(submissions == 1 && waits == 1 && pools_destroyed == 1);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_b[0] == 20 && host_b[1] == 7 + t);

    // Reusing a read-only input must preserve the first consumer's input,
    // although its results are written to a different allocation.
    auto run_input = [&](halide_buffer_t &input, halide_buffer_t &output, int32_t scalar) {
        size_t sizes[] = {sizeof(scalar), sizeof(halide_buffer_t *), sizeof(halide_buffer_t *), 0};
        void *args[] = {&scalar, &input, &output};
        int8_t is_buffer[] = {0, 1, 1};
        binding.storage_buffer_count = 2;
        binding.push_constant_bytes = push ? 12 : 0;
        int result = halide_vulkan_run(nullptr, state, "test", 1, 1, 1, 1, 1, 1, 0, sizes, args, is_buffer);
        binding.storage_buffer_count = 1;
        binding.push_constant_bytes = push ? 8 : 0;
        return result;
    };
    int32_t host_c[4] = {};
    halide_buffer_t c = a;
    c.host = (uint8_t *)host_c;
    c.device = 0;
    c.device_interface = nullptr;
    check(halide_vulkan_device_malloc(nullptr, &c) == 0);
    host_a[0] = 30;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    reset_counts();
    check(run_input(a, b, 1) == 0);
    host_a[0] = 40;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    check(executed_dispatches == 1 && descriptor_sets == 0);
    check(run_input(a, c, 2) == 0);
    check(halide_vulkan_device_sync(nullptr, &c) == 0);
    check(executed_dispatches == 2);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(halide_vulkan_copy_to_host(nullptr, &c) == 0);
    check(host_b[0] == 61 + t && host_c[0] == 82 + t);
    check(halide_vulkan_device_free(nullptr, &c) == 0);

    // Parent uploads conflict with references to crops, even after the crop's
    // metadata has been freed. Uploads through crops also conflict with parents.
    crop = a;
    crop.device = (uint64_t)allocator->create_crop(nullptr, (MemoryRegion *)a.device, {1});
    check(crop.device != 0);
    reset_counts();
    check(run_input(crop, b, 3) == 0);
    check(halide_vulkan_device_release_crop(nullptr, &crop) == 0);
    host_a[1] = 50;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    check(executed_dispatches == 1);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_b[0] == 2 * (7 + t) + 3 + t);
    crop = a;
    crop.device = (uint64_t)allocator->create_crop(nullptr, (MemoryRegion *)a.device, {1});
    check(crop.device != 0);
    halide_dimension_t crop_dim = {0, 1, 1, 0};
    int32_t crop_host = 60;
    crop.dim = &crop_dim;
    crop.host = (uint8_t *)&crop_host;
    reset_counts();
    check(run_input(a, b, 4) == 0);
    check(halide_vulkan_copy_to_device(nullptr, &crop) == 0);
    check(executed_dispatches == 1);
    check(halide_vulkan_device_release_crop(nullptr, &crop) == 0);
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
    check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
    check(host_a[1] == 60 && host_b[0] == 84 + t);

    // Freeing an intermediate keeps its allocation and descriptors alive until
    // the whole group has executed, and does not split the group.
    halide_buffer_t intermediate = a;
    intermediate.device = 0;
    intermediate.device_interface = nullptr;
    check(halide_vulkan_device_malloc(nullptr, &intermediate) == 0);
    check(halide_vulkan_copy_to_device(nullptr, &intermediate) == 0);
    reset_counts();
    check(run(intermediate, 4) == 0);
    check(halide_vulkan_device_free(nullptr, &intermediate) == 0);
    check(intermediate.device == 0 && submissions == 0 && waits == 0);
    check(run(a, 5) == 0);
    check(halide_vulkan_device_sync(nullptr, &a) == 0);
    check(submissions == 1 && executed_dispatches == 2 && descriptor_sets == 0);

    // A failed dispatch setup leaves earlier queued work intact.
    reset_counts();
    check(run(a, 6) == 0);
    descriptor_result = VK_ERROR_OUT_OF_POOL_MEMORY;
    check(run(b, 7) != 0);
    descriptor_result = VK_SUCCESS;
    check(descriptor_sets == 1 && submissions == 0);
    check(halide_vulkan_device_sync(nullptr, nullptr) == 0);
    check(submissions == 1 && executed_dispatches == 1 && descriptor_sets == 0);

    // Submission errors are reported at sync, and can be retried without
    // freeing the resources or appending to an ended command buffer.
    reset_counts();
    check(run(a, 8) == 0);
    submit_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    check(halide_vulkan_device_sync(nullptr, nullptr) != 0);
    check(submissions == 1 && waits == 0 && descriptor_sets == 1);
    check(run(b, 9) != 0);
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) != 0);
    int maps_before_retry = maps;
    check(halide_vulkan_copy_to_device(nullptr, &b) != 0);
    check(maps == maps_before_retry && submissions == 2 && waits == 0 && descriptor_sets == 1);
    submit_result = VK_SUCCESS;
    check(halide_vulkan_device_sync(nullptr, nullptr) == 0);
    check(submissions == 3 && executed_dispatches == 1 && descriptor_sets == 0);

    // A failed wait must retain all resources and must not resubmit work.
    reset_counts();
    check(run(a, 10) == 0);
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) == 0);
    wait_result = VK_ERROR_DEVICE_LOST;
    check(halide_vulkan_device_sync(nullptr, nullptr) != 0);
    check(submissions == 1 && descriptor_sets == 1 && pools_destroyed == 0);
    check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) != 0);
    maps_before_retry = maps;
    check(halide_vulkan_copy_to_device(nullptr, &b) != 0);
    check(maps == maps_before_retry && submissions == 1 && descriptor_sets == 1 && pools_destroyed == 0);
    wait_result = VK_SUCCESS;
    check(halide_vulkan_device_sync(nullptr, nullptr) == 0);
    check(submissions == 1 && executed_dispatches == 1 && descriptor_sets == 0 && pools_destroyed == 1);

    // Ending a command buffer can fail before submission; clean it up and
    // allow a subsequent dispatch to start a fresh batch.
    reset_counts();
    check(run(a, 11) == 0);
    end_result = VK_ERROR_OUT_OF_HOST_MEMORY;
    check(halide_vulkan_device_sync(nullptr, nullptr) != 0);
    check(submissions == 0 && descriptor_sets == 0 && pools_destroyed == 1);
    end_result = VK_SUCCESS;
    check(run(a, 12) == 0);
    check(halide_vulkan_device_release(nullptr) == 0);
    check(submissions == 1 && executed_dispatches == 1 && command_batches == nullptr);

    if (mapped) {
        host_a[0] = host_b[0] = 0;
        check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
        check(halide_vulkan_copy_to_device(nullptr, &b) == 0);
        reset_counts();
        check(run(a, 1) == 0 && run(b, 2) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);
        check(pools_created == 1 && pools_destroyed == 1);
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        halide_buffer_t host_buffer = b;
        host_buffer.device = 0;
        host_buffer.device_interface = nullptr;
        check(halide_vulkan_buffer_copy(nullptr, &b, nullptr, &host_buffer) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);
        check(host_a[0] == 1 + t && host_b[0] == 2 + t);

        // A mapped upload through buffer_copy waits only for the pending batch.
        reset_counts();
        check(run(a, 0) == 0);
        halide_buffer_t upload = a;
        upload.device = 0;
        upload.device_interface = nullptr;
        check(halide_vulkan_buffer_copy(nullptr, &upload, &vulkan_device_interface, &a) == 0);
        check(submissions == 1 && waits == 1 && executed_dispatches == 1);

        // The buffer-copy entry point also folds host visibility into a batch.
        reset_counts();
        check(run(b, 3) == 0);
        check(halide_vulkan_buffer_copy(nullptr, &b, nullptr, &host_buffer) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);
        check(host_b[0] == 7 + 3 * t);

        // Copy-only work invalidates readiness, just like a shader dispatch.
        reset_counts();
        check(halide_vulkan_buffer_copy(nullptr, &a, &vulkan_device_interface, &b) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);
        check(host_b[0] == host_a[0]);

        // A device-only flush leaves one host dependency for the first read.
        check(run(a, 4) == 0);
        check(halide_vulkan_device_sync(nullptr, &a) == 0);
        reset_counts();
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);

        // Standalone staging transfers also invalidate the queue-wide dependency.
        // Force a dedicated device-only allocation alongside the mapped buffers.
        MemoryRequest request = {};
        request.size = a.size_in_bytes();
        request.dedicated = true;
        request.properties.usage = MemoryUsage::TransferSrcDst;
        request.properties.caching = MemoryCaching::Uncached;
        request.properties.visibility = MemoryVisibility::DeviceOnly;
        reject_unified_allocation = true;
        MemoryRegion *staged_region = allocator->reserve(nullptr, request);
        check(staged_region != nullptr && !reject_unified_allocation);
        check(!vk_can_copy_mapped(nullptr, allocator, staged_region));
        int32_t staged_host[4] = {1, 2, 3, 4};
        halide_buffer_t staged = a;
        staged.host = (uint8_t *)staged_host;
        staged.device = (uint64_t)staged_region;
        reset_counts();
        check(halide_vulkan_copy_to_device(nullptr, &staged) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(submissions == 2 && waits == 2 && host_barriers == 1);
        reset_counts();
        check(halide_vulkan_copy_to_host(nullptr, &staged) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 2 && waits == 2 && host_barriers == 1);
        halide_buffer_t staged_host_buffer = staged;
        staged_host_buffer.device = 0;
        staged_host_buffer.device_interface = nullptr;
        for (int direction = 0; direction < 2; direction++) {
            reset_counts();
            check(direction ? halide_vulkan_buffer_copy(nullptr, &staged, nullptr, &staged_host_buffer) == 0 :
                              halide_vulkan_buffer_copy(nullptr, &staged_host_buffer, &vulkan_device_interface, &staged) == 0);
            const int submitted = submissions, waited = waits;
            check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
            check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
            check(submissions == submitted + 1 && waits == waited + 1 && host_barriers == 1);
        }
        check(allocator->reclaim(nullptr, staged_region) == 0);

        // Readiness is scoped to the queue and survives release of another one.
        reset_counts();
        test_queue = (VkQueue)2;
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);
        check(halide_vulkan_device_release(nullptr) == 0);
        test_queue = (VkQueue)1;
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 1);

        // Failed submissions retain the barrier, without permitting host reads.
        reset_counts();
        check(run(a, 5) == 0);
        int maps_before = maps;
        submit_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        check(halide_vulkan_copy_to_host(nullptr, &a) != 0);
        check(submissions == 1 && waits == 0 && host_barriers == 1 && maps == maps_before);
        submit_result = VK_SUCCESS;
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 2 && waits == 1 && host_barriers == 1);

        // Failed waits neither mark readiness nor resubmit the command buffer.
        reset_counts();
        check(run(a, 6) == 0);
        maps_before = maps;
        wait_result = VK_ERROR_DEVICE_LOST;
        check(halide_vulkan_copy_to_host(nullptr, &a) != 0);
        check(halide_vulkan_copy_to_host(nullptr, &b) != 0);
        check(submissions == 1 && waits == 2 && host_barriers == 1 && maps == maps_before);
        check(pools_destroyed == 0);
        wait_result = VK_SUCCESS;
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(submissions == 1 && waits == 3 && host_barriers == 1 && pools_destroyed == 1);

        // An already-ended device-only batch needs a separate host dependency.
        reset_counts();
        check(run(a, 7) == 0);
        submit_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        check(halide_vulkan_device_sync(nullptr, &a) != 0);
        submit_result = VK_SUCCESS;
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(submissions == 3 && waits == 2 && host_barriers == 1);

        // An end failure discards the batch; a fresh barrier can still be retried.
        reset_counts();
        check(run(a, 8) == 0);
        maps_before = maps;
        end_result = VK_ERROR_OUT_OF_HOST_MEMORY;
        check(halide_vulkan_copy_to_host(nullptr, &a) != 0);
        check(submissions == 0 && waits == 0 && maps == maps_before);
        end_result = VK_SUCCESS;
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0);
        check(halide_vulkan_copy_to_host(nullptr, &b) == 0);
        check(submissions == 1 && waits == 1 && host_barriers == 2);
        check(halide_vulkan_device_release(nullptr) == 0);
    }

    // Cross a descriptor pool boundary and reuse both pools on the next batch.
    host_a[0] = 0;
    check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
    int pools_before = descriptor_pools_created;
    for (int batch = 0; batch < 2; batch++) {
        reset_counts();
        for (int i = 0; i < 65; i++) {
            check(run(a, -t) == 0);
        }
        check(descriptor_pools == 2 && descriptor_sets == 65);
        check(descriptor_pools_created == pools_before + 2);
        check(maps == (batch == 0 && !push ? 1 : 0) && unmaps == 0);
        check(halide_vulkan_device_sync(nullptr, &a) == 0);
        check(descriptor_sets == 0 && descriptor_pools == 2 && executed_dispatches == 65);
    }
    check(halide_vulkan_copy_to_host(nullptr, &a) == 0 && host_a[0] == 0);
    check(halide_vulkan_device_release(nullptr) == 0);

    // Exercise optional descriptor paths with immutable snapshots per dispatch.
    for (uint32_t extensions = 1; extensions <= 3; extensions++) {
        allocator->set_dispatch_extensions(extensions);
        vk_destroy_shader_binding_pipelines(nullptr, allocator, &binding);
        descriptor_layout = VK_NULL_HANDLE;
        module.pipeline_layout = VK_NULL_HANDLE;
        pools_before = descriptor_pools_created;
        const int updates_before = template_updates, pushes_before = descriptor_pushes;
        host_a[0] = 0;
        check(halide_vulkan_copy_to_device(nullptr, &a) == 0);
        check(run(a, 1) == 0 && run(a, 2) == 0);
        check(descriptor_sets == (binding.push_descriptors ? 0 : 2));
        check(descriptor_pools_created == pools_before + (binding.push_descriptors ? 0 : 1));
        check(template_updates == updates_before + ((extensions & 2) ? 2 : 0));
        check(descriptor_pushes == pushes_before + (binding.push_descriptors ? 2 : 0));
        check(halide_vulkan_copy_to_host(nullptr, &a) == 0 && host_a[0] == 4 + 3 * t);
        check(halide_vulkan_device_release(nullptr) == 0);
        if (binding.descriptor_update_template) {
            vkDestroyDescriptorUpdateTemplate(test_device, binding.descriptor_update_template, nullptr);
            binding.descriptor_update_template = VK_NULL_HANDLE;
        }
        check(descriptor_templates == 0);
    }
    allocator->set_dispatch_extensions(0);

    check(halide_vulkan_device_free(nullptr, &a) == 0);
    check(halide_vulkan_device_free(nullptr, &b) == 0);
    halide_vulkan_finalize_kernels(nullptr, state);
    auto destroy_entry = [&](VulkanCompilationCacheEntry *e) {
        if (!e) {
            return;
        }
        check(e == &entry);
        vk_destroy_shader_binding_pipelines(nullptr, allocator, &binding);
    };
    compilation_cache.delete_context(nullptr, test_device, destroy_entry);
    check(pipelines == 0 && descriptor_sets == 0 && pending_commands == nullptr);
    check(descriptor_pools == 0 && dispatch_arenas == nullptr);
    // Collect cached blocks through the allocator's callback context before
    // destroying the allocator itself.
    allocator->collect(nullptr);
    check(allocator->blocks_allocated() == 0);
    check(VulkanMemoryAllocator::destroy(nullptr, allocator) == 0);
    allocator = nullptr;
    test_device = (VkDevice)1;
    halide_reuse_device_allocations(nullptr, false);
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
    vkCreateDescriptorPool = create_descriptor_pool;
    vkCreateDescriptorSetLayout = create_descriptor_layout;
    vkCreatePipelineLayout = create_pipeline_layout;
    vkDestroyDescriptorPool = destroy_descriptor_pool;
    vkAllocateDescriptorSets = allocate_descriptors;
    vkResetDescriptorPool = reset_descriptor_pool;
    vkUpdateDescriptorSets = update_descriptors;
    vkCreateComputePipelines = create_pipelines;
    vkDestroyPipeline = destroy_pipeline;
    vkCmdBindPipeline = bind_pipeline;
    vkCmdBindDescriptorSets = bind_descriptors;
    vkCmdDispatch = dispatch;
    vkCmdPushConstants = push_constants;
    vkCreateDescriptorUpdateTemplate = create_descriptor_template;
    vkDestroyDescriptorUpdateTemplate = destroy_descriptor_template;
    vkUpdateDescriptorSetWithTemplate = update_with_template;
    vkCmdPushDescriptorSetKHR = push_descriptors;
    vkCmdPushDescriptorSetWithTemplateKHR = push_with_template;
    Halide::Runtime::Internal::Vulkan::acquire_context = acquire;
    Halide::Runtime::Internal::Vulkan::release_context = release;
    halide_set_error_handler(ignore_error);
    halide_reuse_device_allocations(nullptr, false);

    test_scalar_argument_alignment();
    test_transfers(true, true);
    test_transfers(false, true);
    test_transfers(true, false);
    test_transfers(true, true, true);
    test_transfers(true, true, false, true);
    test_command_grouping(false, true);
    test_command_grouping(true, true);
    test_command_grouping(false, false);
    test_command_grouping(true, false);
    test_command_grouping(false, true, false);
    test_command_grouping(true, true, false);
    test_command_grouping(false, true, true, true);
    test_command_grouping(false, true, true, false, true);
    test_command_grouping(true, false, true, false, true);
    halide_print(nullptr, "Success!\n");
    return 0;
}
