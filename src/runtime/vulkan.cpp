#include "HalideRuntimeVulkan.h"

#include "device_buffer_utils.h"
#include "device_interface.h"
#include "runtime_internal.h"
#include "vulkan_context.h"
#include "vulkan_extensions.h"
#include "vulkan_internal.h"
#include "vulkan_memory.h"
#include "vulkan_resources.h"

using namespace Halide::Runtime::Internal::Vulkan;

// --------------------------------------------------------------------------

extern "C" {

// --------------------------------------------------------------------------

// The default implementation of halide_acquire_vulkan_context uses
// the global pointers above, and serializes access with a spin lock.
// Overriding implementations of acquire/release must implement the
// following behavior:

//  - halide_acquire_vulkan_context should always store a valid
//   instance/device/queue in the corresponding out parameters,
//   or return an error code.
// - A call to halide_acquire_vulkan_context is followed by a matching
//   call to halide_release_vulkan_context. halide_acquire_vulkan_context
//   should block while a previous call (if any) has not yet been
//   released via halide_release_vulkan_context.
WEAK int halide_default_vulkan_acquire_context(void *user_context,
                                               halide_vulkan_memory_allocator **allocator,
                                               VkInstance *instance,
                                               VkDevice *device,
                                               VkPhysicalDevice *physical_device,
                                               VkQueue *queue,
                                               uint32_t *queue_family_index,
                                               VkDebugUtilsMessengerEXT *messenger,
                                               bool create) {
#ifdef DEBUG_RUNTIME
    halide_start_clock(user_context);
#endif
    halide_debug_assert(user_context, instance != nullptr);
    halide_debug_assert(user_context, device != nullptr);
    halide_debug_assert(user_context, queue != nullptr);
    halide_debug_assert(user_context, &thread_lock != nullptr);
    halide_mutex_lock(&thread_lock);

    // If the context has not been initialized, initialize it now.
    if ((cached_instance == nullptr) && create) {
        int error_code = vk_create_context(user_context,
                                           reinterpret_cast<VulkanMemoryAllocator **>(&cached_allocator),
                                           &cached_instance,
                                           &cached_device,
                                           &cached_physical_device,
                                           &cached_queue,
                                           &cached_queue_family_index,
                                           &cached_messenger);
        if (error_code != halide_error_code_success) {
            debug(user_context) << "halide_vulkan_acquire_context: FAILED to create context!\n";
            halide_mutex_unlock(&thread_lock);
            return error_code;
        }
    }

    *allocator = cached_allocator;
    *instance = cached_instance;
    *device = cached_device;
    *physical_device = cached_physical_device;
    *queue = cached_queue;
    *queue_family_index = cached_queue_family_index;
    *messenger = cached_messenger;
    return halide_error_code_success;
}

WEAK int halide_default_vulkan_release_context(void *user_context, VkInstance instance, VkDevice device, VkQueue queue, VkDebugUtilsMessengerEXT messenger) {
    halide_mutex_unlock(&thread_lock);
    return halide_error_code_success;
}

}  // extern "C"

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Vulkan {

WEAK halide_vulkan_acquire_context_t acquire_context = halide_default_vulkan_acquire_context;
WEAK halide_vulkan_release_context_t release_context = halide_default_vulkan_release_context;

// Context access serializes a queue. The extra lock protects this list when
// custom context handlers return different devices or queues concurrently.
struct VulkanCommandResources {
    VulkanCommandResources *next;
    size_t buffer_count;
    MemoryRegion **buffers;
};

struct VulkanDescriptorPool {
    VulkanDescriptorPool *next;
    VkDescriptorPool pool;
    uint32_t uniform_count;
    uint32_t storage_count;
    uint32_t used;
};

struct VulkanArgumentPage {
    VulkanArgumentPage *next;
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint8_t *mapped;
    size_t capacity;
    size_t used;
};

// One arena per allocator/queue. Only its current command batch uses its
// resources; cursors and descriptor pools are reset after that batch completes.
struct VulkanDispatchArena {
    VulkanDispatchArena *next;
    VulkanMemoryAllocator *allocator;
    VkQueue queue;
    VulkanDescriptorPool *pools;
    VulkanArgumentPage *pages;
    bool host_reads_ready;  // A completed host dependency, retained across batches.
};

struct VulkanDeferredFree {
    VulkanDeferredFree *next;
    MemoryRegion *region;
    bool reuse;
};

struct VulkanCommandBatch {
    VulkanCommandBatch *next;
    VulkanMemoryAllocator *allocator;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VulkanCommandResources *resources;
    VulkanDispatchArena *arena;
    VulkanDeferredFree *frees;
    bool ended;
    bool submitted;
    bool host_read_barrier;
};

WEAK VulkanCommandBatch *command_batches = nullptr;
WEAK VulkanDispatchArena *dispatch_arenas = nullptr;
WEAK halide_mutex command_batches_mutex;

namespace {

void vk_destroy_command_resources(void *user_context, VulkanMemoryAllocator *allocator,
                                  VulkanCommandResources *resources) {
    vk_host_free(user_context, resources, allocator->callbacks());
}

VulkanDispatchArena *vk_find_dispatch_arena(VulkanMemoryAllocator *allocator, VkQueue queue) {
    for (auto *arena = dispatch_arenas; arena; arena = arena->next) {
        if (arena->allocator == allocator && arena->queue == queue) {
            return arena;
        }
    }
    return nullptr;
}

void vk_invalidate_host_reads(VulkanMemoryAllocator *allocator, VkQueue queue) {
    ScopedMutexLock lock(&command_batches_mutex);
    if (auto *arena = vk_find_dispatch_arena(allocator, queue)) {
        arena->host_reads_ready = false;
    }
}

int vk_get_dispatch_arena(void *user_context, VulkanCommandBatch *batch) {
    if (batch->arena) {
        return halide_error_code_success;
    }
    if (auto *arena = vk_find_dispatch_arena(batch->allocator, batch->queue)) {
        batch->arena = arena;
        return halide_error_code_success;
    }
    auto *arena = (VulkanDispatchArena *)vk_host_malloc(user_context, sizeof(VulkanDispatchArena), 0,
                                                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, batch->allocator->callbacks());
    if (!arena) {
        return halide_error_code_out_of_memory;
    }
    *arena = {dispatch_arenas, batch->allocator, batch->queue, nullptr, nullptr, false};
    dispatch_arenas = batch->arena = arena;
    return halide_error_code_success;
}

void vk_destroy_argument_page(void *user_context, VulkanMemoryAllocator *allocator, VulkanArgumentPage *page) {
    VkDevice device = allocator->current_device();
    if (page->mapped) {
        vkUnmapMemory(device, page->memory);
    }
    if (page->buffer) {
        vkDestroyBuffer(device, page->buffer, allocator->callbacks());
    }
    if (page->memory) {
        vkFreeMemory(device, page->memory, allocator->callbacks());
    }
    vk_host_free(user_context, page, allocator->callbacks());
}

// These allocations must be separate from the buffer allocator's blocks:
// persistent mappings must not collide with transient upload/readback mappings.
int vk_allocate_argument_slice(void *user_context, VulkanDispatchArena *arena, size_t size,
                               VkDescriptorBufferInfo *info, uint8_t **mapped) {
    VulkanMemoryAllocator *allocator = arena->allocator;
    VkPhysicalDeviceLimits limits = allocator->current_physical_device_limits();
    if (size > limits.maxUniformBufferRange) {
        return halide_error_code_incompatible_device_interface;
    }
    size_t alignment = limits.minUniformBufferOffsetAlignment;
    alignment = alignment ? alignment : 1;
    for (auto *page = arena->pages; page; page = page->next) {
        size_t offset = aligned_offset(page->used, alignment);
        if (offset <= page->capacity && size <= page->capacity - offset) {
            *info = {page->buffer, offset, size};
            *mapped = page->mapped + offset;
            page->used = offset + size;
            return halide_error_code_success;
        }
    }

    auto *page = (VulkanArgumentPage *)vk_host_malloc(user_context, sizeof(VulkanArgumentPage), 0,
                                                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, allocator->callbacks());
    if (!page) {
        return halide_error_code_out_of_memory;
    }
    memset(page, 0, sizeof(*page));
    page->capacity = size > 65536 ? size : 65536;
    VkDevice device = allocator->current_device();
    VkBufferCreateInfo buffer_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = page->capacity;
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VkResult result = vkCreateBuffer(device, &buffer_info, allocator->callbacks(), &page->buffer);
    if (result == VK_SUCCESS) {
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, page->buffer, &requirements);
        VkPhysicalDeviceMemoryProperties properties;
        vkGetPhysicalDeviceMemoryProperties(allocator->current_physical_device(), &properties);
        result = VK_ERROR_FEATURE_NOT_PRESENT;
        const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
            if ((requirements.memoryTypeBits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & required) == required) {
                VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                alloc_info.allocationSize = requirements.size;
                alloc_info.memoryTypeIndex = i;
                result = vkAllocateMemory(device, &alloc_info, allocator->callbacks(), &page->memory);
                if (result == VK_SUCCESS) {
                    break;
                }
            }
        }
        if (result == VK_SUCCESS) {
            result = vkBindBufferMemory(device, page->buffer, page->memory, 0);
        }
        if (result == VK_SUCCESS) {
            result = vkMapMemory(device, page->memory, 0, requirements.size, 0, (void **)&page->mapped);
        }
    }
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: Failed to allocate argument page: " << vk_get_error_name(result) << "\n";
        vk_destroy_argument_page(user_context, allocator, page);
        return halide_error_code_generic_error;
    }
    page->used = size;
    page->next = arena->pages;
    arena->pages = page;
    *info = {page->buffer, 0, size};
    *mapped = page->mapped;
    return halide_error_code_success;
}

int vk_allocate_dispatch_descriptor_set(void *user_context, VulkanDispatchArena *arena,
                                        VulkanShaderBinding *binding, VkDescriptorSetLayout layout, VkDescriptorSet *set) {
    constexpr uint32_t sets_per_pool = 64;
    for (auto *pool = arena->pools; pool; pool = pool->next) {
        if (pool->used < sets_per_pool && pool->uniform_count == binding->uniform_buffer_count && pool->storage_count == binding->storage_buffer_count) {
            int status = vk_create_descriptor_set(user_context, arena->allocator, layout, pool->pool, set);
            if (status == halide_error_code_success) {
                pool->used++;
            }
            return status;
        }
    }
    auto *pool = (VulkanDescriptorPool *)vk_host_malloc(user_context, sizeof(VulkanDescriptorPool), 0,
                                                        VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, arena->allocator->callbacks());
    if (!pool) {
        return halide_error_code_out_of_memory;
    }
    *pool = {nullptr, VK_NULL_HANDLE, binding->uniform_buffer_count, binding->storage_buffer_count, 0};
    int status = vk_create_descriptor_pool(user_context, arena->allocator, pool->uniform_count, pool->storage_count, &pool->pool, sets_per_pool);
    if (status != halide_error_code_success) {
        vk_host_free(user_context, pool, arena->allocator->callbacks());
        return status;
    }
    pool->next = arena->pools;
    arena->pools = pool;
    status = vk_create_descriptor_set(user_context, arena->allocator, layout, pool->pool, set);
    if (status == halide_error_code_success) {
        pool->used++;
    }
    return status;
}

void vk_reset_dispatch_arena(void *user_context, VulkanDispatchArena *arena) {
    if (!arena) {
        return;
    }
    for (auto *page = arena->pages; page; page = page->next) {
        page->used = 0;
    }
    auto **slot = &arena->pools;
    while (*slot) {
        auto *pool = *slot;
        if (vkResetDescriptorPool(arena->allocator->current_device(), pool->pool, 0) == VK_SUCCESS) {
            pool->used = 0;
            slot = &pool->next;
        } else {
            *slot = pool->next;
            vk_destroy_descriptor_pool(user_context, arena->allocator, pool->pool);
            vk_host_free(user_context, pool, arena->allocator->callbacks());
        }
    }
}

void vk_destroy_dispatch_arena(void *user_context, VulkanMemoryAllocator *allocator, VkQueue queue) {
    auto **slot = &dispatch_arenas;
    while (*slot) {
        auto *arena = *slot;
        if (arena->allocator != allocator || arena->queue != queue) {
            slot = &arena->next;
            continue;
        }
        *slot = arena->next;
        while (arena->pools) {
            auto *pool = arena->pools;
            arena->pools = pool->next;
            vk_destroy_descriptor_pool(user_context, allocator, pool->pool);
            vk_host_free(user_context, pool, allocator->callbacks());
        }
        while (arena->pages) {
            auto *page = arena->pages;
            arena->pages = page->next;
            vk_destroy_argument_page(user_context, allocator, page);
        }
        vk_host_free(user_context, arena, allocator->callbacks());
    }
}

struct ScopedVulkanCommandResources {
    void *user_context;
    VulkanMemoryAllocator *allocator;
    VulkanCommandResources *resources;

    ~ScopedVulkanCommandResources() {
        if (resources) {
            vk_destroy_command_resources(user_context, allocator, resources);
        }
    }
};

VulkanCommandBatch **vk_find_command_batch(VulkanMemoryAllocator *allocator, VkQueue queue) {
    VulkanCommandBatch **batch = &command_batches;
    while (*batch && ((*batch)->allocator != allocator || (*batch)->queue != queue)) {
        batch = &(*batch)->next;
    }
    return batch;
}

void vk_destroy_command_batch(void *user_context, VulkanCommandBatch *batch) {
    VulkanMemoryAllocator *allocator = batch->allocator;
    if (batch->command_pool) {
        // Destroy command buffers before any objects referenced by them.
        vk_destroy_command_pool(user_context, allocator, batch->command_pool);
    }
    while (batch->resources) {
        VulkanCommandResources *resources = batch->resources;
        batch->resources = resources->next;
        vk_destroy_command_resources(user_context, allocator, resources);
    }
    vk_reset_dispatch_arena(user_context, batch->arena);
    while (batch->frees) {
        VulkanDeferredFree *pending = batch->frees;
        batch->frees = pending->next;
        if (pending->reuse) {
            allocator->release(user_context, pending->region);
        } else {
            allocator->reclaim(user_context, pending->region);
        }
        vk_host_free(user_context, pending, allocator->callbacks());
    }
    vk_host_free(user_context, batch, allocator->callbacks());
}

// Called with both the context and command_batches_mutex held. Failed waits
// retain the submitted batch, so its resources are never freed while in use.
int vk_flush_command_batch_locked(void *user_context, VulkanMemoryAllocator *allocator, VkQueue queue,
                                  bool host_read = false) {
    VulkanCommandBatch **slot = vk_find_command_batch(allocator, queue);
    VulkanCommandBatch *batch = *slot;
    if (!batch) {
        return halide_error_code_success;
    }
    if (!batch->ended) {
        // Make writes visible to subsequent device work and, for readback, the host.
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        VkPipelineStageFlags destination_stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        if (host_read) {
            barrier.dstAccessMask |= VK_ACCESS_HOST_READ_BIT;
            destination_stages |= VK_PIPELINE_STAGE_HOST_BIT;
        }
        vkCmdPipelineBarrier(batch->command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             destination_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        batch->host_read_barrier = host_read;
        VkResult result = vkEndCommandBuffer(batch->command_buffer);
        if (result != VK_SUCCESS) {
            error(user_context) << "Vulkan: vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
            *slot = batch->next;
            vk_destroy_command_batch(user_context, batch);
            return halide_error_code_generic_error;
        }
        batch->ended = true;
    }
    if (!batch->submitted) {
        int error_code = vk_submit_command_buffer(user_context, queue, batch->command_buffer);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
        batch->submitted = true;
    }
    VkResult result = vkQueueWaitIdle(queue);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_generic_error;
    }
    *slot = batch->next;
    if (batch->host_read_barrier) {
        batch->arena->host_reads_ready = true;
    }
    vk_destroy_command_batch(user_context, batch);
    return halide_error_code_success;
}

int vk_flush_command_batch(void *user_context, VulkanMemoryAllocator *allocator, VkQueue queue) {
    ScopedMutexLock lock(&command_batches_mutex);
    return vk_flush_command_batch_locked(user_context, allocator, queue);
}

int vk_prepare_upload(void *user_context, const VulkanContext &ctx, MemoryRegion *region) {
    ScopedMutexLock lock(&command_batches_mutex);
    VulkanCommandBatch *batch = *vk_find_command_batch(ctx.allocator, ctx.queue);
    if (!batch) {
        return halide_error_code_success;
    }
    // Retry failed submissions/waits before allowing any host writes. A
    // submitted batch may still be accessing the allocator's memory blocks.
    if (batch->ended) {
        return vk_flush_command_batch_locked(user_context, ctx.allocator, ctx.queue);
    }
    MemoryRegion *owner = ctx.allocator->owner_of(user_context, region);
    for (VulkanCommandResources *resources = batch->resources; resources; resources = resources->next) {
        for (size_t i = 0; i < resources->buffer_count; i++) {
            if (resources->buffers[i] == owner) {
                return vk_flush_command_batch_locked(user_context, ctx.allocator, ctx.queue);
            }
        }
    }
    // Uploading an independent buffer can precede this unsubmitted batch.
    return halide_error_code_success;
}

int vk_get_command_batch(void *user_context, const VulkanContext &ctx, VulkanCommandBatch **out) {
    VulkanCommandBatch **slot = vk_find_command_batch(ctx.allocator, ctx.queue);
    if (*slot) {
        // A failed submit/wait must be retried by a synchronizing operation.
        if ((*slot)->ended) {
            return halide_error_code_generic_error;
        }
        (*slot)->arena->host_reads_ready = false;
        *out = *slot;
        return halide_error_code_success;
    }
    auto *batch = (VulkanCommandBatch *)vk_host_malloc(user_context, sizeof(VulkanCommandBatch), 0,
                                                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, ctx.allocator->callbacks());
    if (!batch) {
        return halide_error_code_out_of_memory;
    }
    memset(batch, 0, sizeof(*batch));
    batch->allocator = ctx.allocator;
    batch->queue = ctx.queue;
    int error_code = vk_create_command_pool(user_context, ctx.allocator, ctx.queue_family_index, &batch->command_pool);
    if (error_code == halide_error_code_success) {
        error_code = vk_create_command_buffer(user_context, ctx.allocator, batch->command_pool, &batch->command_buffer);
    }
    if (error_code == halide_error_code_success) {
        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkResult result = vkBeginCommandBuffer(batch->command_buffer, &begin_info);
        if (result != VK_SUCCESS) {
            error(user_context) << "Vulkan: vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
            error_code = halide_error_code_generic_error;
        }
    }
    if (error_code == halide_error_code_success) {
        error_code = vk_get_dispatch_arena(user_context, batch);
    }
    if (error_code != halide_error_code_success) {
        vk_destroy_command_batch(user_context, batch);
        return error_code;
    }
    *slot = batch;
    batch->arena->host_reads_ready = false;
    *out = batch;
    return halide_error_code_success;
}

// Readback shares one submission and wait with any pending work. The global
// memory barrier covers later reads of other allocations on this queue too.
int vk_prepare_host_read(void *user_context, const VulkanContext &ctx) {
    ScopedMutexLock lock(&command_batches_mutex);
    VulkanCommandBatch *batch = *vk_find_command_batch(ctx.allocator, ctx.queue);
    if (batch && batch->ended && !batch->host_read_barrier) {
        // A previous submit or wait failed after ending a device-only batch.
        // Finish it before recording the missing host dependency.
        int error_code = vk_flush_command_batch_locked(user_context, ctx.allocator, ctx.queue);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
        batch = nullptr;
    }
    if (!batch) {
        auto *arena = vk_find_dispatch_arena(ctx.allocator, ctx.queue);
        if (arena && arena->host_reads_ready) {
            return halide_error_code_success;
        }
        int error_code = vk_get_command_batch(user_context, ctx, &batch);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
    }
    return vk_flush_command_batch_locked(user_context, ctx.allocator, ctx.queue, true);
}

int vk_enqueue_device_copy(void *user_context, const VulkanContext &ctx,
                           halide_buffer_t *src, halide_buffer_t *dst, device_copy copy) {
    ScopedMutexLock lock(&command_batches_mutex);
    auto *src_region = reinterpret_cast<MemoryRegion *>(src->device);
    auto *dst_region = reinterpret_cast<MemoryRegion *>(dst->device);
    MemoryRegion *src_owner = ctx.allocator->owner_of(user_context, src_region);
    MemoryRegion *dst_owner = ctx.allocator->owner_of(user_context, dst_region);
    if (!src_owner || !dst_owner || !src_owner->handle || !dst_owner->handle) {
        return halide_error_code_internal_error;
    }

    // Track both allocation owners so later host writes flush the batch, even
    // when a crop's metadata has been released after recording the copy.
    size_t resources_size = sizeof(VulkanCommandResources) + 2 * sizeof(MemoryRegion *);
    auto *resources = (VulkanCommandResources *)vk_host_malloc(user_context, resources_size, 0,
                                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, ctx.allocator->callbacks());
    if (!resources) {
        return halide_error_code_out_of_memory;
    }
    memset(resources, 0, sizeof(*resources));
    resources->buffers = reinterpret_cast<MemoryRegion **>(resources + 1);
    resources->buffer_count = 2;
    resources->buffers[0] = src_owner;
    resources->buffers[1] = dst_owner;
    ScopedVulkanCommandResources scoped_resources = {user_context, ctx.allocator, resources};

    VulkanCommandBatch *batch = nullptr;
    int error_code = vk_get_command_batch(user_context, ctx, &batch);
    if (error_code != halide_error_code_success) {
        return error_code;
    }

    // Preserve dependencies on earlier dispatches and transfers. Later
    // dispatches already make transfer writes visible to shader accesses.
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(batch->command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    uint64_t src_offset = copy.src_begin + src_region->indexing.offset * src->type.bytes();
    uint64_t dst_offset = copy.dst_begin + dst_region->indexing.offset * dst->type.bytes();
    copy.src = (uint64_t)src_owner->handle;
    copy.dst = (uint64_t)dst_owner->handle;
    error_code = vk_do_multidimensional_copy(user_context, batch->command_buffer, copy,
                                             src_offset, dst_offset, src->dimensions, false, false);
    if (error_code != halide_error_code_success) {
        return error_code;
    }
    resources->next = batch->resources;
    batch->resources = resources;
    scoped_resources.resources = nullptr;
    return halide_error_code_success;
}

// Keep freed intermediate buffers allocated until all queued commands finish.
int vk_defer_device_free(void *user_context, const VulkanContext &ctx, MemoryRegion *region) {
    ScopedMutexLock lock(&command_batches_mutex);
    VulkanCommandBatch *batch = *vk_find_command_batch(ctx.allocator, ctx.queue);
    bool reuse = halide_can_reuse_device_allocations(user_context);
    if (!batch) {
        return reuse ? ctx.allocator->release(user_context, region) : ctx.allocator->reclaim(user_context, region);
    }
    auto *pending = (VulkanDeferredFree *)vk_host_malloc(user_context, sizeof(VulkanDeferredFree), 0,
                                                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, ctx.allocator->callbacks());
    if (!pending) {
        return halide_error_code_out_of_memory;
    }
    *pending = {batch->frees, region, reuse};
    batch->frees = pending;
    return halide_error_code_success;
}

bool vk_can_copy_mapped(void *user_context, VulkanMemoryAllocator *allocator, MemoryRegion *region) {
    const VulkanMemoryAllocation *allocation = allocator->allocation_for(user_context, region);
    const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return allocation && ((allocation->properties & required) == required);
}

// Coherent device-local memory can be copied directly, without a staging buffer or
// issuing a device-to-device copy. The context lock serializes mappings of the
// same memory block. Callers synchronize uploads with vk_prepare_upload or
// vk_flush_command_batch, and reads with vk_prepare_host_read.
int vk_copy_mapped(void *user_context, const VulkanContext &ctx, MemoryRegion *region,
                   device_copy copy, bool to_host, uint32_t element_size) {
    const VulkanMemoryAllocation *allocation = ctx.allocator->allocation_for(user_context, region);
    // Map the whole block to include suballocations and cropped buffers.
    void *mapped = nullptr;
    VkResult result = vkMapMemory(ctx.device, allocation->memory, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkMapMemory returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_device_buffer_copy_failed;
    }
    MemoryRegion *owner = ctx.allocator->owner_of(user_context, region);
    uint64_t device_ptr = (uint64_t)mapped + owner->allocation.offset +
                          (int64_t)region->indexing.offset * element_size;
    if (to_host) {
        copy.src = device_ptr;
    } else {
        copy.dst = device_ptr;
    }
    copy_memory(copy, user_context);
    vkUnmapMemory(ctx.device, allocation->memory);
    // Queue submission makes these host writes visible to subsequent device
    // operations; no additional host-to-device pipeline barrier is needed.
    return halide_error_code_success;
}

}  // namespace

}  // namespace Vulkan
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

extern "C" {

WEAK int halide_vulkan_acquire_context(void *user_context,
                                       halide_vulkan_memory_allocator **allocator,
                                       VkInstance *instance,
                                       VkDevice *device,
                                       VkPhysicalDevice *physical_device,
                                       VkQueue *queue,
                                       uint32_t *queue_family_index,
                                       VkDebugUtilsMessengerEXT *messenger,
                                       bool create) {
    return Vulkan::acquire_context(user_context, allocator, instance, device,
                                   physical_device, queue, queue_family_index, messenger, create);
}

WEAK halide_vulkan_acquire_context_t halide_set_vulkan_acquire_context(halide_vulkan_acquire_context_t handler) {
    halide_vulkan_acquire_context_t result = Vulkan::acquire_context;
    Vulkan::acquire_context = handler;
    return result;
}

WEAK int halide_vulkan_release_context(void *user_context, VkInstance instance, VkDevice device, VkQueue queue, VkDebugUtilsMessengerEXT messenger) {
    return Vulkan::release_context(user_context, instance, device, queue, messenger);
}

WEAK halide_vulkan_release_context_t halide_set_vulkan_release_context(halide_vulkan_release_context_t handler) {
    halide_vulkan_release_context_t result = Vulkan::release_context;
    Vulkan::release_context = handler;
    return result;
}

WEAK bool halide_vulkan_is_initialized() {
    halide_mutex_lock(&thread_lock);
    bool is_initialized = (cached_instance != nullptr) && (cached_device != nullptr);
    halide_mutex_unlock(&thread_lock);
    return is_initialized;
}

WEAK int halide_vulkan_device_free(void *user_context, halide_buffer_t *halide_buffer) {
    debug(user_context)
        << "halide_vulkan_device_free (user_context: " << user_context
        << ", halide_buffer: " << halide_buffer << ")\n";

    // halide_vulkan_device_free, at present, can be exposed to clients and they
    // should be allowed to call halide_vulkan_device_free on any halide_buffer_t
    // including ones that have never been used with a GPU.
    if (halide_buffer->device == 0) {
        return halide_error_code_success;
    }

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // get the allocated region for the device
    MemoryRegion *device_region = reinterpret_cast<MemoryRegion *>(halide_buffer->device);
    MemoryRegion *memory_region = ctx.allocator->owner_of(user_context, device_region);
    if (ctx.allocator && memory_region && memory_region->handle) {
        int error_code = vk_defer_device_free(user_context, ctx, memory_region);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
    }
    halide_buffer->device = 0;
    halide_buffer->device_interface->impl->release_module();
    halide_buffer->device_interface = nullptr;

#ifdef DEBUG_RUNTIME
    debug(user_context) << "Vulkan: Released memory for device region ("
                        << "user_context: " << user_context << ", "
                        << "buffer: " << halide_buffer << ", "
                        << "size_in_bytes: " << (uint64_t)device_region->allocation.size << ")\n";

    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return halide_error_code_success;
}

WEAK int halide_vulkan_compute_capability(void *user_context, int *major, int *minor) {
    debug(user_context) << " halide_vulkan_compute_capability (user_context: " << user_context << ")\n";
    return vk_find_compute_capability(user_context, major, minor);
}

WEAK int halide_vulkan_initialize_kernels(void *user_context, void **state_ptr, const char *src, int size) {
    debug(user_context)
        << "halide_vulkan_init_kernels (user_context: " << user_context
        << ", state_ptr: " << state_ptr
        << ", program: " << (void *)src
        << ", size: " << size << "\n";

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif
    debug(user_context) << "halide_vulkan_initialize_kernels got compilation_cache mutex.\n";

    VulkanCompilationCacheEntry *cache_entry = nullptr;
    if (!compilation_cache.kernel_state_setup(user_context, state_ptr, ctx.device, cache_entry,
                                              Halide::Runtime::Internal::Vulkan::vk_compile_kernel_module,
                                              user_context, ctx.allocator, src, size)) {
        error(user_context) << "Vulkan: Failed to setup compilation cache!\n";
        return halide_error_code_generic_error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return halide_error_code_success;
}

WEAK void halide_vulkan_finalize_kernels(void *user_context, void *state_ptr) {
    debug(user_context)
        << "halide_vulkan_finalize_kernels (user_context: " << user_context
        << ", state_ptr: " << state_ptr << "\n";

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    VulkanContext ctx(user_context);
    if (ctx.error == halide_error_code_success) {
        compilation_cache.release_hold(user_context, ctx.device, state_ptr);
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif
}

// Used to generate correct timings when tracing
WEAK int halide_vulkan_device_sync(void *user_context, halide_buffer_t *) {
    debug(user_context) << "halide_vulkan_device_sync (user_context: " << user_context << ")\n";

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    ScopedMutexLock lock(&command_batches_mutex);
    bool has_batch = *vk_find_command_batch(ctx.allocator, ctx.queue) != nullptr;
    int error_code = vk_flush_command_batch_locked(user_context, ctx.allocator, ctx.queue);
    if (error_code != halide_error_code_success) {
        return error_code;
    }
    VkResult result = has_batch ? VK_SUCCESS : vkQueueWaitIdle(ctx.queue);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_generic_error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return halide_error_code_success;
}

WEAK int halide_vulkan_device_release(void *user_context) {
    debug(user_context)
        << "halide_vulkan_device_release (user_context: " << user_context << ")\n";

    VulkanMemoryAllocator *allocator = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family_index = 0;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

    int destroy_status = halide_error_code_success;
    int acquire_status = halide_vulkan_acquire_context(user_context,
                                                       reinterpret_cast<halide_vulkan_memory_allocator **>(&allocator),
                                                       &instance, &device, &physical_device, &queue, &queue_family_index, &messenger, false);

    if (acquire_status == halide_error_code_success) {
        destroy_status = vk_flush_command_batch(user_context, allocator, queue);
        if (destroy_status != halide_error_code_success) {
            halide_vulkan_release_context(user_context, instance, device, queue, messenger);
            return destroy_status;
        }
        {
            ScopedMutexLock lock(&command_batches_mutex);
            vk_destroy_dispatch_arena(user_context, allocator, queue);
        }
        // Destroy the context if we created it
        if ((instance == cached_instance) && (device == cached_device)) {
            destroy_status = vk_destroy_context(user_context, allocator, instance, device, physical_device, queue, messenger);
            cached_allocator = nullptr;
            cached_device = VK_NULL_HANDLE;
            cached_physical_device = VK_NULL_HANDLE;
            cached_queue = VK_NULL_HANDLE;
            cached_queue_family_index = 0;
            cached_instance = VK_NULL_HANDLE;
            cached_messenger = VK_NULL_HANDLE;
        }

        halide_vulkan_release_context(user_context, instance, device, queue, messenger);
    }

    return destroy_status;
}

WEAK int halide_vulkan_device_malloc(void *user_context, halide_buffer_t *buf) {
    debug(user_context)
        << "halide_vulkan_device_malloc (user_context: " << user_context
        << ", buf: " << buf << ")\n";

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

    size_t size = buf->size_in_bytes();
    if (buf->device) {
        MemoryRegion *device_region = (MemoryRegion *)(buf->device);
        if (device_region->allocation.size >= size) {
            debug(user_context) << "Vulkan: Requested allocation for existing device memory ... using existing buffer!\n";
            return halide_error_code_success;
        } else {
            debug(user_context) << "Vulkan: Requested allocation of different size ... reallocating buffer!\n";
            int error_code = vk_defer_device_free(user_context, ctx, ctx.allocator->owner_of(user_context, device_region));
            if (error_code != halide_error_code_success) {
                return error_code;
            }
            buf->device = 0;
        }
    }

    for (int i = 0; i < buf->dimensions; i++) {
        halide_debug_assert(user_context, buf->dim[i].stride >= 0);
    }

#ifdef DEBUG_RUNTIME
    debug(user_context) << "    allocating buffer: ";
    if (buf && buf->dim) {
        debug(user_context) << "extents: ";
        for (int i = 0; i < buf->dimensions; i++) {
            debug(user_context) << buf->dim[i].extent << " ";
        }
        debug(user_context) << "strides: ";
        for (int i = 0; i < buf->dimensions; i++) {
            debug(user_context) << buf->dim[i].stride << " ";
        }
    }
    debug(user_context) << "type: " << buf->type << " "
                        << "size_in_bytes: " << (uint64_t)size << " "
                        << "(or " << (size * 1e-6f) << "MB)\n";

    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // Request device-local memory; the allocator prefers a host-visible type
    // so transfers can bypass staging when the device supports it.
    MemoryRequest request = {0};
    request.size = size;
    request.properties.usage = MemoryUsage::TransferSrcDst;
    request.properties.caching = MemoryCaching::Uncached;
    request.properties.visibility = MemoryVisibility::DeviceOnly;

    // allocate a new region
    MemoryRegion *device_region = ctx.allocator->reserve(user_context, request);
    if ((device_region == nullptr) || (device_region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return halide_error_code_device_malloc_failed;
    }

    buf->device = (uint64_t)device_region;
    buf->device_interface = &vulkan_device_interface;
    buf->device_interface->impl->use_module();

#ifdef DEBUG_RUNTIME
    debug(user_context)
        << "    allocated device region=" << (void *)device_region << "\n"
        << "    containing device buffer=" << (void *)device_region->handle << "\n"
        << "    for halide buffer " << buf << "\n";
#endif

#if 0
    // retrieve the buffer from the region
    VkBuffer *device_buffer = reinterpret_cast<VkBuffer *>(device_region->handle);
    if (device_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve device buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

    ScopedVulkanCommandBufferAndPool cmds(user_context, ctx.allocator, ctx.queue_family_index);
    if (cmds.error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to create command buffer and pool for context!\n";
        return cmds.error_code;
    }

    int error_code = vk_clear_device_buffer(user_context, ctx.allocator, cmds.command_buffer, ctx.queue, *device_buffer);
    if (error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to clear device buffer!\n";
    }
#else
    int error_code = halide_error_code_success;
#endif

#ifdef DEBUG_RUNTIME
    debug(user_context) << "Vulkan: Reserved memory for device region ("
                        << "user_context: " << user_context << ", "
                        << "buffer: " << buf << ", "
                        << "size_in_bytes: " << (uint64_t)size << ")\n";

    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return error_code;
}

WEAK int halide_vulkan_copy_to_device(void *user_context, halide_buffer_t *halide_buffer) {
    debug(user_context)
        << "halide_vulkan_copy_to_device (user_context: " << user_context
        << ", halide_buffer: " << halide_buffer << ")\n";

    // Acquire the context so we can use the command queue.
    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    if ((halide_buffer->host == nullptr) || (halide_buffer->device == 0)) {
        error(user_context) << "Vulkan: Missing host/device pointers for halide buffer!\n";
        return halide_error_code_internal_error;
    }
    MemoryRegion *mapped_region = reinterpret_cast<MemoryRegion *>(halide_buffer->device);
    int flush_error = vk_prepare_upload(user_context, ctx, mapped_region);
    if (flush_error != halide_error_code_success) {
        return flush_error;
    }
    device_copy copy_helper = make_host_to_device_copy(halide_buffer);

    if (vk_can_copy_mapped(user_context, ctx.allocator, mapped_region)) {
        return vk_copy_mapped(user_context, ctx, mapped_region, copy_helper, false, halide_buffer->type.bytes());
    }

    // We construct a staging buffer to copy into from host memory.  Then,
    // we use vkCmdCopyBuffer() to copy from the staging buffer into the
    // the actual device memory.
    MemoryRequest request = {0};
    request.size = halide_buffer->size_in_bytes();
    request.properties.usage = MemoryUsage::TransferSrc;
    request.properties.caching = MemoryCaching::UncachedCoherent;
    request.properties.visibility = MemoryVisibility::HostToDevice;

    // allocate a new region
    MemoryRegion *staging_region = ctx.allocator->reserve(user_context, request);
    if ((staging_region == nullptr) || (staging_region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return halide_error_code_device_malloc_failed;
    }

    // map the region to a host ptr
    uint8_t *stage_host_ptr = (uint8_t *)ctx.allocator->map(user_context, staging_region);
    if (stage_host_ptr == nullptr) {
        error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
        return halide_error_code_internal_error;
    }

    // copy to the (host-visible/coherent) staging buffer
    copy_helper.dst = (uint64_t)(stage_host_ptr);
    copy_memory(copy_helper, user_context);

    // retrieve the buffer from the region
    VkBuffer *staging_buffer = reinterpret_cast<VkBuffer *>(staging_region->handle);
    if (staging_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve staging buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

    // unmap the pointer
    int error_code = ctx.allocator->unmap(user_context, staging_region);
    if (error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to unmap host pointer to device memory!\n";
        return error_code;
    }

    // get the allocated region for the device
    MemoryRegion *device_region = reinterpret_cast<MemoryRegion *>(halide_buffer->device);
    if (device_region == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve device region for buffer!\n";
        return halide_error_code_internal_error;
    }

    MemoryRegion *memory_region = ctx.allocator->owner_of(user_context, device_region);
    if (memory_region == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve memory region for device!\n";
        return halide_error_code_internal_error;
    }

    // retrieve the buffer from the region
    VkBuffer *device_buffer = reinterpret_cast<VkBuffer *>(memory_region->handle);
    if (device_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

#ifdef DEBUG_RUNTIME
    debug(user_context)
        << "  copying into device region=" << (void *)device_region << "\n"
        << "  containing device buffer=" << (void *)device_buffer << "\n"
        << "  from halide buffer=" << halide_buffer << "\n";
#endif

    ScopedVulkanCommandBufferAndPool cmds(user_context, ctx.allocator, ctx.queue_family_index);
    if (cmds.error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to create command buffer and pool!\n";
        return cmds.error_code;
    }

    // begin the command buffer
    VkCommandBufferBeginInfo command_buffer_begin_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // struct type
            nullptr,                                      // pointer to struct extending this
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
            nullptr                                       // pointer to parent command buffer
    };

    VkResult result = vkBeginCommandBuffer(cmds.command_buffer, &command_buffer_begin_info);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    // define the src and dst config
    bool from_host = true;
    bool to_host = false;

    uint64_t src_offset = copy_helper.src_begin;
    uint64_t dst_offset = copy_helper.dst_begin + (device_region->indexing.offset * halide_buffer->type.bytes());

    copy_helper.src = (uint64_t)(staging_buffer);
    copy_helper.dst = (uint64_t)(device_buffer);

    // enqueue the copy operation, using the allocated buffers
    error_code = vk_do_multidimensional_copy(user_context, cmds.command_buffer, copy_helper,
                                             src_offset, dst_offset,
                                             halide_buffer->dimensions,
                                             from_host, to_host);

    if (error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: vk_do_multidimensional_copy failed!\n";
        return error_code;
    }

    // end the command buffer
    result = vkEndCommandBuffer(cmds.command_buffer);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    //// 13. Submit the command buffer to our command queue
    VkSubmitInfo submit_info =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,  // struct type
            nullptr,                        // pointer to struct extending this
            0,                              // wait semaphore count
            nullptr,                        // semaphores
            nullptr,                        // pipeline stages where semaphore waits occur
            1,                              // how many command buffers to execute
            &(cmds.command_buffer),         // the command buffers
            0,                              // number of semaphores to signal
            nullptr                         // the semaphores to signal
    };

    vk_invalidate_host_reads(ctx.allocator, ctx.queue);
    result = vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkQueueSubmit returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    //// 14. Wait until the queue is done with the command buffer
    result = vkQueueWaitIdle(ctx.queue);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    //// 15. Reclaim the staging buffer
    if (halide_can_reuse_device_allocations(user_context)) {
        ctx.allocator->release(user_context, staging_region);
    } else {
        ctx.allocator->reclaim(user_context, staging_region);
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return halide_error_code_success;
}

WEAK int halide_vulkan_copy_to_host(void *user_context, halide_buffer_t *halide_buffer) {
#ifdef DEBUG_RUNTIME
    debug(user_context)
        << "halide_copy_to_host (user_context: " << user_context
        << ", halide_buffer: " << halide_buffer << ")\n";
#endif
    if (halide_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to copy buffer to host ... invalid halide buffer!\n";
        return halide_error_code_copy_to_host_failed;
    }

    // Acquire the context so we can use the command queue. This also avoids multiple
    // redundant calls to enqueue a download when multiple threads are trying to copy
    // the same buffer.
    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif
    if ((halide_buffer->host == nullptr) || (halide_buffer->device == 0)) {
        error(user_context) << "Vulkan: Missing host/device pointers for halide buffer!\n";
        return halide_error_code_internal_error;
    }

    device_copy copy_helper = make_device_to_host_copy(halide_buffer);

    MemoryRegion *mapped_region = reinterpret_cast<MemoryRegion *>(halide_buffer->device);
    if (vk_can_copy_mapped(user_context, ctx.allocator, mapped_region)) {
        int error_code = vk_prepare_host_read(user_context, ctx);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
        return vk_copy_mapped(user_context, ctx, mapped_region, copy_helper, true, halide_buffer->type.bytes());
    }

    int flush_error = vk_flush_command_batch(user_context, ctx.allocator, ctx.queue);
    if (flush_error != halide_error_code_success) {
        return flush_error;
    }

    // This is the inverse of copy_to_device: we create a staging buffer, copy into
    // it, map it so the host can see it, then copy into the host buffer
    MemoryRequest request = {0};
    request.size = halide_buffer->size_in_bytes();
    request.properties.usage = MemoryUsage::TransferDst;
    request.properties.caching = MemoryCaching::UncachedCoherent;
    request.properties.visibility = MemoryVisibility::DeviceToHost;

    // allocate a new region for staging the transfer
    MemoryRegion *staging_region = ctx.allocator->reserve(user_context, request);
    if ((staging_region == nullptr) || (staging_region->handle == nullptr)) {
        error(user_context) << "Vulkan: Failed to allocate device memory!\n";
        return halide_error_code_device_malloc_failed;
    }

    // retrieve the buffer from the region
    VkBuffer *staging_buffer = reinterpret_cast<VkBuffer *>(staging_region->handle);
    if (staging_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve staging buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

    // get the allocated region for the device
    MemoryRegion *device_region = reinterpret_cast<MemoryRegion *>(halide_buffer->device);
    if (device_region == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve device region for buffer!\n";
        return halide_error_code_internal_error;
    }

    MemoryRegion *memory_region = ctx.allocator->owner_of(user_context, device_region);
    if (memory_region == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve memory region for buffer!\n";
        return halide_error_code_internal_error;
    }

    // retrieve the buffer from the region
    VkBuffer *device_buffer = reinterpret_cast<VkBuffer *>(memory_region->handle);
    if (device_buffer == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve buffer for device memory!\n";
        return halide_error_code_internal_error;
    }

#ifdef DEBUG_RUNTIME
    debug(user_context)
        << "  copying from device region=" << (void *)device_region << "\n"
        << "  containing device buffer=" << (void *)device_buffer << "\n"
        << "  into halide buffer=" << halide_buffer << "\n";
#endif

    ScopedVulkanCommandBufferAndPool cmds(user_context, ctx.allocator, ctx.queue_family_index);
    if (cmds.error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to create command buffer and pool!\n";
        return cmds.error_code;
    }

    // begin the command buffer
    VkCommandBufferBeginInfo command_buffer_begin_info =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // struct type
            nullptr,                                      // pointer to struct extending this
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
            nullptr                                       // pointer to parent command buffer
    };

    VkResult result = vkBeginCommandBuffer(cmds.command_buffer, &command_buffer_begin_info);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    // define the src and dst config
    bool from_host = false;
    bool to_host = true;
    uint64_t copy_dst = copy_helper.dst;
    uint64_t src_offset = copy_helper.src_begin + (device_region->indexing.offset * halide_buffer->type.bytes());
    uint64_t dst_offset = copy_helper.dst_begin;

    copy_helper.src = (uint64_t)(device_buffer);
    copy_helper.dst = (uint64_t)(staging_buffer);

    // enqueue the copy operation, using the allocated buffers
    int error_code = vk_do_multidimensional_copy(user_context, cmds.command_buffer, copy_helper,
                                                 src_offset, dst_offset,
                                                 halide_buffer->dimensions,
                                                 from_host, to_host);

    if (error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: vk_do_multidimensional_copy failed!\n";
        return error_code;
    }

    // end the command buffer
    result = vkEndCommandBuffer(cmds.command_buffer);
    if (result != VK_SUCCESS) {
        error(user_context) << "vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
        return result;
    }

    //// 13. Submit the command buffer to our command queue
    VkSubmitInfo submit_info =
        {
            VK_STRUCTURE_TYPE_SUBMIT_INFO,  // struct type
            nullptr,                        // pointer to struct extending this
            0,                              // wait semaphore count
            nullptr,                        // semaphores
            nullptr,                        // pipeline stages where semaphore waits occur
            1,                              // how many command buffers to execute
            &(cmds.command_buffer),         // the command buffers
            0,                              // number of semaphores to signal
            nullptr                         // the semaphores to signal
    };

    vk_invalidate_host_reads(ctx.allocator, ctx.queue);
    result = vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkQueueSubmit returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_copy_to_device_failed;
    }

    //// 14. Wait until the queue is done with the command buffer
    result = vkQueueWaitIdle(ctx.queue);
    if (result != VK_SUCCESS) {
        error(user_context) << "Vulkan: vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
        return halide_error_code_copy_to_device_failed;
    }

    // map the staging region to a host ptr
    uint8_t *stage_host_ptr = (uint8_t *)ctx.allocator->map(user_context, staging_region);
    if (stage_host_ptr == nullptr) {
        error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
        return halide_error_code_copy_to_device_failed;
    }

    // copy to the (host-visible/coherent) staging buffer
    copy_helper.dst = copy_dst;
    copy_helper.src = (uint64_t)(stage_host_ptr);
    copy_memory(copy_helper, user_context);

    // unmap the pointer and reclaim the staging region
    error_code = ctx.allocator->unmap(user_context, staging_region);
    if (error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to umap staging region!\n";
        return error_code;
    }

    if (halide_can_reuse_device_allocations(user_context)) {
        ctx.allocator->release(user_context, staging_region);
    } else {
        ctx.allocator->reclaim(user_context, staging_region);
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return halide_error_code_success;
}

WEAK int halide_vulkan_buffer_copy(void *user_context, struct halide_buffer_t *src,
                                   const struct halide_device_interface_t *dst_device_interface,
                                   struct halide_buffer_t *dst) {
    if (dst->dimensions > MAX_COPY_DIMS) {
        error(user_context) << "Vulkan: Buffer has too many dimensions to copy to/from GPU\n";
        return halide_error_code_buffer_extents_too_large;
    }

    // We only handle copies to Vulkan buffers or to host
    if ((dst_device_interface != nullptr) && (dst_device_interface != &vulkan_device_interface)) {
        error(user_context) << "Vulkan: Unable to copy buffer ... only Vulkan allocated device buffers copying to/from host are supported!\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    if ((src->device_dirty() || src->host == nullptr) && (src->device_interface != &vulkan_device_interface)) {
        // This is handled at the higher level.
        return halide_error_code_incompatible_device_interface;
    }

    bool from_host = (src->device_interface != &vulkan_device_interface) ||
                     (src->device == 0) ||
                     (src->host_dirty() && src->host != nullptr);
    bool to_host = !dst_device_interface;

    if (!(from_host || src->device)) {
        error(user_context) << "Vulkan: halide_vulkan_buffer_copy: invalid copy source\n";
        return halide_error_code_device_buffer_copy_failed;
    }
    if (!(to_host || dst->device)) {
        error(user_context) << "Vulkan: halide_vulkan_buffer_copy: invalid copy destination\n";
        return halide_error_code_device_buffer_copy_failed;
    }

    device_copy copy_helper = make_buffer_copy(src, from_host, dst, to_host);

    int error_code = halide_error_code_success;
    {
        VulkanContext ctx(user_context);
        if (ctx.error != halide_error_code_success) {
            error(user_context) << "Vulkan: Failed to acquire context!\n";
            return ctx.error;
        }

        if (!from_host && !to_host) {
            return vk_enqueue_device_copy(user_context, ctx, src, dst, copy_helper);
        }

        debug(user_context)
            << "halide_vulkan_buffer_copy (user_context: " << user_context
            << ", src: " << src << ", dst: " << dst << ")\n";

#ifdef DEBUG_RUNTIME
        uint64_t t_before = halide_current_time_ns(user_context);
#endif
        if (from_host != to_host) {
            MemoryRegion *mapped_region = reinterpret_cast<MemoryRegion *>(to_host ? src->device : dst->device);
            if (vk_can_copy_mapped(user_context, ctx.allocator, mapped_region)) {
                int error_code = to_host ? vk_prepare_host_read(user_context, ctx) :
                                           vk_flush_command_batch(user_context, ctx.allocator, ctx.queue);
                if (error_code != halide_error_code_success) {
                    return error_code;
                }
                return vk_copy_mapped(user_context, ctx, mapped_region, copy_helper, to_host, src->type.bytes());
            }
        }

        int flush_error = vk_flush_command_batch(user_context, ctx.allocator, ctx.queue);
        if (flush_error != halide_error_code_success) {
            return flush_error;
        }

        MemoryRegion *staging_region = nullptr;
        MemoryRegion *src_buffer_region = nullptr;
        MemoryRegion *dst_buffer_region = nullptr;

        //// wait until the queue is done with the command buffer
        VkResult wait_result = vkQueueWaitIdle(ctx.queue);
        if (wait_result != VK_SUCCESS) {
            error(user_context) << "Vulkan: vkQueueWaitIdle returned " << vk_get_error_name(wait_result) << "\n";
            if (to_host) {
                return halide_error_code_copy_to_host_failed;
            } else {
                return halide_error_code_copy_to_device_failed;
            }
        }

        int error_code = halide_error_code_success;
        if (!from_host && to_host) {
            // Device to Host
            debug(user_context) << " buffer copy from: device to: host\n";

            // Need to make sure all reads and writes to/from source are complete.
            MemoryRequest request = {0};
            request.size = src->size_in_bytes();

            // NOTE: We may re-use this buffer so enable both src and dst
            request.properties.usage = MemoryUsage::TransferSrcDst;
            request.properties.caching = MemoryCaching::UncachedCoherent;
            request.properties.visibility = MemoryVisibility::DeviceToHost;

            // allocate a new region
            staging_region = ctx.allocator->reserve(user_context, request);
            if ((staging_region == nullptr) || (staging_region->handle == nullptr)) {
                error(user_context) << "Vulkan: Failed to allocate device memory!\n";
                return halide_error_code_device_malloc_failed;
            }

            // use the staging region and buffer from the copy destination
            src_buffer_region = reinterpret_cast<MemoryRegion *>(src->device);
            dst_buffer_region = staging_region;

        } else if (from_host && !to_host) {
            // Host to Device
            debug(user_context) << " buffer copy from: host to: device\n";

            // Need to make sure all reads and writes to/from destination are complete.
            MemoryRequest request = {0};
            request.size = src->size_in_bytes();

            // NOTE: We may re-use this buffer so enable both src and dst
            request.properties.usage = MemoryUsage::TransferSrcDst;
            request.properties.caching = MemoryCaching::UncachedCoherent;
            request.properties.visibility = MemoryVisibility::HostToDevice;

            // allocate a new region
            staging_region = ctx.allocator->reserve(user_context, request);
            if ((staging_region == nullptr) || (staging_region->handle == nullptr)) {
                error(user_context) << "Vulkan: Failed to allocate device memory!\n";
                return halide_error_code_device_malloc_failed;
            }

            // map the region to a host ptr
            uint8_t *stage_host_ptr = (uint8_t *)ctx.allocator->map(user_context, staging_region);
            if (stage_host_ptr == nullptr) {
                error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
                return halide_error_code_copy_to_device_failed;
            }

            // copy to the (host-visible/coherent) staging buffer, then restore the dst pointer
            uint64_t copy_dst_ptr = copy_helper.dst;
            copy_helper.dst = (uint64_t)(stage_host_ptr);
            copy_memory(copy_helper, user_context);
            copy_helper.dst = copy_dst_ptr;

            // unmap the pointer
            error_code = ctx.allocator->unmap(user_context, staging_region);
            if (error_code != halide_error_code_success) {
                error(user_context) << "Vulkan: Failed to unmap staging region!\n";
                return halide_error_code_copy_to_device_failed;
            }

            // use the staging region and buffer from the copy source
            src_buffer_region = staging_region;
            dst_buffer_region = reinterpret_cast<MemoryRegion *>(dst->device);

        } else if (from_host && to_host) {
            debug(user_context) << " buffer copy from: host to: host\n";
            copy_memory(copy_helper, user_context);
            return halide_error_code_success;
        }

        if (src_buffer_region == nullptr) {
            error(user_context) << "Vulkan: Failed to retrieve source buffer for device memory!\n";
            return halide_error_code_internal_error;
        }

        if (dst_buffer_region == nullptr) {
            error(user_context) << "Vulkan: Failed to retrieve destination buffer for device memory!\n";
            return halide_error_code_internal_error;
        }

        // get the owning memory region (that holds the allocation)
        MemoryRegion *src_memory_region = ctx.allocator->owner_of(user_context, src_buffer_region);
        MemoryRegion *dst_memory_region = ctx.allocator->owner_of(user_context, dst_buffer_region);

        // retrieve the buffers from the owning allocation region
        VkBuffer *src_device_buffer = reinterpret_cast<VkBuffer *>(src_memory_region->handle);
        VkBuffer *dst_device_buffer = reinterpret_cast<VkBuffer *>(dst_memory_region->handle);

        ScopedVulkanCommandBufferAndPool cmds(user_context, ctx.allocator, ctx.queue_family_index);
        if (cmds.error_code != halide_error_code_success) {
            error(user_context) << "Vulkan: Failed to create command buffer and pool!\n";
            if (to_host) {
                return halide_error_code_copy_to_host_failed;
            } else {
                return halide_error_code_copy_to_device_failed;
            }
        }

        // begin the command buffer
        VkCommandBufferBeginInfo command_buffer_begin_info =
            {
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // struct type
                nullptr,                                      // pointer to struct extending this
                VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
                nullptr                                       // pointer to parent command buffer
        };

        VkResult result = vkBeginCommandBuffer(cmds.command_buffer, &command_buffer_begin_info);
        if (result != VK_SUCCESS) {
            error(user_context) << "Vulkan: vkBeginCommandBuffer returned " << vk_get_error_name(result) << "\n";
            if (to_host) {
                return halide_error_code_copy_to_host_failed;
            } else {
                return halide_error_code_copy_to_device_failed;
            }
        }

        // define the src and dst config
        uint64_t copy_dst = copy_helper.dst;
        uint64_t src_offset = copy_helper.src_begin + (src_buffer_region->indexing.offset * src->type.bytes());
        uint64_t dst_offset = copy_helper.dst_begin + (dst_buffer_region->indexing.offset * dst->type.bytes());

        copy_helper.src = (uint64_t)(src_device_buffer);
        copy_helper.dst = (uint64_t)(dst_device_buffer);

        debug(user_context) << " src region=" << (void *)src_memory_region << " buffer=" << (void *)src_device_buffer << " copy_offset=" << src_offset << "\n";
        debug(user_context) << " dst region=" << (void *)dst_memory_region << " buffer=" << (void *)dst_device_buffer << " copy_offset=" << dst_offset << "\n";

        // enqueue the copy operation, using the allocated buffers
        error_code = vk_do_multidimensional_copy(user_context, cmds.command_buffer, copy_helper,
                                                 src_offset, dst_offset,
                                                 src->dimensions,
                                                 from_host, to_host);

        if (error_code != halide_error_code_success) {
            error(user_context) << "Vulkan: vk_do_multidimensional_copy failed!\n";
            return error_code;
        }

        // end the command buffer
        result = vkEndCommandBuffer(cmds.command_buffer);
        if (result != VK_SUCCESS) {
            error(user_context) << "vkEndCommandBuffer returned " << vk_get_error_name(result) << "\n";
            if (to_host) {
                return halide_error_code_copy_to_host_failed;
            } else {
                return halide_error_code_copy_to_device_failed;
            }
        }

        //// submit the command buffer to our command queue
        VkSubmitInfo submit_info =
            {
                VK_STRUCTURE_TYPE_SUBMIT_INFO,  // struct type
                nullptr,                        // pointer to struct extending this
                0,                              // wait semaphore count
                nullptr,                        // semaphores
                nullptr,                        // pipeline stages where semaphore waits occur
                1,                              // how many command buffers to execute
                &(cmds.command_buffer),         // the command buffers
                0,                              // number of semaphores to signal
                nullptr                         // the semaphores to signal
        };

        vk_invalidate_host_reads(ctx.allocator, ctx.queue);
        result = vkQueueSubmit(ctx.queue, 1, &submit_info, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            error(user_context) << "vkQueueSubmit returned " << vk_get_error_name(result) << "\n";
            return result;
        }

        //// wait until the queue is done with the command buffer
        result = vkQueueWaitIdle(ctx.queue);
        if (result != VK_SUCCESS) {
            error(user_context) << "vkQueueWaitIdle returned " << vk_get_error_name(result) << "\n";
            return result;
        }

        if (!from_host && to_host) {
            // map the staging region to a host ptr
            uint8_t *stage_host_ptr = (uint8_t *)ctx.allocator->map(user_context, staging_region);
            if (stage_host_ptr == nullptr) {
                error(user_context) << "Vulkan: Failed to map host pointer to device memory!\n";
                return halide_error_code_internal_error;
            }

            // copy to the (host-visible/coherent) staging buffer
            copy_helper.dst = copy_dst;
            copy_helper.src = (uint64_t)(stage_host_ptr);
            copy_memory(copy_helper, user_context);

            // unmap the pointer and reclaim the staging region
            error_code = ctx.allocator->unmap(user_context, staging_region);
            if (error_code != halide_error_code_success) {
                error(user_context) << "Vulkan: Failed to unmap pointer for staging region!\n";
                return error_code;
            }
        }

        if (staging_region) {
            if (halide_can_reuse_device_allocations(user_context)) {
                error_code = ctx.allocator->release(user_context, staging_region);
            } else {
                error_code = ctx.allocator->reclaim(user_context, staging_region);
            }
        }

        if (error_code != halide_error_code_success) {
            error(user_context) << "Vulkan: Failed to release staging region allocation!\n";
        }

#ifdef DEBUG_RUNTIME
        uint64_t t_after = halide_current_time_ns(user_context);
        debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif
    }

    return error_code;
}

WEAK int halide_vulkan_device_crop(void *user_context,
                                   const struct halide_buffer_t *src,
                                   struct halide_buffer_t *dst) {
    const int64_t offset = calc_device_crop_byte_offset(src, dst);
    return vk_device_crop_from_offset(user_context, src, offset, dst);
}

WEAK int halide_vulkan_device_slice(void *user_context,
                                    const struct halide_buffer_t *src,
                                    int slice_dim, int slice_pos,
                                    struct halide_buffer_t *dst) {
    const int64_t offset = calc_device_slice_byte_offset(src, slice_dim, slice_pos);
    return vk_device_crop_from_offset(user_context, src, offset, dst);
}

WEAK int halide_vulkan_device_release_crop(void *user_context,
                                           struct halide_buffer_t *halide_buffer) {

    debug(user_context)
        << "Vulkan: halide_vulkan_device_release_crop (user_context: " << user_context
        << ", halide_buffer: " << halide_buffer << ")\n";

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // get the allocated region for the device
    MemoryRegion *device_region = reinterpret_cast<MemoryRegion *>(halide_buffer->device);
    if (device_region == nullptr) {
        error(user_context) << "Vulkan: Failed to retrieve device region for buffer!\n";
        return halide_error_code_internal_error;
    }

    int error_code = ctx.allocator->destroy_crop(user_context, device_region);
    if (error_code != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to destroy crop for device region!\n";
        return error_code;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif

    return halide_error_code_success;
}

WEAK int halide_vulkan_run(void *user_context,
                           void *state_ptr,
                           const char *entry_name,
                           int blocksX, int blocksY, int blocksZ,
                           int threadsX, int threadsY, int threadsZ,
                           int shared_mem_bytes,
                           size_t arg_sizes[],
                           void *args[],
                           int8_t arg_is_buffer[]) {
#ifdef DEBUG_RUNTIME
    debug(user_context)
        << "halide_vulkan_run (user_context: " << user_context << ", "
        << "state_ptr: " << state_ptr << ", "
        << "entry: " << entry_name << ", "
        << "blocks: " << blocksX << "x" << blocksY << "x" << blocksZ << ", "
        << "threads: " << threadsX << "x" << threadsY << "x" << threadsZ << ", "
        << "shmem: " << shared_mem_bytes << "\n";
#endif

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

#ifdef DEBUG_RUNTIME
    uint64_t t_before = halide_current_time_ns(user_context);
#endif

    // Serialize batch recording and protect the list across custom contexts.
    ScopedMutexLock batch_lock(&command_batches_mutex);
    VulkanCommandBatch *pending_batch = *vk_find_command_batch(ctx.allocator, ctx.queue);
    if (pending_batch && pending_batch->ended) {
        return halide_error_code_generic_error;
    }

    // Cache shader modules and pipelines, while keeping descriptors and
    // scalar arguments private to each dispatch until synchronization.
    // 1. Get the shader module cache entry
    VulkanCompilationCacheEntry *cache_entry = nullptr;
    bool found = compilation_cache.lookup(ctx.device, state_ptr, cache_entry);
    if (!found || (cache_entry == nullptr)) {
        error(user_context) << "Vulkan: Failed to locate shader module! Unable to proceed!\n";
        return halide_error_code_internal_error;
    }

    // 1a. Locate the correct entry point from the cache
    bool found_entry_point = false;
    uint32_t entry_point_index = 0;
    VulkanCompiledShaderModule *shader_module = nullptr;
    for (uint32_t m = 0; m < cache_entry->module_count; m++) {
        VulkanCompiledShaderModule *compiled_shader = cache_entry->compiled_modules[m];
        for (uint32_t n = 0; (n < compiled_shader->shader_count) && !found_entry_point; ++n) {
            if (strcmp(compiled_shader->shader_bindings[n].entry_point_name, entry_name) == 0) {
                shader_module = compiled_shader;
                entry_point_index = n;
                found_entry_point = true;
            }
        }
    }

    if (!found_entry_point || (entry_point_index >= shader_module->shader_count) || (shader_module == nullptr)) {
        error(user_context) << "Vulkan: Failed to locate shader entry point! Unable to proceed!\n";
        return halide_error_code_internal_error;
    }

    debug(user_context) << " found entry point ["
                        << (entry_point_index + 1) << " of " << shader_module->shader_count
                        << "] '" << entry_name << "'\n";

    // 2. Create objects for execution
    if (shader_module->descriptor_set_layouts == nullptr) {
        error(user_context) << "Vulkan: Missing descriptor set layouts! Unable to proceed!\n";
        return halide_error_code_internal_error;
    }

    int error_code = halide_error_code_success;
    if (shader_module->pipeline_layout == VK_NULL_HANDLE) {

        // 2a. Create all descriptor set layouts
        for (uint32_t n = 0; n < shader_module->shader_count; ++n) {
            if (((void *)shader_module->descriptor_set_layouts[n]) == VK_NULL_HANDLE) {
                uint32_t uniform_buffer_count = shader_module->shader_bindings[n].uniform_buffer_count;
                uint32_t storage_buffer_count = shader_module->shader_bindings[n].storage_buffer_count;
                debug(user_context) << " creating descriptor set layout [" << n << "] " << shader_module->shader_bindings[n].entry_point_name << "\n";
                auto *binding = &shader_module->shader_bindings[n];
                // The extension guarantees at least 32 push descriptors, and
                // permits only one push-descriptor set in a pipeline layout.
                binding->push_descriptors = shader_module->shader_count == 1 &&
                                            (ctx.allocator->dispatch_extensions() & VulkanPushDescriptors) && vkCmdPushDescriptorSetKHR &&
                                            uniform_buffer_count + storage_buffer_count <= 32;
                error_code = vk_create_descriptor_set_layout(user_context, ctx.allocator, uniform_buffer_count, storage_buffer_count,
                                                             &shader_module->descriptor_set_layouts[n],
                                                             binding->push_descriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0);
                if (error_code != halide_error_code_success) {
                    error(user_context) << "Vulkan: Failed to create descriptor set layout!\n";
                    return error_code;
                }
            }
        }

        // 2b. Create the pipeline layout, including any push-constant arguments.
        uint32_t push_constant_bytes = 0;
        for (uint32_t n = 0; n < shader_module->shader_count; n++) {
            uint32_t size = shader_module->shader_bindings[n].push_constant_bytes;
            push_constant_bytes = size > push_constant_bytes ? size : push_constant_bytes;
        }
        error_code = vk_create_pipeline_layout(user_context, ctx.allocator, shader_module->shader_count, shader_module->descriptor_set_layouts,
                                               &shader_module->pipeline_layout, push_constant_bytes);
        if (error_code != halide_error_code_success) {
            error(user_context) << "Vulkan: Failed to create pipeline layout!\n";
            return error_code;
        }
    }

    VulkanDispatchData dispatch_data = {};
    dispatch_data.shared_mem_bytes = shared_mem_bytes;
    dispatch_data.global_size[0] = blocksX;
    dispatch_data.global_size[1] = blocksY;
    dispatch_data.global_size[2] = blocksZ;
    dispatch_data.local_size[0] = threadsX;
    dispatch_data.local_size[1] = threadsY;
    dispatch_data.local_size[2] = threadsZ;

    VulkanShaderBinding *entry_point_binding = (shader_module->shader_bindings + entry_point_index);

    size_t buffer_count = 0;
    for (size_t i = 0; arg_sizes[i] > 0; i++) {
        buffer_count += arg_is_buffer[i] != 0;
    }
    size_t resources_size = sizeof(VulkanCommandResources) + buffer_count * sizeof(MemoryRegion *);
    auto *resources = (VulkanCommandResources *)vk_host_malloc(user_context, resources_size, 0,
                                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, ctx.allocator->callbacks());
    if (!resources) {
        return halide_error_code_out_of_memory;
    }
    memset(resources, 0, sizeof(*resources));
    resources->buffers = reinterpret_cast<MemoryRegion **>(resources + 1);
    ScopedVulkanCommandResources scoped_resources = {user_context, ctx.allocator, resources};
    // Capture allocation owners while argument buffers and crops are live.
    // Read-only inputs count as references just like writable outputs.
    for (size_t i = 0; arg_sizes[i] > 0; i++) {
        if (arg_is_buffer[i]) {
            auto *region = reinterpret_cast<MemoryRegion *>(((halide_buffer_t *)args[i])->device);
            resources->buffers[resources->buffer_count++] = ctx.allocator->owner_of(user_context, region);
        }
    }

    // The binding owns cached pipelines; dispatch resources only own objects
    // that must remain private to this invocation.
    VkPipeline compute_pipeline = VK_NULL_HANDLE;
    error_code = vk_setup_compute_pipeline(user_context, ctx.allocator, entry_point_binding, &dispatch_data,
                                           shader_module->shader_module, shader_module->pipeline_layout,
                                           &compute_pipeline);
    if (error_code != halide_error_code_success) {
        return error_code;
    }

    VulkanCommandBatch *batch = nullptr;
    error_code = vk_get_command_batch(user_context, ctx, &batch);
    if (error_code != halide_error_code_success) {
        return error_code;
    }
    size_t argument_bytes = vk_estimate_scalar_uniform_buffer_size(user_context, arg_sizes, args, arg_is_buffer);
    uint32_t push_constants[32] = {};
    VkDescriptorBufferInfo args_buffer = {};
    if (entry_point_binding->push_constant_bytes) {
        if (argument_bytes != entry_point_binding->push_constant_bytes || argument_bytes > sizeof(push_constants)) {
            return halide_error_code_internal_error;
        }
        vk_pack_scalar_arguments(user_context, (uint8_t *)push_constants, sizeof(push_constants), arg_sizes, args, arg_is_buffer);
    } else if (entry_point_binding->uniform_buffer_count) {
        uint8_t *mapped = nullptr;
        error_code = vk_allocate_argument_slice(user_context, batch->arena, argument_bytes, &args_buffer, &mapped);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
        vk_pack_scalar_arguments(user_context, mapped, argument_bytes, arg_sizes, args, arg_is_buffer);
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if ((entry_point_binding->uniform_buffer_count || entry_point_binding->storage_buffer_count) && !entry_point_binding->push_descriptors) {
        error_code = vk_allocate_dispatch_descriptor_set(user_context, batch->arena, entry_point_binding,
                                                         shader_module->descriptor_set_layouts[entry_point_index], &descriptor_set);
        if (error_code != halide_error_code_success) {
            return error_code;
        }
    }
    error_code = vk_setup_descriptor_update_template(user_context, ctx.allocator, entry_point_binding,
                                                     shader_module->descriptor_set_layouts[entry_point_index],
                                                     shader_module->pipeline_layout, entry_point_index);
    if (error_code != halide_error_code_success) {
        return error_code;
    }
    error_code = vk_update_descriptor_set(user_context, ctx.allocator,
                                          entry_point_binding->uniform_buffer_count ? &args_buffer : nullptr, entry_point_binding,
                                          arg_sizes, args, arg_is_buffer, descriptor_set,
                                          batch->command_buffer, shader_module->pipeline_layout, entry_point_index);
    if (error_code != halide_error_code_success) {
        return error_code;
    }

    // Preserve RAW, WAR and WAW dependencies between dispatches, as well as
    // dependencies on earlier transfers submitted to this queue.
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT;
    vkCmdPipelineBarrier(batch->command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(batch->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
    if (descriptor_set) {
        vkCmdBindDescriptorSets(batch->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader_module->pipeline_layout,
                                entry_point_index, 1, &descriptor_set, 0, nullptr);
    }
    if (entry_point_binding->push_constant_bytes) {
        vkCmdPushConstants(batch->command_buffer, shader_module->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, entry_point_binding->push_constant_bytes, push_constants);
    }
    vkCmdDispatch(batch->command_buffer, blocksX, blocksY, blocksZ);

    resources->next = batch->resources;
    batch->resources = resources;
    scoped_resources.resources = nullptr;

#ifdef DEBUG_RUNTIME
    debug(user_context) << "halide_vulkan_run: blocks_allocated="
                        << (uint32_t)ctx.allocator->blocks_allocated() << " "
                        << "bytes_allocated_for_blocks=" << (uint32_t)ctx.allocator->bytes_allocated_for_blocks() << " "
                        << "regions_allocated=" << (uint32_t)ctx.allocator->regions_allocated() << " "
                        << "bytes_allocated_for_regions=" << (uint32_t)ctx.allocator->bytes_allocated_for_regions() << " "
                        << "\n";

    uint64_t t_after = halide_current_time_ns(user_context);
    debug(user_context) << "    Time: " << (t_after - t_before) / 1.0e6 << " ms\n";
#endif
    return halide_error_code_success;
}

WEAK int halide_vulkan_device_and_host_malloc(void *user_context, struct halide_buffer_t *buf) {
    return halide_default_device_and_host_malloc(user_context, buf, &vulkan_device_interface);
}

WEAK int halide_vulkan_device_and_host_free(void *user_context, struct halide_buffer_t *buf) {
    return halide_default_device_and_host_free(user_context, buf, &vulkan_device_interface);
}

WEAK int halide_vulkan_wrap_vk_buffer(void *user_context, struct halide_buffer_t *buf, uint64_t vk_buffer) {
    halide_debug_assert(user_context, buf->device == 0);
    if (buf->device != 0) {
        error(user_context) << "Vulkan: Unable to wrap buffer ... invalid device pointer!\n";
        return halide_error_code_device_wrap_native_failed;
    }
    buf->device = vk_buffer;
    buf->device_interface = &vulkan_device_interface;
    buf->device_interface->impl->use_module();
    return halide_error_code_success;
}

WEAK int halide_vulkan_detach_vk_buffer(void *user_context, halide_buffer_t *buf) {
    if (buf->device == 0) {
        return halide_error_code_success;
    }
    if (buf->device_interface != &vulkan_device_interface) {
        error(user_context) << "Vulkan: Unable to detach buffer ... invalid device interface!\n";
        return halide_error_code_incompatible_device_interface;
    }
    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        return ctx.error;
    }
    int error_code = vk_flush_command_batch(user_context, ctx.allocator, ctx.queue);
    if (error_code != halide_error_code_success) {
        return error_code;
    }
    buf->device = 0;
    buf->device_interface->impl->release_module();
    buf->device_interface = nullptr;
    return halide_error_code_success;
}

WEAK uintptr_t halide_vulkan_get_vk_buffer(void *user_context, halide_buffer_t *buf) {
    if (buf->device == 0) {
        return 0;
    }
    halide_debug_assert(user_context, buf->device_interface == &vulkan_device_interface);
    return (uintptr_t)buf->device;
}

// Borrow the allocation's native handle without exposing the allocator's internal layout.
WEAK int halide_vulkan_get_native_buffer(void *user_context, const halide_buffer_t *buf,
                                         uint64_t *buffer, uint64_t *offset) {
    if (!buf || !buffer || !offset || !buf->device ||
        buf->device_interface != &vulkan_device_interface) {
        return halide_error_code_incompatible_device_interface;
    }
    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        return ctx.error;
    }
    MemoryRegion *region = reinterpret_cast<MemoryRegion *>(buf->device);
    MemoryRegion *owner = ctx.allocator->owner_of(user_context, region);
    if (!owner || !owner->handle || region->indexing.offset < 0) {
        return halide_error_code_internal_error;
    }
    *buffer = (uint64_t)*reinterpret_cast<VkBuffer *>(owner->handle);
    // Each VkBuffer is already bound at the owner's allocation offset.
    *offset = (uint64_t)region->indexing.offset * buf->type.bytes();
    return halide_error_code_success;
}

WEAK const struct halide_device_interface_t *halide_vulkan_device_interface() {
    return &vulkan_device_interface;
}

WEAK halide_device_allocation_pool vulkan_allocation_pool;

WEAK int halide_vulkan_release_unused_device_allocations(void *user_context) {
    debug(user_context)
        << "halide_vulkan_release_unused_device_allocations (user_context: " << user_context
        << ")\n";

    VulkanContext ctx(user_context);
    if (ctx.error != halide_error_code_success) {
        error(user_context) << "Vulkan: Failed to acquire context!\n";
        return ctx.error;
    }

    // collect all unused allocations
    if (ctx.allocator) {
        ctx.allocator->collect(user_context);
    }
    return halide_error_code_success;
}

namespace {

WEAK __attribute__((constructor)) void register_vulkan_allocation_pool() {
    vulkan_allocation_pool.release_unused = &halide_vulkan_release_unused_device_allocations;
    halide_register_device_allocation_pool(&vulkan_allocation_pool);
}

WEAK __attribute__((destructor)) void halide_vulkan_cleanup() {
    // NOTE: In some cases, we've observed the NVIDIA driver causing a segfault
    //       at process exit.  It appears to be triggered by running multiple
    //       processes that use the Vulkan API, whereupon, one of the libs in
    //       their driver stack may crash inside the finalizer.  Unfortunately,
    //       any attempt to avoid it may also crash, since the function pointers
    //       obtained from the Vulkan loader appear to be invalid.
    //
    //       https://github.com/halide/Halide/issues/8497
    //
    //       So, we don't do any special handling here ... just clean up like
    //       the other runtimes do.
    //
    if (halide_vulkan_is_initialized()) {
        halide_vulkan_device_release(nullptr);
    }
}

// --------------------------------------------------------------------------

}  // namespace

// --------------------------------------------------------------------------

}  // extern "C" linkage

// --------------------------------------------------------------------------

namespace Halide {
namespace Runtime {
namespace Internal {
namespace Vulkan {

// --------------------------------------------------------------------------

WEAK halide_device_interface_impl_t vulkan_device_interface_impl = {
    halide_use_jit_module,
    halide_release_jit_module,
    halide_vulkan_device_malloc,
    halide_vulkan_device_free,
    halide_vulkan_device_sync,
    halide_vulkan_device_release,
    halide_vulkan_copy_to_host,
    halide_vulkan_copy_to_device,
    halide_vulkan_device_and_host_malloc,
    halide_vulkan_device_and_host_free,
    halide_vulkan_buffer_copy,
    halide_vulkan_device_crop,
    halide_vulkan_device_slice,
    halide_vulkan_device_release_crop,
    halide_vulkan_wrap_vk_buffer,
    halide_vulkan_detach_vk_buffer,
};

WEAK halide_device_interface_t vulkan_device_interface = {
    halide_device_malloc,
    halide_device_free,
    halide_device_sync,
    halide_device_release,
    halide_copy_to_host,
    halide_copy_to_device,
    halide_device_and_host_malloc,
    halide_device_and_host_free,
    halide_buffer_copy,
    halide_device_crop,
    halide_device_slice,
    halide_device_release_crop,
    halide_device_wrap_native,
    halide_device_detach_native,
    halide_vulkan_compute_capability,
    &vulkan_device_interface_impl};

// --------------------------------------------------------------------------

}  // namespace Vulkan
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide
