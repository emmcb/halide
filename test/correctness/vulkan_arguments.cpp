#include "Halide.h"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace Halide;

// Check both sides of the portable push-constant limit without needing a GPU.
// Compiling also validates the generated SPIR-V through the optimizer.
template<typename T = int32_t>
void check_arguments(int scalar_count, bool vulkan13) {
    Target target = get_host_target().with_feature(Target::Vulkan);
    target.set_feature(Target::VulkanV13, vulkan13);
    target.set_feature(Target::VulkanInt8);
    target.set_feature(Target::VulkanInt16);
    target.set_feature(Target::VulkanInt64);
    std::vector<Param<T>> params(scalar_count);
    Var x("x"), tx("tx");
    Expr value = cast<T>(x);
    for (const auto &param : params) {
        value += param;
    }
    Func output("output");
    output(x) = value;
    output.bound(x, 0, 32).gpu_tile(x, tx, 32);
    Module module = output.compile_to_module(output.infer_arguments(), "vulkan_arguments", target);

    bool found = false;
    for (const auto &buffer : module.buffers()) {
        if (buffer.name().find("gpu_source_kernels") == std::string::npos) {
            continue;
        }
        found = true;
        const auto *words = (const uint32_t *)buffer.data();
        assert(words[0] == 1);  // One kernel in the module table.
        const uint32_t *shader = words + 3 + words[2];
        assert(shader[1] == 1);  // One entry point in the shader sidecar.
        uint32_t argument_binding = shader[3 + shader[2]];
        const uint32_t argument_bytes = ((scalar_count * sizeof(T) + 3) & ~3u) + sizeof(int32_t);
        if (sizeof(T) >= 4 && argument_bytes <= 128) {
            assert(argument_binding == (0x80000000u | argument_bytes));
        } else {
            assert(argument_binding == 1);  // One uniform-buffer argument block.
        }
    }
    assert(found);
}

int main(int, char **) {
    for (bool vulkan13 : {false, true}) {
        for (int count : {0, 1, 30, 31, 32, 64}) {
            check_arguments(count, vulkan13);
        }
        check_arguments<int8_t>(1, vulkan13);
        check_arguments<int16_t>(1, vulkan13);
        check_arguments<int64_t>(1, vulkan13);
    }
    std::printf("Success!\n");
    return 0;
}
