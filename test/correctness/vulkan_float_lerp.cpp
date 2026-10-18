#include "Halide.h"
#include <cstdio>

using namespace Halide;

// Compile nested interpolation followed by widening, with scalar and vector math.
void check_lerp(int bits, int lanes) {
    Target target = get_host_target().with_feature(Target::Vulkan);
    target.set_feature(Target::VulkanFloat16);
    target.set_feature(Target::VulkanFloat64);

    ImageParam input(Float(bits), 2, "input");
    Var x("x"), xo("xo"), xi("xi"), bx("bx"), tx("tx");
    Expr a = input(x, 0);
    Expr b = input(x, 1);
    Expr weight = input(x, 2);
    Func output("output");
    output(x) = cast<double>(lerp(lerp(a, b, weight), b, weight));
    output.split(x, xo, xi, lanes).vectorize(xi).gpu_tile(xo, bx, tx, 32);

    // Code generation runs SPIRV-Tools validation without needing a Vulkan device.
    output.compile_to_module(output.infer_arguments(), "vulkan_float_lerp", target);
}

// Exercise every supported floating-point width through the Vulkan compiler.
int main(int argc, char **argv) try {
    for (int bits : {16, 32, 64}) {
        for (int lanes : {1, 4}) {
            check_lerp(bits, lanes);
        }
    }

    std::printf("Success!\n");
    return 0;
} catch (const Halide::Error &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
