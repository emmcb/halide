#include "Halide.h"

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace Halide;
using namespace Halide::Internal;

namespace {

// Fail with a diagnostic rather than depending on assertions enabled at build time.
void check(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}

// Count the versioning decisions, loop copies, and expensive mirror operations.
class CountNodes : public IRVisitor {
    using IRVisitor::visit;

    // Count statement branches without including expression selects.
    void visit(const IfThenElse *op) override {
        branches++;
        IRVisitor::visit(op);
    }

    // Count all loop versions to detect recursive partitioning in the fallback.
    void visit(const For *op) override {
        loops++;
        IRVisitor::visit(op);
    }

    // Mirroring must survive only in the general version.
    void visit(const Mod *op) override {
        remainders++;
        IRVisitor::visit(op);
    }

public:
    int branches = 0;
    int loops = 0;
    int remainders = 0;
};

// Retain a direct-coordinate hint and an expensive general mirror expression.
Expr mirror(Expr coordinate, Expr extent) {
    Expr limit = extent - 1;
    Expr reflected = limit - cast(coordinate.type(), abs(coordinate % (2 * limit) - limit));
    return select(coordinate >= 0 && coordinate < extent,
                  likely(clamp(likely(coordinate), 0, limit)), reflected);
}

// Build a two-dimensional footprint with a policy on the outer reduction loop.
Stmt footprint(Expr x, Expr y, Expr width, Expr height, Partition policy) {
    Expr rx = Variable::make(Int(32), "rx");
    Expr ry = Variable::make(Int(32), "ry");
    Expr value = 0;
    for (int lane = 0; lane < 4; ++lane) {
        value += mirror(x + 4 * rx + lane, width) + 1000 * mirror(y + ry, height);
    }

    Stmt body = Store::make("output", value, rx + 6 * ry, Parameter(),
                            const_true(), ModulusRemainder(), false);
    body = For::make("rx", 0, 5, ForType::Serial, Partition::Auto, DeviceAPI::None, body);
    return For::make("ry", 0, 23, ForType::Serial, policy, DeviceAPI::None, body);
}

// Check one guard for both dimensions, outside both loops, and a direct fast path.
void check_structure() {
    Expr x = Variable::make(Int(32), "x");
    Expr y = Variable::make(Int(32), "y");
    Expr width = Variable::make(Int(32), "width");
    Expr height = Variable::make(Int(32), "height");
    Stmt nest = footprint(x, y, width, height, Partition::GuardWithIf);
    Stmt result = partition_loops(nest);
    const IfThenElse *branch = result.as<IfThenElse>();
    check(branch != nullptr, "Expected one guard outside the selected loop nest");
    check(!expr_uses_var(branch->condition, "rx") && !expr_uses_var(branch->condition, "ry"),
          "The guard still depends on an enclosed reduction variable");

    int comparisons = 0;
    visit_with(branch->condition, [&](auto *self, const LT *op) {
                   comparisons++;
                   self->visit_base(op); }, [&](auto *self, const LE *op) {
                   comparisons++;
                   self->visit_base(op); });
    check(comparisons <= 4, "Unrolled accesses left redundant comparisons in the interior guard");

    CountNodes all, steady, general;
    result.accept(&all);
    simplify(branch->then_case).accept(&steady);
    branch->else_case.accept(&general);
    check(all.branches == 1 && all.loops == 4, "Expected exactly two unpartitioned loop nests");
    check(steady.remainders == 0 && general.remainders > 0,
          "Mirror arithmetic must occur only in the general version");

    // Compare the inferred guard with a full-footprint interior test, including
    // tiny images where there is no steady region and distant repeated mirrors.
    for (int extent : {2, 17, 24, 25, 64}) {
        for (int origin_x : {-48, -1, 0, 1, 40, 63, 128}) {
            for (int origin_y : {-48, -1, 0, 1, 40, 63, 128}) {
                Expr guard = substitute({{"x", origin_x}, {"y", origin_y}, {"width", extent}, {"height", extent}},
                                        branch->condition);
                guard = simplify(guard);
                bool expected = origin_x >= 0 && origin_x + 23 < extent &&
                                origin_y >= 0 && origin_y + 23 < extent;
                check(expected ? is_const_one(guard) : is_const_zero(guard),
                      "The inferred guard does not cover exactly the interior footprint");
            }
        }
    }

    // Enclosing automatic partitioning must leave the explicitly guarded nest
    // in charge of its boundary decisions.
    Stmt outer = For::make("x", -12, 63, ForType::Serial, Partition::Auto, DeviceAPI::None, nest);
    Stmt outer_result = partition_loops(outer);
    CountNodes outer_nodes;
    outer_result.accept(&outer_nodes);
    check(outer_nodes.branches == 1 && outer_nodes.loops == 5,
          "An enclosing automatic loop consumed the guarded nest's hints");

    // Constant guards should eliminate the unused version altogether.
    CountNodes interior_nodes, empty_nodes;
    simplify(partition_loops(footprint(4, 4, 64, 64, Partition::GuardWithIf))).accept(&interior_nodes);
    partition_loops(footprint(4, 4, 2, 2, Partition::GuardWithIf)).accept(&empty_nodes);
    check(interior_nodes.branches == 0 && interior_nodes.remainders == 0 && interior_nodes.loops == 2,
          "A constant interior guard did not fold to the steady version");
    check(empty_nodes.branches == 0 && empty_nodes.loops == 2,
          "An empty steady region did not retain a single general version");
}

// Every vector lane must satisfy the guard, including an unrolled x footprint.
void check_vector_lanes() {
    Expr x = Variable::make(Int(32), "x");
    Expr ry = Variable::make(Int(32), "ry");
    Expr lanes = Ramp::make(x, 1, 4);
    Stmt body = Evaluate::make(mirror(lanes, 64) + mirror(ry, 64));
    Stmt nest = For::make("ry", 0, 23, ForType::Serial, Partition::GuardWithIf,
                          DeviceAPI::None, body);
    Stmt result = partition_loops(nest);
    const IfThenElse *branch = result.as<IfThenElse>();
    check(branch != nullptr && branch->condition.type().is_scalar(),
          "A vector footprint needs one scalar guard covering every lane");
    for (int origin : {-1, 0, 60, 61, 63, 64}) {
        Expr guard = simplify(substitute("x", origin, branch->condition));
        bool expected = origin >= 0 && origin + 3 < 64;
        check(expected ? is_const_one(guard) : is_const_zero(guard),
              "The guard did not check the full vector footprint");
    }
}

// Guard conditions must not speculate reads of mutable memory or impure calls.
void check_unhoistable_conditions() {
    Expr memory = Load::make(Int(32), "mutable", 0, Buffer<>(), Parameter(),
                             const_true(), ModulusRemainder(), false);
    Expr impure = Call::make(Int(32), "impure", {}, Call::Extern);
    for (Expr value : {memory, impure}) {
        Stmt body = Evaluate::make(select(value > 0, likely(Expr(1)), Expr(2)));
        Stmt nest = For::make("r", 0, 23, ForType::Serial, Partition::GuardWithIf,
                              DeviceAPI::None, body);
        CountNodes nodes;
        partition_loops(nest).accept(&nodes);
        check(nodes.branches == 0 && nodes.loops == 1,
              "A memory-dependent or impure guard was hoisted");
    }
}

// Never-partition GPU ancestors must still establish synchronization scope.
void check_gpu_synchronization() {
    Expr x = Variable::make(Int(32), "x");
    Stmt barrier = Evaluate::make(Call::make(Int(32), Call::gpu_thread_barrier, {0}, Call::Intrinsic));
    Stmt body = Block::make(Evaluate::make(mirror(x, 64)), barrier);
    body = For::make("r", 0, 23, ForType::Serial, Partition::GuardWithIf, DeviceAPI::None, body);
    body = For::make("kernel.__thread_id_x", 0, 31, ForType::GPUThread,
                     Partition::Never, DeviceAPI::OpenCL, body);
    body = For::make("kernel.__block_id_x", 0, 1, ForType::GPUBlock,
                     Partition::Never, DeviceAPI::OpenCL, body);
    CountNodes nodes;
    partition_loops(body).accept(&nodes);
    check(nodes.branches == 0 && nodes.loops == 3,
          "A guarded nest containing GPU synchronization was versioned");
}

// Compare real mirror-interior reductions with Never for awkward sizes and mins.
void check_reductions() {
    ImageParam input(Int(16), 2, "input");
    Func bounded = BoundaryConditions::mirror_interior(input);
    Var x("x"), y("y");
    RDom r(0, 24, 0, 24, "r");
    Func reference("reference"), guarded("guarded");
    Expr sample = cast<int32_t>(bounded(16 * x - 12 + r.x, 16 * y - 12 + r.y));
    reference(x, y) = 0;
    reference(x, y) += sample;
    guarded(x, y) = 0;
    guarded(x, y) += sample;
    reference.never_partition_all();
    reference.update().never_partition_all();
    guarded.never_partition_all();
    RVar rxo("rxo"), rxi("rxi");
    guarded.update().split(r.x, rxo, rxi, 4).unroll(rxi).partition(r.y, Partition::GuardWithIf);

    Pipeline pipeline({reference, guarded});
    const std::vector<std::pair<int, int>> shapes{
        {2, 2}, {2, 17}, {17, 2}, {7, 9}, {16, 16}, {17, 23}, {63, 47}, {128, 96}, {129, 97}, {257, 191}};
    for (auto [width, height] : shapes) {
        Buffer<int16_t> image(width + 3, height);
        image.crop(0, 0, width);
        image.set_min(-3, 5);
        for (int yy = image.dim(1).min(); yy <= image.dim(1).max(); ++yy) {
            for (int xx = image.dim(0).min(); xx <= image.dim(0).max(); ++xx) {
                image(xx, yy) = (xx * 991 + yy * 1297) & 32767;
            }
        }

        input.set(image);
        Realization result = pipeline.realize({20, 16}, get_host_target());
        Buffer<int32_t> expected = result[0];
        Buffer<int32_t> actual = result[1];
        for (int yy = 0; yy < actual.height(); ++yy) {
            for (int xx = 0; xx < actual.width(); ++xx) {
                check(actual(xx, yy) == expected(xx, yy),
                      "Guarded mirror-interior reduction differs from the unpartitioned reference");
            }
        }
    }
}

}  // namespace

// Exercise both the IR contract and end-to-end numerical behavior.
int main(int argc, char **argv) {
    check_structure();
    check_vector_lanes();
    check_unhoistable_conditions();
    check_gpu_synchronization();
    check_reductions();
    printf("Success!\n");
    return 0;
}
