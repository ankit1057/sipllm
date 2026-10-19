#include "llm/mem_manager.h"
#include "tests/test_util.h"

using namespace llm;

TEST(mem_manager_accounting) {
    MemoryManager mm(1000);
    CHECK(mm.limit() == 1000);
    CHECK(mm.total_allocated() == 0);

    CHECK(mm.allocate(300, MemCategory::Weights));
    CHECK(mm.allocated(MemCategory::Weights) == 300);
    CHECK(mm.total_allocated() == 300);
    CHECK(mm.peak_allocated() == 300);

    CHECK(mm.allocate(200, MemCategory::KVCache));
    CHECK(mm.allocated(MemCategory::KVCache) == 200);
    CHECK(mm.total_allocated() == 500);
    CHECK(mm.peak_allocated() == 500);

    // Deallocate
    mm.deallocate(100, MemCategory::Weights);
    CHECK(mm.allocated(MemCategory::Weights) == 200);
    CHECK(mm.total_allocated() == 400);
    CHECK(mm.peak_allocated() == 500); // peak preserved
}

TEST(mem_manager_pressure_and_limits) {
    MemoryManager mm(1000);
    CHECK(!mm.has_pressure());

    CHECK(mm.allocate(860, MemCategory::Workspace));
    CHECK(mm.has_pressure()); // >= 85%
    CHECK(mm.pressure_ratio() > 0.85f);

    // Over-allocation must be rejected
    CHECK(!mm.allocate(200, MemCategory::General));

    // Under-allocation within budget succeeds
    CHECK(mm.allocate(100, MemCategory::General));
    CHECK(mm.total_allocated() == 960);
}

TEST(mem_manager_global_singleton) {
    auto& g = global_mem_manager();
    size_t before = g.total_allocated();
    CHECK(g.allocate(64, MemCategory::Tensors));
    CHECK(g.total_allocated() == before + 64);
    g.deallocate(64, MemCategory::Tensors);
    CHECK(g.total_allocated() == before);
}

int main() {
    printf("== test_mem_manager ==\n");
    return llmtest::run_all();
}
