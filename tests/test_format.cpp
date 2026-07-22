// test_format.cpp — Phase 2 verification: write a .llmw, read it back through
// both the pread path and the mmap path, and confirm metadata + tensor bytes
// survive the round trip with correct 64-byte alignment.
#include "llm/format.h"
#include "tests/test_util.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace llm;

static std::string tmp_path(const char* name) { return llmtest::scratch_path(name); }

TEST(format_roundtrip_pread) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> a(37), b(4 * 8);
    for (auto& v : a) v = d(rng);
    for (auto& v : b) v = d(rng);

    ModelWriter w;
    w.set_meta("n_layers", (int64_t)2);
    w.set_meta("rope_theta", 10000.0);
    w.set_meta("arch", std::string("toy"));
    w.add_f32("vec", {37}, a.data());
    w.add_f32("mat", {4, 8}, b.data());
    std::string path = tmp_path("rt.llmw");
    w.write(path);

    ModelFile f(path, /*mmap=*/false);
    CHECK(f.meta_int("n_layers") == 2);
    APPROX(f.meta_float("rope_theta"), 10000.0, 1e-9);
    CHECK(f.meta_str("arch") == "toy");
    CHECK(f.tensors().size() == 2);

    const TensorInfo* vec = f.find("vec");
    const TensorInfo* mat = f.find("mat");
    CHECK(vec && mat);
    CHECK(vec->numel() == 37);
    CHECK(mat->shape[0] == 4 && mat->shape[1] == 8);
    // 64-byte alignment of data regions.
    CHECK(vec->offset % 64 == 0);
    CHECK(mat->offset % 64 == 0);

    std::vector<float> ra(37), rb(32);
    f.read_raw(*vec, ra.data());
    f.read_raw(*mat, rb.data());
    for (int i = 0; i < 37; ++i) APPROX(ra[i], a[i], 1e-12);
    for (int i = 0; i < 32; ++i) APPROX(rb[i], b[i], 1e-12);
}

TEST(format_roundtrip_mmap) {
    std::vector<float> a(16);
    for (int i = 0; i < 16; ++i) a[i] = i * 0.5f;
    ModelWriter w;
    w.add_f32("x", {16}, a.data());
    std::string path = tmp_path("rt_mmap.llmw");
    w.write(path);

    ModelFile f(path, /*mmap=*/true);
    CHECK(f.mmap_base() != nullptr);
    const TensorInfo* x = f.find("x");
    std::vector<float> rx(16);
    f.read_raw(*x, rx.data());           // goes through mmap memcpy path
    for (int i = 0; i < 16; ++i) APPROX(rx[i], i * 0.5, 1e-12);
    // Direct zero-copy view.
    const float* view = reinterpret_cast<const float*>(f.mmap_base() + x->offset);
    for (int i = 0; i < 16; ++i) APPROX(view[i], i * 0.5, 1e-12);
}

TEST(format_bad_magic_rejected) {
    std::string path = tmp_path("bad.llmw");
    FILE* fp = std::fopen(path.c_str(), "wb");
    const char junk[16] = "NOTAMODELFILE!";
    std::fwrite(junk, 1, sizeof(junk), fp);
    std::fclose(fp);
    bool threw = false;
    try { ModelFile f(path); } catch (const Error&) { threw = true; }
    CHECK(threw);
}

int main() {
    printf("== test_format ==\n");
    return llmtest::run_all();
}
