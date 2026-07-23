// test_rope.cpp — RoPE and the llama3 (#9) frequency scaling.
//
// Plain RoPE is verified against closed-form cos/sin rotations. The llama3
// stretch is verified by its three regimes, each with an *independent*
// reference (not the implementation's own formula):
//   - high-frequency pairs pass through unchanged (identical to plain RoPE),
//   - low-frequency pairs have their angle divided by `factor`,
//   - the medium band is strictly interpolated between those two.
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <cmath>
#include <vector>

using namespace llm;

// head_dim=8, theta=10000 => freqs 10000^(-i/4): 1, 0.1, 0.01, 0.001.
static constexpr int64_t HD = 8;
static constexpr float   THETA = 10000.f;

// Rotate the pairs (1,0),(1,0),(1,0),(1,0): after RoPE, pair i is
// (cos(angle_i), sin(angle_i)), so the applied angle is recoverable directly.
static std::vector<float> unit_pairs() {
    std::vector<float> v(HD, 0.f);
    for (int64_t i = 0; i < HD / 2; ++i) v[2 * i] = 1.f;
    return v;
}
static double angle_of(const std::vector<float>& v, int64_t pair) {
    return std::atan2((double)v[2 * pair + 1], (double)v[2 * pair]);
}

TEST(rope_plain_matches_closed_form) {
    const int64_t pos = 2;
    auto v = unit_pairs();
    Transformer::apply_rope(v.data(), /*n_heads=*/1, HD, pos, THETA);  // default: plain
    double freq[4] = {1.0, 0.1, 0.01, 0.001};
    for (int64_t i = 0; i < 4; ++i) {
        APPROX(v[2 * i],     std::cos(pos * freq[i]), 1e-5);
        APPROX(v[2 * i + 1], std::sin(pos * freq[i]), 1e-5);
    }
}

TEST(rope_llama3_low_high_bands) {
    const int64_t pos = 2;
    Transformer::RopeScaling rs;
    rs.llama3 = true; rs.factor = 8.f;
    rs.low_freq_factor = 1.f; rs.high_freq_factor = 4.f; rs.orig_ctx_len = 64.f;
    // Bands for orig_ctx=64: high_wl=16, low_wl=64. Wavelengths 2*pi/freq:
    //   i0 ~6.28 (<16, high), i1 ~62.8 (medium), i2 ~628 (>64, low), i3 ~6283 (low).
    auto v = unit_pairs();
    Transformer::apply_rope(v.data(), 1, HD, pos, THETA, rs);

    // i0 high-frequency: unchanged vs plain RoPE (angle = pos * 1.0).
    APPROX(angle_of(v, 0), pos * 1.0, 1e-5);
    // i2, i3 low-frequency: angle divided by factor.
    APPROX(angle_of(v, 2), pos * (0.01 / 8.0), 1e-6);
    APPROX(angle_of(v, 3), pos * (0.001 / 8.0), 1e-7);
}

TEST(rope_llama3_medium_interpolates) {
    const int64_t pos = 2;
    Transformer::RopeScaling rs;
    rs.llama3 = true; rs.factor = 8.f;
    rs.low_freq_factor = 1.f; rs.high_freq_factor = 4.f; rs.orig_ctx_len = 64.f;
    auto v = unit_pairs();
    Transformer::apply_rope(v.data(), 1, HD, pos, THETA, rs);
    // i1 sits in the medium band: its angle must lie strictly between the
    // divided-by-factor angle and the untouched (plain) angle.
    const double medium = angle_of(v, 1);
    const double plain  = pos * 0.1;          // untouched
    const double scaled = pos * (0.1 / 8.0);  // divided by factor
    CHECK(medium > scaled && medium < plain);
}

TEST(rope_llama3_disabled_equals_plain) {
    // A default RopeScaling (llama3=false) must reproduce plain RoPE exactly.
    const int64_t pos = 5;
    auto a = unit_pairs(), b = unit_pairs();
    Transformer::apply_rope(a.data(), 1, HD, pos, THETA);
    Transformer::apply_rope(b.data(), 1, HD, pos, THETA, Transformer::RopeScaling{});
    for (int64_t i = 0; i < HD; ++i) APPROX(a[i], b[i], 0);
}

int main() { return llmtest::run_all(); }
