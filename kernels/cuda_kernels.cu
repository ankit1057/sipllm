// cuda_kernels.cu — GPU matvec / small-batch matmul for GGUF weights.
//
// Compiled OFFLINE to PTX by `make ptx` (clang CUDA, no CUDA toolkit, no CUDA
// headers) and embedded into src/cuda_ptx.inc. The default build never needs a
// CUDA compiler: src/cuda_backend.cpp loads the embedded PTX at runtime through
// the dlopen'ed driver API and the driver JIT-compiles it for the actual GPU.
//
//   make ptx   # regenerate src/cuda_ptx.inc from this file
//
// Everything here uses clang NVVM builtins + inline PTX only (-nocudainc).
//
// Math: y[t][r] = sum_i W[r][i] * x[t][i], fp32 activations and accumulation,
// weights dequantized in-register from their (device-side) block layout:
//
//   Q4_K  144 B / 256 weights   — byte-identical to GGUF
//   Q5_K  176 B / 256 weights   — byte-identical to GGUF
//   Q6_K  212 B / 256 weights   — GGUF's 210-byte block padded to 212 at upload
//                                 time so every field is 4-byte aligned
//   F16   512 B / 256 weights, F32 1024 B / 256 weights
//
// Every type is processed in 256-element "super-blocks"; within one, each of a
// warp's 32 lanes decodes exactly 8 weights that sit at two float4-aligned
// offsets (p0..p0+3, p1..p1+3). The kernels are then type-agnostic templates.

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef signed char        i8;

#define DEV   __attribute__((device)) static inline
#define DEVS  __attribute__((device)) inline   // explicit specializations
#define KERN  extern "C" __attribute__((global))
#define SHMEM __attribute__((shared))

struct __attribute__((aligned(16))) f4 { float x, y, z, w; };

DEV int tid()   { return __nvvm_read_ptx_sreg_tid_x(); }
DEV int bidx()  { return __nvvm_read_ptx_sreg_ctaid_x(); }
DEV int bidy()  { return __nvvm_read_ptx_sreg_ctaid_y(); }

DEV float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

DEV float warp_sum(float v) {
    v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, 16, 0x1f);
    v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, 8, 0x1f);
    v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, 4, 0x1f);
    v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, 2, 0x1f);
    v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, 1, 0x1f);
    return v;
}

enum { T_F32 = 0, T_F16 = 1, T_Q4K = 2, T_Q5K = 3, T_Q6K = 4 };

template <int T> struct BS;
template <> struct BS<T_F32> { enum { v = 1024 }; };
template <> struct BS<T_F16> { enum { v = 512 }; };
template <> struct BS<T_Q4K> { enum { v = 144 }; };
template <> struct BS<T_Q5K> { enum { v = 176 }; };
template <> struct BS<T_Q6K> { enum { v = 212 }; };

// ggml get_scale_min_k4: 8 x (6-bit scale, 6-bit min) packed in 12 bytes.
DEV void scale_min_k4(int j, const u8* q, float& sc, float& mn) {
    if (j < 4) {
        sc = (float)(q[j] & 63);
        mn = (float)(q[j + 4] & 63);
    } else {
        sc = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Decode this lane's 8 weights of super-block `blk`; w[0..3] belong at
// offsets p0..p0+3 and w[4..7] at p1..p1+3 within the 256-element block.
template <int T> DEV void decode(const u8* blk, int lane, float* w, int& p0, int& p1);

template <> DEVS void decode<T_F32>(const u8* blk, int lane, float* w, int& p0, int& p1) {
    p0 = 8 * lane; p1 = p0 + 4;
    const f4* p = (const f4*)(blk + 32 * lane);
    f4 a = p[0], b = p[1];
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
    w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
}

template <> DEVS void decode<T_F16>(const u8* blk, int lane, float* w, int& p0, int& p1) {
    p0 = 8 * lane; p1 = p0 + 4;
    const u32* p = (const u32*)(blk + 16 * lane);
    u32 v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];
    w[0] = h2f((u16)(v0 & 0xffff)); w[1] = h2f((u16)(v0 >> 16));
    w[2] = h2f((u16)(v1 & 0xffff)); w[3] = h2f((u16)(v1 >> 16));
    w[4] = h2f((u16)(v2 & 0xffff)); w[5] = h2f((u16)(v2 >> 16));
    w[6] = h2f((u16)(v3 & 0xffff)); w[7] = h2f((u16)(v3 >> 16));
}

// Q4_K: lane -> 64-weight group j = lane/8, byte quad sub = lane%8 of qs[32j..].
// Low nibbles are weights 64j+4sub+{0..3} (scale 2j), high nibbles +32 (2j+1).
template <> DEVS void decode<T_Q4K>(const u8* blk, int lane, float* w, int& p0, int& p1) {
    const int j = lane >> 3, sub = lane & 7;
    const u32 dd = *(const u32*)blk;
    const float d = h2f((u16)(dd & 0xffff)), dmin = h2f((u16)(dd >> 16));
    float s1, m1, s2, m2;
    scale_min_k4(2 * j, blk + 4, s1, m1);
    scale_min_k4(2 * j + 1, blk + 4, s2, m2);
    const float a1 = d * s1, b1 = dmin * m1, a2 = d * s2, b2 = dmin * m2;
    const u32 q = *(const u32*)(blk + 16 + 32 * j + 4 * sub);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i]     = a1 * (float)((q >> (8 * i)) & 0xF) - b1;
        w[4 + i] = a2 * (float)((q >> (8 * i + 4)) & 0xF) - b2;
    }
    p0 = 64 * j + 4 * sub; p1 = p0 + 32;
}

// Q5_K: Q4_K plus a 5th bit from qh[32] (bit 2j for low nibble, 2j+1 for high).
template <> DEVS void decode<T_Q5K>(const u8* blk, int lane, float* w, int& p0, int& p1) {
    const int j = lane >> 3, sub = lane & 7;
    const u32 dd = *(const u32*)blk;
    const float d = h2f((u16)(dd & 0xffff)), dmin = h2f((u16)(dd >> 16));
    float s1, m1, s2, m2;
    scale_min_k4(2 * j, blk + 4, s1, m1);
    scale_min_k4(2 * j + 1, blk + 4, s2, m2);
    const float a1 = d * s1, b1 = dmin * m1, a2 = d * s2, b2 = dmin * m2;
    const u32 h = *(const u32*)(blk + 16 + 4 * sub);
    const u32 q = *(const u32*)(blk + 48 + 32 * j + 4 * sub);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const u32 hb = h >> (8 * i);
        w[i]     = a1 * (float)(((q >> (8 * i)) & 0xF) + (((hb >> (2 * j)) & 1) << 4)) - b1;
        w[4 + i] = a2 * (float)(((q >> (8 * i + 4)) & 0xF) + (((hb >> (2 * j + 1)) & 1) << 4)) - b2;
    }
    p0 = 64 * j + 4 * sub; p1 = p0 + 32;
}

// Q6_K (padded to 212 B: ql[128] qh[64] scales[16] d u16 pad u16).
// Half n = lane/16 covers weights 128n..128n+127. Within it, lane m = lane%16
// takes l' = 4*(m%8)..+3 and output pair m/8: pair 0 -> (q1 @ +0, q3 @ +64),
// pair 1 -> (q2 @ +32, q4 @ +96), exactly ggml's dequantize_row_q6_K.
template <> DEVS void decode<T_Q6K>(const u8* blk, int lane, float* w, int& p0, int& p1) {
    const int n = lane >> 4, m = lane & 15, pair = m >> 3, l0 = 4 * (m & 7);
    const float d = h2f(*(const u16*)(blk + 208));
    const i8* sc = (const i8*)(blk + 192) + 8 * n;
    const int is = l0 >> 4;
    const float sa = d * (float)sc[is + 2 * pair];
    const float sb = d * (float)sc[is + 2 * pair + 4];
    const u32 ql = *(const u32*)(blk + 64 * n + 32 * pair + l0);
    const u32 qh = *(const u32*)(blk + 128 + 32 * n + l0);
    const int sA = 2 * pair, sB = 2 * pair + 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int qa = (int)(((ql >> (8 * i)) & 0xF) | (((qh >> (8 * i + sA)) & 3) << 4)) - 32;
        const int qb = (int)(((ql >> (8 * i + 4)) & 0xF) | (((qh >> (8 * i + sB)) & 3) << 4)) - 32;
        w[i]     = sa * (float)qa;
        w[4 + i] = sb * (float)qb;
    }
    p0 = 128 * n + 32 * pair + l0; p1 = p0 + 64;
}

DEV float dot8(const float* w, f4 a, f4 b) {
    return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w +
           w[4] * b.x + w[5] * b.y + w[6] * b.z + w[7] * b.w;
}

// ---- matvec: one warp per output row, 4 warps per CTA (128 threads) -------
template <int T>
DEV void matvec(const u8* __restrict__ W, const float* __restrict__ x,
                float* __restrict__ y, int n_out, int nb) {
    const int lane = tid() & 31;
    const int row = bidx() * 4 + (tid() >> 5);
    if (row >= n_out) return;
    const u8* rp = W + (unsigned long long)row * nb * BS<T>::v;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        float w[8]; int p0, p1;
        decode<T>(rp + (unsigned long long)b * BS<T>::v, lane, w, p0, p1);
        const float* xb = x + 256 * b;
        acc += dot8(w, *(const f4*)(xb + p0), *(const f4*)(xb + p1));
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ---- small-batch matmul (prefill): Y[t][r] = W[r] . X[t] ------------------
// CTA = 8 warps; each warp owns RPW rows (8*RPW rows per CTA); CTA covers TB tokens. Per super-block
// the CTA stages X[TB][256] in shared memory once, then every warp dequantizes
// each of its rows' blocks ONCE and dots it against all TB staged tokens.
#define TB  16
// Rows per warp: 4 (x reuse across rows), except Q6_K whose heavier decode
// would push 4 rows past 128 registers and halve occupancy. Host mirrors this.
template <int T> struct RPWOf { enum { v = 4 }; };
template <> struct RPWOf<T_Q6K> { enum { v = 2 }; };
template <int T>
DEV void matmul(const u8* __restrict__ W, const float* __restrict__ X,
                float* __restrict__ Y, int n_out, int nb, int m) {
    constexpr int RPW = RPWOf<T>::v;
    SHMEM __attribute__((aligned(16))) float xs[TB * 256];
    const int lane = tid() & 31, warp = tid() >> 5;
    const int row0 = (bidx() * 8 + warp) * RPW;
    const int t0 = bidy() * TB;
    const int n_in = nb * 256;
    float acc[RPW][TB];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int t = 0; t < TB; ++t) acc[r][t] = 0.f;

    for (int b = 0; b < nb; ++b) {
        __nvvm_barrier_sync(0);
        for (int idx = tid(); idx < TB * 64; idx += 256) {
            const int t = idx >> 6, c = idx & 63;
            f4* dst = (f4*)(xs + t * 256 + 4 * c);
            if (t0 + t < m) *dst = *(const f4*)(X + (unsigned long long)(t0 + t) * n_in + 256 * b + 4 * c);
            else { dst->x = 0.f; dst->y = 0.f; dst->z = 0.f; dst->w = 0.f; }
        }
        __nvvm_barrier_sync(0);
        // Decode this warp's RPW rows once (p0/p1 depend only on lane+type),
        // then load each staged token's x ONCE and reuse it across all rows —
        // RPW x fewer shared-memory loads per FMA.
        float w[RPW][8]; int p0 = 0, p1 = 0;
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const int row = row0 + r;
            const int rc = row < n_out ? row : n_out - 1;   // clamp: stay in bounds
            decode<T>(W + ((unsigned long long)rc * nb + b) * BS<T>::v, lane, w[r], p0, p1);
            if (row >= n_out)
#pragma unroll
                for (int i = 0; i < 8; ++i) w[r][i] = 0.f;
        }
#pragma unroll
        for (int t = 0; t < TB; ++t) {
            const f4 xa = *(const f4*)(xs + t * 256 + p0);
            const f4 xb = *(const f4*)(xs + t * 256 + p1);
#pragma unroll
            for (int r = 0; r < RPW; ++r) acc[r][t] += dot8(w[r], xa, xb);
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int t = 0; t < TB; ++t) {
            const float s = warp_sum(acc[r][t]);
            if (lane == 0 && row < n_out && t0 + t < m)
                Y[(unsigned long long)(t0 + t) * n_out + row] = s;
        }
    }
}

#define INSTANTIATE(NAME, T)                                                   \
    KERN void mv_##NAME(const u8* W, const float* x, float* y, int n_out, int nb) { \
        matvec<T>(W, x, y, n_out, nb);                                         \
    }                                                                          \
    KERN void mm_##NAME(const u8* W, const float* X, float* Y, int n_out, int nb, int m) { \
        matmul<T>(W, X, Y, n_out, nb, m);                                      \
    }

INSTANTIATE(f32, T_F32)
INSTANTIATE(f16, T_F16)
INSTANTIATE(q4k, T_Q4K)
INSTANTIATE(q5k, T_Q5K)
INSTANTIATE(q6k, T_Q6K)
