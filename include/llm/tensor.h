// tensor.h — Task 1: basic matrix/vector storage.
//
// Deliberately small. Activations flowing through the network are fp32 and
// rarely more than a few MB (hidden state is 4096 floats ≈ 16 KB per token),
// so a plain contiguous, row-major, optionally-owning float buffer covers
// everything the forward pass needs. Quantized *weights* are NOT Tensors —
// they live on disk and are dequantized on the fly (see quant.h / loader.h).
#pragma once

#include "llm/common.h"

#include <initializer_list>
#include <numeric>
#include <vector>

namespace llm {

class Tensor {
public:
    Tensor() = default;

    // Owning tensor with the given shape, zero-initialized.
    explicit Tensor(std::vector<int64_t> shape) { reshape_alloc(std::move(shape)); }
    Tensor(std::initializer_list<int64_t> shape)
        : Tensor(std::vector<int64_t>(shape)) {}

    // Non-owning view over external memory (e.g. an mmap'd region). The caller
    // guarantees the pointer outlives the view.
    static Tensor view(float* data, std::vector<int64_t> shape) {
        Tensor t;
        t.shape_ = std::move(shape);
        t.data_ = data;
        t.owns_ = false;
        t.n_ = product(t.shape_);
        return t;
    }

    // move-only for the owning case; views copy fine but we keep it uniform.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept { move_from(o); }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) { release(); move_from(o); }
        return *this;
    }
    ~Tensor() { release(); }

    // Allocate/replace storage.
    void reshape_alloc(std::vector<int64_t> shape) {
        int64_t n = product(shape);
        if (owns_ && n == n_) { shape_ = std::move(shape); return; }
        release();
        shape_ = std::move(shape);
        n_ = n;
        data_ = static_cast<float*>(aligned_alloc64(sizeof(float) * (n_ > 0 ? n_ : 1)));
        owns_ = true;
        zero();
    }

    // Reinterpret shape without touching storage; element count must match.
    void reshape(std::vector<int64_t> shape) {
        LLM_CHECK(product(shape) == n_, "reshape: element count mismatch");
        shape_ = std::move(shape);
    }

    float* data() { return data_; }
    const float* data() const { return data_; }
    int64_t numel() const { return n_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(int i) const { return shape_.at(i); }
    int ndim() const { return static_cast<int>(shape_.size()); }
    bool empty() const { return n_ == 0; }

    float& operator[](int64_t i) { return data_[i]; }
    float operator[](int64_t i) const { return data_[i]; }

    // 2-D accessor (row-major): element at (r, c) of a [rows, cols] tensor.
    float& at(int64_t r, int64_t c) {
        return data_[r * shape_[1] + c];
    }
    float at(int64_t r, int64_t c) const {
        return data_[r * shape_[1] + c];
    }

    void zero() { if (data_ && n_) std::memset(data_, 0, sizeof(float) * n_); }
    void fill(float v) { for (int64_t i = 0; i < n_; ++i) data_[i] = v; }

    // Copy `count` floats from src into this tensor's buffer.
    void copy_from(const float* src, int64_t count) {
        LLM_CHECK(count <= n_, "copy_from: overflow");
        std::memcpy(data_, src, sizeof(float) * count);
    }

    static int64_t product(const std::vector<int64_t>& s) {
        return std::accumulate(s.begin(), s.end(), (int64_t)1,
                               std::multiplies<int64_t>());
    }

private:
    void release() {
        if (owns_ && data_) aligned_free(data_);
        data_ = nullptr; owns_ = false; n_ = 0; shape_.clear();
    }
    void move_from(Tensor& o) {
        shape_ = std::move(o.shape_);
        data_ = o.data_; owns_ = o.owns_; n_ = o.n_;
        o.data_ = nullptr; o.owns_ = false; o.n_ = 0;
    }

    std::vector<int64_t> shape_;
    float* data_ = nullptr;
    bool owns_ = false;
    int64_t n_ = 0;
};

} // namespace llm
