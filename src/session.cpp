// session.cpp — SIPS v1 session (committed tokens + KV cache) serializer.
// See include/llm/session.h for the file format and rationale (Phase-5
// persistence seed, RTK #53/#54). Dependency-free: <cstdio> + standard library.

#include "llm/session.h"

#include <cstdio>

namespace llm {

namespace {

constexpr uint32_t kMagic = 0x53495053u; // 'SIPS'
constexpr uint32_t kVersion = 1u;

// RAII wrapper so every early return closes the FILE* exactly once.
struct FileHandle {
  FILE *f = nullptr;
  explicit FileHandle(FILE *fp) : f(fp) {}
  ~FileHandle() {
    if (f)
      std::fclose(f);
  }
  FileHandle(const FileHandle &) = delete;
  FileHandle &operator=(const FileHandle &) = delete;
};

template <typename T> bool write_one(FILE *f, const T &v) {
  return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T> bool write_n(FILE *f, const T *p, size_t n) {
  if (n == 0)
    return true;
  return std::fwrite(p, sizeof(T), n, f) == n;
}

template <typename T> bool read_one(FILE *f, T &v) {
  return std::fread(&v, sizeof(T), 1, f) == 1;
}

template <typename T> bool read_n(FILE *f, T *p, size_t n) {
  if (n == 0)
    return true;
  return std::fread(p, sizeof(T), n, f) == n;
}

} // namespace

bool session_write(const std::string &path, const std::vector<int64_t> &tokens,
                   const KVCache &kv, int64_t seq_len, uint64_t model_id) {
  if (seq_len < 0)
    return false;

  FileHandle fh(std::fopen(path.c_str(), "wb"));
  if (!fh.f)
    return false;
  FILE *f = fh.f;

  const int64_t n_layers = kv.n_layers();
  const int64_t kv_dim = kv.kv_dim();

  // Header.
  if (!write_one(f, kMagic))
    return false;
  if (!write_one(f, kVersion))
    return false;
  if (!write_one(f, model_id))
    return false;
  if (!write_one(f, n_layers))
    return false;
  if (!write_one(f, kv_dim))
    return false;
  if (!write_one(f, seq_len))
    return false;

  // Tokens: write exactly seq_len entries. The caller always passes
  // tokens.size() == seq_len; guard against a short vector by zero-padding.
  for (int64_t i = 0; i < seq_len; ++i) {
    const int64_t tok =
        (static_cast<size_t>(i) < tokens.size()) ? tokens[i] : 0;
    if (!write_one(f, tok))
      return false;
  }

  // Keys, then values: for each layer, each position, kv_dim floats.
  for (int64_t layer = 0; layer < n_layers; ++layer) {
    for (int64_t pos = 0; pos < seq_len; ++pos) {
      if (!write_n(f, kv.k(layer, pos), static_cast<size_t>(kv_dim)))
        return false;
    }
  }
  for (int64_t layer = 0; layer < n_layers; ++layer) {
    for (int64_t pos = 0; pos < seq_len; ++pos) {
      if (!write_n(f, kv.v(layer, pos), static_cast<size_t>(kv_dim)))
        return false;
    }
  }

  return std::fflush(f) == 0;
}

bool session_read(const std::string &path, std::vector<int64_t> &tokens,
                  KVCache &kv, int64_t *seq_len_out,
                  uint64_t expected_model_id) {
  FileHandle fh(std::fopen(path.c_str(), "rb"));
  if (!fh.f)
    return false;
  FILE *f = fh.f;

  // Header + validation.
  uint32_t magic = 0, version = 0;
  int64_t n_layers = 0, kv_dim = 0, seq_len = 0;
  if (!read_one(f, magic) || magic != kMagic)
    return false;
  if (!read_one(f, version) || version != kVersion)
    return false;
  uint64_t model_id = 0;
  if (!read_one(f, model_id) || model_id != expected_model_id)
    return false;
  if (!read_one(f, n_layers) || n_layers != kv.n_layers())
    return false;
  if (!read_one(f, kv_dim) || kv_dim != kv.kv_dim())
    return false;
  if (!read_one(f, seq_len))
    return false;
  if (seq_len < 0 || seq_len > kv.max_ctx())
    return false;

  // Ensure the cache can hold seq_len positions before writing into it.
  kv.set_seq_len(seq_len);

  // Tokens.
  tokens.resize(static_cast<size_t>(seq_len));
  if (!read_n(f, tokens.data(), static_cast<size_t>(seq_len)))
    return false;

  // Keys, then values (same order as written).
  for (int64_t layer = 0; layer < n_layers; ++layer) {
    for (int64_t pos = 0; pos < seq_len; ++pos) {
      if (!read_n(f, kv.k(layer, pos), static_cast<size_t>(kv_dim)))
        return false;
    }
  }
  for (int64_t layer = 0; layer < n_layers; ++layer) {
    for (int64_t pos = 0; pos < seq_len; ++pos) {
      if (!read_n(f, kv.v(layer, pos), static_cast<size_t>(kv_dim)))
        return false;
    }
  }

  if (seq_len_out)
    *seq_len_out = seq_len;
  return true;
}

} // namespace llm
