// session.h — Phase-5 persistent context (session save/restore seed, RTK
// #53/#54).
//
// Serializes a Runtime's committed prefix — the tokens whose keys/values are
// resident in the KV cache at positions [0, seq_len) — to a flat binary file,
// so cross-turn context reuse can survive a process restart. This is the
// on-disk seed the Runtime glue (save_session/load_session) builds on; it is
// opt-in and changes no default behavior.
//
// File format ("SIPS" v1, binary, little-endian, written sequentially):
//   uint32 magic   = 0x53495053   // 'SIPS'
//   uint32 version = 1
//   uint64 model_id              // opaque caller-supplied model identity guard
//   int64  n_layers
//   int64  kv_dim
//   int64  seq_len                // committed positions (== token count)
//   int64  tokens[seq_len]        // committed token ids
//   float  k[n_layers*seq_len*kv_dim]  // layer 0..n_layers-1, pos
//   0..seq_len-1, kv_dim floats float  v[n_layers*seq_len*kv_dim]  // same
//   order (all k first, then all v)
//
// On read every header field is validated against the target KVCache's dims and
// max_ctx; any mismatch, bad magic/version, or short/failed IO returns false
// (never throws). Uses only KVCache's public accessors — KVCache is not
// modified.
#pragma once

#include "llm/kv_cache.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// Write committed tokens + resident KV [0,seq_len) to `path`. false on IO
// error.
bool session_write(const std::string &path, const std::vector<int64_t> &tokens,
                   const KVCache &kv, int64_t seq_len, uint64_t model_id);

// Read a session file: fill `tokens`, write KV for [0,seq_len), set
// *seq_len_out. Returns false on IO error, bad magic/version, or if file
// (n_layers,kv_dim) do not match `kv`'s dims, or if seq_len > kv.max_ctx(). On
// failure the caller treats the runtime as needing reset(); partial
// modification is acceptable.
bool session_read(const std::string &path, std::vector<int64_t> &tokens,
                  KVCache &kv, int64_t *seq_len_out,
                  uint64_t expected_model_id);

} // namespace llm
