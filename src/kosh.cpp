// kosh.cpp — Kosh v0: token-run collapse with accounting (see kosh.h).
//
// v0 capability: collapse any run of the same token id longer than `max_run_`
// down to `max_run_` copies. This targets the common "pasted padding" case —
// long runs of newline / space / filler tokens — and demonstrates the whole
// boundary (interception -> optimization -> accounting -> fallback) with a
// transform that is trivial to reason about. Opt-in: default generation never
// constructs this plugin. optimize() never throws.
#include "llm/kosh.h"

#include <algorithm>

namespace llm {
namespace {

class KoshV0 final : public KoshPlugin {
public:
  explicit KoshV0(int max_run) : max_run_(max_run < 1 ? 1 : max_run) {}

  const char *name() const override { return "kosh-v0-runcollapse"; }

  KoshResult optimize(const KoshRequest &req) override {
    KoshResult r;
    r.tokens_in = (int64_t)req.tokens.size();

    // Single pass: copy tokens, but never emit more than max_run_ consecutive
    // identical ids. Preserves order and every distinct transition.
    r.tokens.reserve(req.tokens.size());
    int64_t run_id = 0;
    int run_len = 0;
    bool have = false;
    for (int64_t t : req.tokens) {
      if (have && t == run_id) {
        if (++run_len <= max_run_)
          r.tokens.push_back(t);
        // else: within an over-long run — drop this copy.
      } else {
        run_id = t;
        run_len = 1;
        have = true;
        r.tokens.push_back(t);
      }
    }

    r.tokens_out = (int64_t)r.tokens.size();
    r.modified = (r.tokens_out != r.tokens_in);

    // Accounting.
    ++m_.requests;
    m_.tokens_in += (uint64_t)r.tokens_in;
    m_.tokens_out += (uint64_t)r.tokens_out;
    if (r.modified)
      ++m_.reductions;
    return r;
  }

  KoshMetrics metrics() const override { return m_; }

private:
  int max_run_;
  KoshMetrics m_;
};

// v1: collapse consecutive repeated blocks (see make_kosh_v1 in kosh.h).
class KoshV1 final : public KoshPlugin {
public:
  KoshV1(int keep, int max_block_len)
      : keep_(keep < 1 ? 1 : keep),
        max_block_len_(max_block_len < 1 ? 1 : max_block_len) {}

  const char *name() const override { return "kosh-v1-blockcollapse"; }

  KoshResult optimize(const KoshRequest &req) override {
    const std::vector<int64_t> &in = req.tokens;
    const int64_t n = (int64_t)in.size();
    KoshResult r;
    r.tokens_in = n;
    r.tokens.reserve(in.size());

    int64_t i = 0;
    while (i < n) {
      // Smallest-block-first: the smallest block length b in [1,max_block_len_]
      // whose block [i,i+b) equals the immediately following block [i+b,i+2b).
      int64_t chosen_b = 0;
      const int64_t bmax = std::min<int64_t>(max_block_len_, (n - i) / 2);
      for (int64_t b = 1; b <= bmax; ++b) {
        if (std::equal(in.begin() + i, in.begin() + i + b,
                       in.begin() + i + b)) {
          chosen_b = b;
          break;
        }
      }
      if (chosen_b == 0) {
        r.tokens.push_back(in[i]);
        ++i;
        continue;
      }
      const int64_t b = chosen_b;
      // Count the maximal consecutive run of this block starting at i.
      int64_t copies = 0;
      int64_t j = i;
      while (j + b <= n &&
             std::equal(in.begin() + i, in.begin() + i + b, in.begin() + j)) {
        ++copies;
        j += b;
      }
      const int64_t emit = std::min<int64_t>(copies, keep_);
      for (int64_t c = 0; c < emit; ++c)
        r.tokens.insert(r.tokens.end(), in.begin() + i, in.begin() + i + b);
      i += copies * b;
    }

    r.tokens_out = (int64_t)r.tokens.size();
    r.modified = (r.tokens_out != r.tokens_in);

    ++m_.requests;
    m_.tokens_in += (uint64_t)r.tokens_in;
    m_.tokens_out += (uint64_t)r.tokens_out;
    if (r.modified)
      ++m_.reductions;
    return r;
  }

  KoshMetrics metrics() const override { return m_; }

private:
  int keep_;
  int max_block_len_;
  KoshMetrics m_;
};

} // namespace

std::unique_ptr<KoshPlugin> make_kosh_v0(int max_run) {
  return std::make_unique<KoshV0>(max_run);
}

std::unique_ptr<KoshPlugin> make_kosh_v1(int keep, int max_block_len) {
  return std::make_unique<KoshV1>(keep, max_block_len);
}

} // namespace llm
