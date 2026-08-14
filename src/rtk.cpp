// rtk.cpp — RTK v0: runtime-state (KV) observer with metrics (see rtk.h).
//
// v0 is an OBSERVER only: it accumulates RtkMetrics from the KV views the
// runtime reports at prefill and each decode step. It does not touch the KV
// math — it establishes the runtime-state contract that Nishachar / SipMesh and
// later precision/persistence policies build on. Opt-in: default generation
// never constructs this plugin.
#include "llm/rtk.h"

namespace llm {
namespace {

class RtkV0 final : public RtkPlugin {
public:
  const char *name() const override { return "rtk-v0-observer"; }

  bool init(const ModelConfig &, RtkKvView initial) override {
    observe(initial);
    return true;
  }

  void on_prefill(RtkKvView view) override { observe(view); }
  void on_step(RtkKvView view) override {
    ++m_.steps;
    observe(view);
  }

  RtkMetrics metrics() const override { return m_; }

private:
  void observe(const RtkKvView &v) {
    if (v.seq_len > m_.max_seq)
      m_.max_seq = v.seq_len;
    if (v.bytes > m_.peak_kv_bytes)
      m_.peak_kv_bytes = v.bytes;
  }
  RtkMetrics m_;
};

} // namespace

std::unique_ptr<RtkPlugin> make_rtk_v0() { return std::make_unique<RtkV0>(); }

} // namespace llm
