// plugin.h — PluginHost: lifecycle + graceful fallback for runtime plugins.
//
// The first seed of the Phase-2 plugin runtime (#51). Today it hosts two
// in-process plugins — Kosh (context intelligence, #52) and RTK (runtime/KV
// intelligence, #53) — behind stable interfaces. A later dynamic-loading /
// sandboxed / permissioned host (dlopen, capability manifest) can implement the
// SAME registration surface without changing callers.
//
// Contract:
//   - A null plugin means "not installed" — callers apply the identity path.
//   - init() fans out to installed plugins; a plugin whose init throws or
//     returns false is DISABLED (dropped) and logged, never fatal.
//   - Every call the Runtime makes into a plugin is exception-guarded at the
//     call site too, so a misbehaving plugin degrades to fallback, not a crash.
#pragma once

#include "llm/kosh.h"
#include "llm/rtk.h"

#include <memory>

namespace llm {

class PluginHost {
public:
  PluginHost() = default;
  ~PluginHost();
  PluginHost(const PluginHost &) = delete;
  PluginHost &operator=(const PluginHost &) = delete;

  // Install plugins (takes ownership). Passing nullptr clears the slot.
  void set_kosh(std::unique_ptr<KoshPlugin> k);
  void set_rtk(std::unique_ptr<RtkPlugin> r);

  // Accessors — may return nullptr (⇒ identity path). Valid only after a
  // successful init(); a plugin disabled during init() reads back as null.
  KoshPlugin *kosh() const { return kosh_.get(); }
  RtkPlugin *rtk() const { return rtk_.get(); }

  // Fan-out lifecycle. Idempotent: init() runs installed plugins' init()
  // exactly once; a plugin that fails is dropped. Safe to call with no
  // plugins installed (no-op).
  bool init(const ModelConfig &cfg, RtkKvView initial);
  bool inited() const { return inited_; }
  void shutdown();

private:
  std::unique_ptr<KoshPlugin> kosh_;
  std::unique_ptr<RtkPlugin> rtk_;
  bool inited_ = false;
};

} // namespace llm
