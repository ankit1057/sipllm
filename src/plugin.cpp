// plugin.cpp — PluginHost lifecycle + graceful fallback (see plugin.h).
#include "llm/plugin.h"

#include <cstdio>
#include <exception>

namespace llm {

PluginHost::~PluginHost() { shutdown(); }

void PluginHost::set_kosh(std::unique_ptr<KoshPlugin> k) {
  kosh_ = std::move(k);
}
void PluginHost::set_rtk(std::unique_ptr<RtkPlugin> r) { rtk_ = std::move(r); }

bool PluginHost::init(const ModelConfig &cfg, RtkKvView initial) {
  if (inited_)
    return true;

  // Each plugin's init is guarded: a failure DISABLES that plugin (drop it and
  // log) rather than aborting the run — the caller then takes the identity
  // path.
  if (kosh_) {
    bool ok = false;
    try {
      ok = kosh_->init(cfg);
    } catch (const std::exception &e) {
      fprintf(stderr, "[plugin] kosh '%s' init threw: %s — disabled\n",
              kosh_->name(), e.what());
    } catch (...) {
      fprintf(stderr, "[plugin] kosh init threw (unknown) — disabled\n");
    }
    if (!ok)
      kosh_.reset();
  }

  if (rtk_) {
    bool ok = false;
    try {
      ok = rtk_->init(cfg, initial);
    } catch (const std::exception &e) {
      fprintf(stderr, "[plugin] rtk '%s' init threw: %s — disabled\n",
              rtk_->name(), e.what());
    } catch (...) {
      fprintf(stderr, "[plugin] rtk init threw (unknown) — disabled\n");
    }
    if (!ok)
      rtk_.reset();
  }

  inited_ = true;
  return true;
}

void PluginHost::shutdown() {
  if (kosh_) {
    try {
      kosh_->shutdown();
    } catch (...) {
    }
  }
  if (rtk_) {
    try {
      rtk_->shutdown();
    } catch (...) {
    }
  }
  inited_ = false;
}

} // namespace llm
