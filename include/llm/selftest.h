// selftest.h — runnable verification of all 11 roadmap tasks.
//
// Each task is exercised through the real code paths on freshly-generated toy
// fixtures and reports pass/fail + a one-line detail + timing. Surfaced by the
// web GUI's self-test panel and by `llm --selftest`.
#pragma once

#include <string>
#include <vector>

namespace llm {

struct SelfTestResult {
    int         task;      // 1..11
    std::string name;
    bool        ok;
    std::string detail;
    double      ms;
};

// Run all task checks, writing temporary fixtures under `tmpdir`.
std::vector<SelfTestResult> run_selftests(const std::string& tmpdir = "/tmp");

} // namespace llm
