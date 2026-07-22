// test_util.h — a dependency-free test harness.
//
// No gtest, no catch2 — the project's rule is "no external libraries", so we
// roll a ~40-line runner. Tests register themselves via TEST(name){...} and
// main() runs them all, printing a summary and returning non-zero on failure.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace llmtest {

// A writable per-run scratch directory for test fixtures. Honors $TMPDIR and
// falls back to /tmp; the tests write tiny synthetic model files here.
inline std::string scratch_path(const std::string& name) {
    const char* base = std::getenv("TMPDIR");
    std::string dir = std::string(base && *base ? base : "/tmp") + "/llm-tests";
    ::mkdir(dir.c_str(), 0777);
    return dir + "/" + name;
}

struct Case { std::string name; std::function<void()> fn; };

inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }

struct Registrar {
    Registrar(const std::string& n, std::function<void()> f) {
        registry().push_back({n, std::move(f)});
    }
};

struct Failure { std::string msg; };

inline void check(bool cond, const std::string& msg) {
    if (!cond) throw Failure{msg};
}

inline void approx(double got, double want, double tol, const std::string& what) {
    if (std::isnan(got) || std::abs(got - want) > tol) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: got %.8g want %.8g (tol %.3g)",
                 what.c_str(), got, want, tol);
        throw Failure{buf};
    }
}

inline int run_all() {
    int passed = 0, failed = 0;
    for (auto& c : registry()) {
        try {
            c.fn();
            printf("  \x1b[32mPASS\x1b[0m  %s\n", c.name.c_str());
            ++passed;
        } catch (const Failure& f) {
            printf("  \x1b[31mFAIL\x1b[0m  %s\n        %s\n", c.name.c_str(),
                   f.msg.c_str());
            ++failed;
        } catch (const std::exception& e) {
            printf("  \x1b[31mFAIL\x1b[0m  %s\n        exception: %s\n",
                   c.name.c_str(), e.what());
            ++failed;
        }
    }
    printf("\n%d passed, %d failed, %d total\n", passed, failed,
           passed + failed);
    return failed == 0 ? 0 : 1;
}

} // namespace llmtest

#define TEST(name)                                                             \
    static void name();                                                        \
    static ::llmtest::Registrar reg_##name(#name, name);                       \
    static void name()

#define CHECK(cond) ::llmtest::check((cond), #cond)
#define CHECK_MSG(cond, msg) ::llmtest::check((cond), (msg))
#define APPROX(got, want, tol) ::llmtest::approx((got), (want), (tol), #got)
