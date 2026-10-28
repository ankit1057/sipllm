#include "llm/device_profile.h"
#include "llm/common.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <array>
#include <memory>
#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <pwd.h>
#endif

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace llm {

// Helper to run a command and get output (for CPU string if sysctl not available on linux)
static std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
#if defined(__linux__) || defined(__APPLE__)
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    if (!result.empty() && result.back() == '\n') result.pop_back();
#endif
    return result;
}

HardwareInfo get_hardware_info() {
    HardwareInfo info;

#if defined(__APPLE__)
    char buffer[256];
    size_t size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, NULL, 0) == 0) {
        info.cpu_model = buffer;
    }
    
    int val = 0;
    size = sizeof(val);
    if (sysctlbyname("hw.physicalcpu", &val, &size, NULL, 0) == 0) info.physical_cores = val;
    if (sysctlbyname("hw.logicalcpu", &val, &size, NULL, 0) == 0) info.logical_cores = val;
    
    int64_t mem = 0;
    size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, NULL, 0) == 0) info.ram_bytes = mem;
    
    // hw.pagesize is 8 bytes on arm64 macOS: read it into a 64-bit value with its
    // own size (it used to land in the 4-byte `val` with size still 8 from
    // hw.memsize, overwriting 4 bytes of stack; ASan stack-buffer-overflow).
    int64_t page = 0;
    size = sizeof(page);
    if (sysctlbyname("hw.pagesize", &page, &size, NULL, 0) == 0) info.page_size = page;

#if defined(__aarch64__) || defined(_M_ARM64)
    info.architecture = "arm64";
#else
    info.architecture = "x86_64";
#endif

#elif defined(__linux__)
    info.cpu_model = exec("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'");
    if (info.cpu_model.empty()) info.cpu_model = "Unknown Linux CPU";

    info.logical_cores = sysconf(_SC_NPROCESSORS_ONLN);
    
    // Approximation for physical cores if lscpu not parsed
    std::string cores_str = exec("lscpu -p | egrep -v '^#' | sort -u -t, -k 2,2 | wc -l");
    if (!cores_str.empty()) {
        try { info.physical_cores = std::stoi(cores_str); } catch(...) { info.physical_cores = info.logical_cores; }
    } else {
        info.physical_cores = info.logical_cores;
    }

    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    info.page_size = page_size;
    info.ram_bytes = (size_t)pages * (size_t)page_size;

#if defined(__aarch64__)
    info.architecture = "arm64";
#else
    info.architecture = "x86_64";
#endif
#else
    info.cpu_model = "Unknown OS CPU";
    info.logical_cores = std::thread::hardware_concurrency();
    info.physical_cores = info.logical_cores;
    info.architecture = "unknown";
#endif

    // Fallbacks
    if (info.logical_cores <= 0) info.logical_cores = std::thread::hardware_concurrency();
    if (info.physical_cores <= 0) info.physical_cores = info.logical_cores;

    return info;
}

std::string HardwareInfo::hardware_id() const {
    // A deterministic string identifying the hardware for caching
    // Example: arm64_apple_m3_max_16c_128gb
    std::string model = cpu_model;
    for (char& c : model) {
        if (!isalnum(c)) c = '_';
        else c = tolower(c);
    }
    // collapse multiple underscores
    std::string clean_model;
    bool last_was_us = false;
    for (char c : model) {
        if (c == '_') {
            if (!last_was_us) clean_model += c;
            last_was_us = true;
        } else {
            clean_model += c;
            last_was_us = false;
        }
    }
    while (!clean_model.empty() && clean_model.back() == '_') clean_model.pop_back();
    while (!clean_model.empty() && clean_model.front() == '_') clean_model.erase(0, 1);

    size_t gb = ram_bytes / (1024 * 1024 * 1024);
    
    char buf[256];
    snprintf(buf, sizeof(buf), "%s_%s_%dc_%zugh", architecture.c_str(), clean_model.c_str(), logical_cores, gb);
    return std::string(buf);
}

std::string HardwareInfo::to_json(int indent) const {
    std::string ind(indent, ' ');
    std::stringstream ss;
    ss << ind << "{\n";
    ss << ind << "  \"cpu_model\": \"" << cpu_model << "\",\n";
    ss << ind << "  \"architecture\": \"" << architecture << "\",\n";
    ss << ind << "  \"logical_cores\": " << logical_cores << ",\n";
    ss << ind << "  \"physical_cores\": " << physical_cores << ",\n";
    ss << ind << "  \"ram_bytes\": " << ram_bytes << ",\n";
    ss << ind << "  \"page_size\": " << page_size << ",\n";
    ss << ind << "  \"hardware_id\": \"" << hardware_id() << "\"\n";
    ss << ind << "}";
    return ss.str();
}

std::string RuntimeProfile::to_json(int indent) const {
    std::string ind(indent, ' ');
    std::stringstream ss;
    ss << ind << "{\n";
    ss << ind << "  \"threads\": " << threads << ",\n";
    ss << ind << "  \"schedule_policy\": " << schedule_policy << ",\n";
    ss << ind << "  \"thread_barrier_ms\": " << thread_barrier_ms << ",\n";
    ss << ind << "  \"thread_throughput_relative\": " << thread_throughput_relative << "\n";
    ss << ind << "}";
    return ss.str();
}

size_t available_ram_bytes() {
#if defined(__APPLE__)
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) return 0;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return 0;
    return (size_t)(vm.free_count + vm.inactive_count + vm.speculative_count) * (size_t)page;
#elif defined(__linux__)
    std::ifstream in("/proc/meminfo");
    std::string line;
    while (std::getline(in, line)) {
        unsigned long long kb = 0;
        if (std::sscanf(line.c_str(), "MemAvailable: %llu kB", &kb) == 1) return (size_t)kb * 1024;
    }
    return 0;
#else
    return 0;
#endif
}

std::string get_sipllm_home() {
    std::string home;
#if defined(__linux__) || defined(__APPLE__)
    const char* h = getenv("HOME");
    if (h) home = h;
    else {
        struct passwd* pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
#endif
    if (home.empty()) home = ".";
    std::string dir = home + "/.sipllm";
#if defined(__linux__) || defined(__APPLE__)
    mkdir(dir.c_str(), 0755);
#endif
    return dir;
}

std::string get_hardware_profile_path() {
    return get_sipllm_home() + "/hardware.json";
}

std::string get_runtime_profile_path(const std::string& hw_id) {
    std::string runtime_dir = get_sipllm_home() + "/runtime";
#if defined(__linux__) || defined(__APPLE__)
    mkdir(runtime_dir.c_str(), 0755);
#endif
    return runtime_dir + "/" + hw_id + ".json";
}

bool load_hardware_profile(HardwareInfo& out) {
    std::ifstream in(get_hardware_profile_path());
    if (!in.is_open()) return false;
    // We can parse it minimally, but it's just a cache for the user.
    // In practice, we re-detect hardware quickly anyway, so loading it is optional.
    return false; // Skip full parsing for now since detection is < 5ms
}

bool load_runtime_profile(const std::string& hw_id, RuntimeProfile& out) {
    std::ifstream in(get_runtime_profile_path(hw_id));
    if (!in.is_open()) return false;
    
    std::string line;
    int found = 0;
    bool in_selected = false;
    while (std::getline(in, line)) {
        if (line.find("\"selected\":") != std::string::npos) {
            in_selected = true;
        }
        if (in_selected && line.find("\"threads\":") != std::string::npos) {
            auto pos = line.find(":");
            if (pos != std::string::npos && line.find("{") == std::string::npos) {
                try { out.threads = std::stoi(line.substr(pos + 1)); found++; } catch(...) {}
            }
        } else if (in_selected && line.find("\"schedule_policy\":") != std::string::npos) {
            auto pos = line.find(":");
            if (pos != std::string::npos && line.find("{") == std::string::npos) {
                try { out.schedule_policy = std::stoi(line.substr(pos + 1)); found++; } catch(...) {}
            }
        }
    }
    return found >= 2;
}

void save_hardware_profile(const HardwareInfo& info) {
    std::ofstream out(get_hardware_profile_path());
    if (out.is_open()) {
        out << "{\n";
        out << "  \"version\": 2,\n";
        out << "  \"created\": " << now_sec() << ",\n";
        out << "  \"hardware\": " << info.to_json(2) << "\n";
        out << "}\n";
    }
}

void save_runtime_profile(const std::string& hw_id, const RuntimeProfile& profile) {
    std::ofstream out(get_runtime_profile_path(hw_id));
    if (out.is_open()) {
        out << "{\n";
        out << "  \"version\": 2,\n";
        out << "  \"created\": " << now_sec() << ",\n";
#if defined(__clang__)
        out << "  \"compiler\": \"clang " << __clang_major__ << "." << __clang_minor__ << "\",\n";
#elif defined(__GNUC__)
        out << "  \"compiler\": \"gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "\",\n";
#else
        out << "  \"compiler\": \"unknown\",\n";
#endif
        out << "  \"sipllm_version\": \"M4-dev\",\n";
        out << "  \"hardware_id\": \"" << hw_id << "\",\n";
        out << "  \"measurements\": " << profile.measurements_json << ",\n";
        out << "  \"selected\": " << profile.to_json(2) << "\n";
        out << "}\n";
    }
}

} // namespace llm
