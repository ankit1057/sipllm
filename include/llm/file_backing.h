// file_backing.h — POSIX file handle shared by every WeightSource.
//
// Owns an fd and offers two access modes:
//   * pread(offset, dst, n)  — positional, streaming, thread-safe. This is the
//     primary path (Phase 6: "No mmap initially. Use pread()."). Positional
//     reads let the storage thread pull layer N+1 while inference reads other
//     regions, with no shared file cursor to serialize on.
//   * mmap_base()            — optional demand-paged view for the mmap variant.
#pragma once

#include "llm/common.h"

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace llm {

class FileBacking {
public:
    explicit FileBacking(const std::string& path, bool use_mmap = false) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        LLM_CHECK(fd_ >= 0, "open failed: " + path);
        struct stat st{};
        LLM_CHECK(::fstat(fd_, &st) == 0, "fstat failed: " + path);
        size_ = static_cast<uint64_t>(st.st_size);
        path_ = path;
        if (use_mmap) enable_mmap();
    }

    ~FileBacking() {
        if (map_) ::munmap(const_cast<uint8_t*>(map_), size_);
        if (fd_ >= 0) ::close(fd_);
    }

    FileBacking(const FileBacking&) = delete;
    FileBacking& operator=(const FileBacking&) = delete;

    // Positional read of exactly n bytes at absolute offset. Loops over short
    // reads; throws on EOF/error. Safe to call concurrently from many threads.
    void pread_exact(uint64_t offset, void* dst, uint64_t n) const {
        uint8_t* p = static_cast<uint8_t*>(dst);
        uint64_t got = 0;
        while (got < n) {
            ssize_t r = ::pread(fd_, p + got, n - got, offset + got);
            if (r < 0) {
                if (errno == EINTR) continue;
                throw Error("pread failed on " + path_ + ": " + std::strerror(errno));
            }
            LLM_CHECK(r != 0, "pread hit EOF (file truncated?): " + path_);
            got += static_cast<uint64_t>(r);
        }
    }

    void enable_mmap() {
        if (map_) return;
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        LLM_CHECK(p != MAP_FAILED, "mmap failed: " + path_);
        map_ = static_cast<const uint8_t*>(p);
        // Sequential-ish access with the kernel doing readahead/eviction.
        ::madvise(const_cast<uint8_t*>(map_), size_, MADV_RANDOM);
    }

    // Hint the kernel to start pulling a region into the page cache (used by
    // the prefetch thread when running in mmap mode).
    void prefetch(uint64_t offset, uint64_t n) const {
#if defined(POSIX_FADV_WILLNEED)
        ::posix_fadvise(fd_, offset, n, POSIX_FADV_WILLNEED);
#endif
        if (map_) ::madvise(const_cast<uint8_t*>(map_) + offset, n, MADV_WILLNEED);
    }

    int fd() const { return fd_; }
    uint64_t size() const { return size_; }
    const uint8_t* map_base() const { return map_; }
    const std::string& path() const { return path_; }

private:
    int fd_ = -1;
    uint64_t size_ = 0;
    const uint8_t* map_ = nullptr;
    std::string path_;
};

} // namespace llm
