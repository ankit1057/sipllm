#include "llm/remote_weight_source.h"
#include "llm/remote_protocol.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace llm {

using namespace remote;

static void write_all(int fd, const void* buf, size_t count) {
    const char* p = (const char*)buf;
    while (count > 0) {
        ssize_t n = write(fd, p, count);
        if (n <= 0) throw std::runtime_error("Socket write failed");
        p += n;
        count -= n;
    }
}

static void read_all(int fd, void* buf, size_t count) {
    char* p = (char*)buf;
    while (count > 0) {
        ssize_t n = read(fd, p, count);
        if (n <= 0) throw std::runtime_error("Socket read failed");
        p += n;
        count -= n;
    }
}

static void write_str(int fd, const std::string& s) {
    uint32_t len = s.size();
    write_all(fd, &len, sizeof(len));
    write_all(fd, s.data(), len);
}

static std::string read_str(int fd) {
    uint32_t len;
    read_all(fd, &len, sizeof(len));
    std::string s(len, '\0');
    if (len > 0) read_all(fd, &s[0], len);
    return s;
}

RemoteWeightSource::RemoteWeightSource(const std::string& host, int port) {
    sock_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_ < 0) throw std::runtime_error("Failed to create socket");

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        throw std::runtime_error("Invalid address");
    }
    if (connect(sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        throw std::runtime_error("Connection failed");
    }

    cache_file_ = tmpfile();
    if (!cache_file_) throw std::runtime_error("Failed to create temp cache file");

    fetch_file_size();
    fetch_tensors();
}

RemoteWeightSource::~RemoteWeightSource() {
    if (sock_ >= 0) close(sock_);
    if (cache_file_) fclose(cache_file_);
}

void RemoteWeightSource::fetch_file_size() {
    std::lock_guard<std::mutex> lock(mu_);
    Cmd cmd = {CMD_FILE_SIZE, 0, 0, 0};
    write_all(sock_, &cmd, sizeof(cmd));
    read_all(sock_, &file_size_, sizeof(file_size_));
}

void RemoteWeightSource::fetch_tensors() {
    std::lock_guard<std::mutex> lock(mu_);
    Cmd cmd = {CMD_TENSORS, 0, 0, 0};
    write_all(sock_, &cmd, sizeof(cmd));
    
    uint32_t num_tensors = 0;
    read_all(sock_, &num_tensors, sizeof(num_tensors));
    tensors_.resize(num_tensors);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        tensors_[i].name = read_str(sock_);
        read_all(sock_, &tensors_[i].dtype, sizeof(tensors_[i].dtype));
        
        uint32_t rank = 0;
        read_all(sock_, &rank, sizeof(rank));
        tensors_[i].shape.resize(rank);
        if (rank > 0) {
            read_all(sock_, tensors_[i].shape.data(), rank * sizeof(int64_t));
        }
        read_all(sock_, &tensors_[i].offset, sizeof(tensors_[i].offset));
        read_all(sock_, &tensors_[i].nbytes, sizeof(tensors_[i].nbytes));
    }
}

const TensorInfo* RemoteWeightSource::find(const std::string& name) const {
    for (const auto& t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

void RemoteWeightSource::read_raw(const TensorInfo& t, void* dst) const {
    read_raw_at(t.offset, dst, t.nbytes);
}

void RemoteWeightSource::fetch_block(uint64_t block_idx) const {
    // caller holds mu_
    if (cached_blocks_[block_idx]) return;
    
    uint64_t offset = block_idx * BLOCK_SIZE;
    uint64_t size = BLOCK_SIZE;
    if (offset + size > file_size_) {
        size = file_size_ > offset ? file_size_ - offset : 0;
    }
    if (size == 0) return;

    Cmd cmd = {CMD_READ_RAW, 0, offset, size};
    write_all(sock_, &cmd, sizeof(cmd));
    
    std::vector<char> buf(size);
    read_all(sock_, buf.data(), size);
    
    fseek(cache_file_, offset, SEEK_SET);
    if (fwrite(buf.data(), 1, size, cache_file_) != size) {
        throw std::runtime_error("Failed to write to cache file");
    }
    cached_blocks_[block_idx] = true;
}

void RemoteWeightSource::read_raw_at(uint64_t offset, void* dst, uint64_t n) const {
    std::lock_guard<std::mutex> lock(mu_);
    
    uint64_t end = offset + n;
    uint64_t start_block = offset / BLOCK_SIZE;
    uint64_t end_block = (end + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (uint64_t b = start_block; b < end_block; ++b) {
        fetch_block(b);
    }
    
    fseek(cache_file_, offset, SEEK_SET);
    if (fread(dst, 1, n, cache_file_) != n) {
        throw std::runtime_error("Failed to read from cache file");
    }
}

bool RemoteWeightSource::has_meta(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (has_meta_cache_.count(key)) return has_meta_cache_[key];
    
    Cmd cmd = {CMD_HAS_META, (uint32_t)key.size(), 0, 0};
    write_all(sock_, &cmd, sizeof(cmd));
    write_all(sock_, key.data(), key.size());
    
    uint8_t has;
    read_all(sock_, &has, sizeof(has));
    has_meta_cache_[key] = (has != 0);
    return has_meta_cache_[key];
}

const MetaValue* RemoteWeightSource::meta(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (meta_cache_.count(key)) return &meta_cache_[key];
    
    Cmd cmd = {CMD_GET_META, (uint32_t)key.size(), 0, 0};
    write_all(sock_, &cmd, sizeof(cmd));
    write_all(sock_, key.data(), key.size());
    
    uint8_t has;
    read_all(sock_, &has, sizeof(has));
    if (!has) return nullptr;
    
    MetaValue val;
    read_all(sock_, &val.kind, sizeof(val.kind));
    read_all(sock_, &val.i, sizeof(val.i));
    read_all(sock_, &val.f, sizeof(val.f));
    val.s = read_str(sock_);
    
    uint32_t ia_len;
    read_all(sock_, &ia_len, sizeof(ia_len));
    val.ia.resize(ia_len);
    if (ia_len > 0) read_all(sock_, val.ia.data(), ia_len * sizeof(int64_t));
    
    uint32_t fa_len;
    read_all(sock_, &fa_len, sizeof(fa_len));
    val.fa.resize(fa_len);
    if (fa_len > 0) read_all(sock_, val.fa.data(), fa_len * sizeof(double));
    
    uint32_t sa_len;
    read_all(sock_, &sa_len, sizeof(sa_len));
    val.sa.resize(sa_len);
    for (uint32_t i = 0; i < sa_len; ++i) {
        val.sa[i] = read_str(sock_);
    }
    
    meta_cache_[key] = std::move(val);
    return &meta_cache_[key];
}

} // namespace llm
