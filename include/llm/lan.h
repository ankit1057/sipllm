// lan.h — Issue #70: pool RAM across machines on one network by splitting the
// model's layers between them.
//
// Topology (v1): the coordinator runs layers [0, a) plus the tokenizer,
// embeddings, final norm and LM head; each worker runs one contiguous range
// after it, in order, and together they cover every layer exactly once. Per
// token only the residual stream crosses the network (dim floats per hop).
// Every node keeps its own --ram-budget, pins only its own layers and holds KV
// only for them, so each machine donates a bounded slice of RAM.
//
// Security (v1): a worker listens on an explicit address only, and every
// connection must present a pre-shared token (SIPLLM_LAN_TOKEN or the file in
// SIPLLM_LAN_TOKEN_FILE; never argv), compared in constant time. All lengths
// read from the network are bounded before allocating. Traffic is
// authenticated but NOT encrypted: activations can reveal prompt content to
// anyone who can read the network.
#pragma once

#include "llm/transformer.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace llm {

class Runtime;

namespace lan {

constexpr uint32_t kMagic      = 0x4E4C5053u;      // "SPLN"
constexpr uint16_t kVersion    = 1;
constexpr uint32_t kMaxPayload = 64u << 20;        // hard cap on any frame body
constexpr size_t   kHeaderSize = 16;
constexpr size_t   kMinToken   = 16;

enum MsgType : uint16_t {
    MSG_HELLO = 1, MSG_HELLO_OK = 2, MSG_ERROR = 3, MSG_RUN = 4, MSG_RESULT = 5, MSG_BYE = 6,
};

// Everything that must match for a worker's arithmetic to equal the
// coordinator's (same file, same settings) plus the layer range it serves.
struct Fingerprint {
    uint64_t file_size = 0;
    uint64_t tensor_hash = 0;   // FNV-1a over every tensor's name/dtype/shape/offset/size
    uint32_t n_layers = 0, dim = 0, max_ctx = 0;
    uint8_t  kv_precision = 0, fast_quant = 0, residency = 0;
    uint32_t layer_begin = 0, layer_end = 0;
};
Fingerprint fingerprint(const Runtime& rt);
// "" when `have` can serve `want` (all fields equal); otherwise every difference.
std::string describe_mismatch(const Fingerprint& want, const Fingerprint& have);

// Header helpers (exposed for tests). parse_header rejects a wrong magic or
// version, an unknown type and any length above kMaxPayload.
void write_header(uint8_t* h, uint16_t type, uint32_t len);
bool parse_header(const uint8_t* h, uint16_t* type, uint32_t* len, std::string* err);
bool tokens_equal(const std::string& a, const std::string& b);   // constant time in length of b

// Pre-shared token from SIPLLM_LAN_TOKEN, else the file named by
// SIPLLM_LAN_TOKEN_FILE. Throws if absent or shorter than kMinToken.
std::string token_from_env();

struct WorkerConfig {
    std::string bind_ip;   // required, e.g. 192.168.1.20 or 127.0.0.1
    int port = 0;          // 0 = pick a free port (tests)
    std::string token;
};
// Serve coordinators one at a time, forever (or until *stop). `rt` must have
// been built with the worker's layer range. on_listening receives the bound
// port. Throws only on setup failure (bad address, bind error).
void serve_worker(Runtime& rt, const WorkerConfig& wc, const std::atomic<bool>* stop = nullptr,
                  const std::function<void(int port)>& on_listening = {});

struct NodeSpec { std::string host; int port = 0; int begin = 0, end = 0; };
// "host:port:A-B[,host:port:C-D...]" -> sorted, validated list.
std::vector<NodeSpec> parse_nodes(const std::string& s);

// Coordinator side: one connection to one worker.
class RemoteExecutor : public LayerExecutor {
public:
    RemoteExecutor(const NodeSpec& node, const std::string& token, const Fingerprint& expect);
    ~RemoteExecutor() override;
    void run(int64_t begin, int64_t end, float* resid, int64_t n, int64_t start_pos) override;
    uint64_t bytes_sent() const { return sent_; }
    uint64_t bytes_received() const { return recv_; }
    double   wait_ms() const { return wait_ms_; }   // send -> result, i.e. network + remote compute
    uint64_t calls() const { return calls_; }
private:
    int fd_ = -1;
    NodeSpec node_;
    uint32_t dim_ = 0;
    uint64_t sent_ = 0, recv_ = 0, calls_ = 0;
    double wait_ms_ = 0;
    std::vector<uint8_t> buf_;
};

} // namespace lan
} // namespace llm
