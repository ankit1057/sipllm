// lan.cpp — Issue #70: split a model's layers across machines (see lan.h).
#include "llm/lan.h"
#include "llm/common.h"
#include "llm/neon.h"
#include "llm/runtime.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sstream>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace llm {
namespace lan {

namespace {

#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;   // Linux/Android: no SIGPIPE on a dead peer
#else
constexpr int kSendFlags = 0;              // macOS: SO_NOSIGPIPE is set per socket
#endif

void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
uint32_t get32(const uint8_t* p) { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)p[i] << (8 * i); return v; }
uint64_t get64(const uint8_t* p) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i); return v; }

void socket_opts(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);   // one small frame per hop
#if defined(SO_NOSIGPIPE)
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
}

void set_recv_timeout(int fd, int seconds) {
    timeval tv{seconds, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
}

bool send_all(int fd, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (n > 0) {
        ssize_t w = ::send(fd, p, n, kSendFlags);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}

bool recv_all(int fd, void* data, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(data);
    while (n > 0) {
        ssize_t r = ::recv(fd, p, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;   // closed, error or timeout
        p += r; n -= (size_t)r;
    }
    return true;
}

bool send_frame(int fd, uint16_t type, const void* payload, uint32_t len) {
    uint8_t h[kHeaderSize];
    write_header(h, type, len);
    return send_all(fd, h, sizeof h) && (len == 0 || send_all(fd, payload, len));
}

// Read one frame; the body is bounded by kMaxPayload (parse_header) AND by the
// caller's `max_len` before any allocation.
bool recv_frame(int fd, uint16_t* type, std::vector<uint8_t>& body, uint32_t max_len, std::string* err) {
    uint8_t h[kHeaderSize];
    if (!recv_all(fd, h, sizeof h)) { *err = "connection closed"; return false; }
    uint32_t len = 0;
    if (!parse_header(h, type, &len, err)) return false;
    if (len > max_len) { *err = "frame too large for its type"; return false; }
    body.resize(len);
    if (len && !recv_all(fd, body.data(), len)) { *err = "truncated frame"; return false; }
    return true;
}

void send_error(int fd, const std::string& msg) {
    const std::string m = msg.substr(0, 1024);
    send_frame(fd, MSG_ERROR, m.data(), (uint32_t)m.size());
}

constexpr size_t kFpBytes = 8 + 8 + 4 * 3 + 3 + 4 * 2;   // serialized Fingerprint

void put_fp(uint8_t* p, const Fingerprint& f) {
    put64(p, f.file_size); put64(p + 8, f.tensor_hash);
    put32(p + 16, f.n_layers); put32(p + 20, f.dim); put32(p + 24, f.max_ctx);
    p[28] = f.kv_precision; p[29] = f.fast_quant; p[30] = f.residency;
    put32(p + 31, f.layer_begin); put32(p + 35, f.layer_end);
}
Fingerprint get_fp(const uint8_t* p) {
    Fingerprint f;
    f.file_size = get64(p); f.tensor_hash = get64(p + 8);
    f.n_layers = get32(p + 16); f.dim = get32(p + 20); f.max_ctx = get32(p + 24);
    f.kv_precision = p[28]; f.fast_quant = p[29]; f.residency = p[30];
    f.layer_begin = get32(p + 31); f.layer_end = get32(p + 35);
    return f;
}

// Process peak RSS (kernel high-water mark): what this machine actually donated.
double peak_rss_mb() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return ru.ru_maxrss / 1e6;            // bytes on macOS
#else
    return ru.ru_maxrss * 1024.0 / 1e6;   // kilobytes on Linux/Android
#endif
}

uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

} // namespace

// ---- header / token / fingerprint ------------------------------------------

void write_header(uint8_t* h, uint16_t type, uint32_t len) {
    put32(h, kMagic);
    h[4] = (uint8_t)kVersion; h[5] = (uint8_t)(kVersion >> 8);
    h[6] = (uint8_t)type;     h[7] = (uint8_t)(type >> 8);
    put32(h + 8, len);
    put32(h + 12, 0);
}

bool parse_header(const uint8_t* h, uint16_t* type, uint32_t* len, std::string* err) {
    if (get32(h) != kMagic) { *err = "not a SipLLM LAN peer (bad magic)"; return false; }
    const uint16_t ver = (uint16_t)(h[4] | (h[5] << 8));
    if (ver != kVersion) { *err = "protocol version mismatch"; return false; }
    *type = (uint16_t)(h[6] | (h[7] << 8));
    if (*type < MSG_HELLO || *type > MSG_BYE) { *err = "unknown message type"; return false; }
    *len = get32(h + 8);
    if (*len > kMaxPayload) { *err = "frame exceeds the protocol limit"; return false; }
    return true;
}

bool tokens_equal(const std::string& a, const std::string& b) {
    // Runs over all of b regardless of where a differs; a length mismatch still
    // compares b's bytes against a zero pad.
    unsigned char diff = (unsigned char)(a.size() != b.size());
    for (size_t i = 0; i < b.size(); ++i)
        diff |= (unsigned char)((i < a.size() ? (unsigned char)a[i] : 0) ^ (unsigned char)b[i]);
    return diff == 0;
}

std::string token_from_env() {
    std::string t;
    if (const char* e = std::getenv("SIPLLM_LAN_TOKEN")) t = e;
    else if (const char* f = std::getenv("SIPLLM_LAN_TOKEN_FILE")) {
        std::ifstream in(f);
        LLM_CHECK(in.good(), std::string("cannot read SIPLLM_LAN_TOKEN_FILE: ") + f);
        std::getline(in, t);
    }
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
    LLM_CHECK(!t.empty(), "LAN mode needs a pre-shared token: set SIPLLM_LAN_TOKEN (or "
                          "SIPLLM_LAN_TOKEN_FILE) to the same secret on every machine");
    LLM_CHECK(t.size() >= kMinToken, "SIPLLM_LAN_TOKEN must be at least 16 characters");
    return t;
}

Fingerprint fingerprint(const Runtime& rt) {
    Fingerprint f;
    const WeightSource& src = rt.source();
    f.file_size = src.file_size();
    uint64_t h = 1469598103934665603ull;
    for (const TensorInfo& t : src.tensors()) {
        h = fnv1a(h, t.name.data(), t.name.size());
        const uint32_t dt = (uint32_t)t.dtype;
        h = fnv1a(h, &dt, sizeof dt);
        for (int64_t d : t.shape) h = fnv1a(h, &d, sizeof d);
        h = fnv1a(h, &t.offset, sizeof t.offset);
        h = fnv1a(h, &t.nbytes, sizeof t.nbytes);
    }
    f.tensor_hash = h;
    f.n_layers = (uint32_t)rt.config().n_layers;
    f.dim = (uint32_t)rt.config().dim;
    f.max_ctx = (uint32_t)rt.max_ctx();
    f.kv_precision = (uint8_t)rt.kv_precision();
    f.fast_quant = (uint8_t)fast_quant_enabled();   // effective: requested AND kernel present
    f.residency = (uint8_t)rt.residency();
    f.layer_begin = (uint32_t)rt.layer_begin();
    f.layer_end = (uint32_t)rt.layer_end();
    return f;
}

std::string describe_mismatch(const Fingerprint& w, const Fingerprint& h) {
    std::ostringstream o;
    auto chk = [&](const char* what, uint64_t a, uint64_t b) {
        if (a != b) o << " " << what << " (coordinator " << a << ", worker " << b << ");";
    };
    chk("model file size", w.file_size, h.file_size);
    chk("model tensor table", w.tensor_hash, h.tensor_hash);
    chk("n_layers", w.n_layers, h.n_layers);
    chk("dim", w.dim, h.dim);
    chk("context (--ctx)", w.max_ctx, h.max_ctx);
    chk("KV precision", w.kv_precision, h.kv_precision);
    chk("int8 kernels (--fast, and the CPU must have them)", w.fast_quant, h.fast_quant);
    chk("residency", w.residency, h.residency);
    chk("layer range begin", w.layer_begin, h.layer_begin);
    chk("layer range end", w.layer_end, h.layer_end);
    return o.str();
}

// ---- --nodes parsing -------------------------------------------------------

std::vector<NodeSpec> parse_nodes(const std::string& s) {
    std::vector<NodeSpec> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // host:port:A-B  (host may not contain ':' in v1 — IPv4 or a name)
        const size_t c2 = item.rfind(':');
        const size_t c1 = c2 == std::string::npos ? std::string::npos : item.rfind(':', c2 - 1);
        const size_t dash = c2 == std::string::npos ? std::string::npos : item.find('-', c2);
        LLM_CHECK(c1 != std::string::npos && c1 > 0 && dash != std::string::npos,
                  "--nodes entry must be host:port:A-B, got '" + item + "'");
        NodeSpec n;
        n.host = item.substr(0, c1);
        n.port = std::stoi(item.substr(c1 + 1, c2 - c1 - 1));
        n.begin = std::stoi(item.substr(c2 + 1, dash - c2 - 1));
        n.end = std::stoi(item.substr(dash + 1));
        LLM_CHECK(n.port > 0 && n.port < 65536, "--nodes: bad port in '" + item + "'");
        LLM_CHECK(n.begin >= 0 && n.begin < n.end, "--nodes: bad layer range in '" + item + "' (A-B, end exclusive)");
        out.push_back(n);
    }
    LLM_CHECK(!out.empty(), "--nodes: no workers given");
    std::sort(out.begin(), out.end(), [](const NodeSpec& a, const NodeSpec& b) { return a.begin < b.begin; });
    for (size_t i = 1; i < out.size(); ++i)
        LLM_CHECK(out[i].begin == out[i - 1].end, "--nodes: worker ranges must be contiguous, no gaps or overlaps");
    return out;
}

// ---- worker ----------------------------------------------------------------

void serve_worker(Runtime& rt, const WorkerConfig& wc, const std::atomic<bool>* stop,
                  const std::function<void(int)>& on_listening) {
    LLM_CHECK(!wc.bind_ip.empty(), "worker: an explicit --listen address is required");
    LLM_CHECK(wc.token.size() >= kMinToken, "worker: token too short");
    const Fingerprint mine = fingerprint(rt);
    const uint32_t dim = mine.dim;

    int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    LLM_CHECK(lfd >= 0, "worker: socket() failed");
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)wc.port);
    LLM_CHECK(inet_pton(AF_INET, wc.bind_ip.c_str(), &addr.sin_addr) == 1,
              "worker: --listen needs an IPv4 address, got '" + wc.bind_ip + "'");
    if (::bind(lfd, (sockaddr*)&addr, sizeof addr) != 0 || ::listen(lfd, 1) != 0) {
        ::close(lfd);
        throw Error("worker: cannot listen on " + wc.bind_ip + ":" + std::to_string(wc.port) +
                    " (" + std::strerror(errno) + ")");
    }
    socklen_t al = sizeof addr;
    getsockname(lfd, (sockaddr*)&addr, &al);
    const int port = ntohs(addr.sin_port);
    if (wc.bind_ip == "0.0.0.0")
        fprintf(stderr, "worker: WARNING listening on every interface; prefer this machine's LAN address\n");
    fprintf(stderr, "worker: layers [%u, %u) of %u, listening on %s:%d (authenticated, not encrypted)\n",
            mine.layer_begin, mine.layer_end, mine.n_layers, wc.bind_ip.c_str(), port);
    if (on_listening) on_listening(port);

    std::vector<uint8_t> body, out;
    // A RUN body: begin u32, end u32, start_pos u64, n u32, dim u32, then n*dim floats.
    constexpr uint32_t kRunHead = 24;
    while (!(stop && stop->load())) {
        // Wait for a connection with poll() so a stop flag is honored (tests);
        // SO_RCVTIMEO does not bound accept() on every platform (macOS).
        pollfd pfd{lfd, POLLIN, 0};
        const int pr = ::poll(&pfd, 1, 1000);
        if (pr <= 0 || !(pfd.revents & POLLIN)) continue;
        sockaddr_in peer{};
        socklen_t pl = sizeof peer;
        int fd = ::accept(lfd, (sockaddr*)&peer, &pl);
        if (fd < 0) continue;
        socket_opts(fd);
        char pip[INET_ADDRSTRLEN] = "?";
        inet_ntop(AF_INET, &peer.sin_addr, pip, sizeof pip);
        std::string err;
        uint16_t type = 0;

        // Handshake: 10 s to present a valid HELLO.
        set_recv_timeout(fd, 10);
        bool ok = recv_frame(fd, &type, body, 2 + 256 + (uint32_t)kFpBytes, &err) && type == MSG_HELLO;
        if (ok) {
            const uint16_t tl = (uint16_t)(body.size() >= 2 ? body[0] | (body[1] << 8) : 0);
            ok = body.size() == 2u + tl + kFpBytes && tl <= 256;
            if (!ok) err = "malformed HELLO";
            else {
                const std::string tok(reinterpret_cast<const char*>(body.data() + 2), tl);
                if (!tokens_equal(tok, wc.token)) { ok = false; err = "authentication failed"; }
                else {
                    const Fingerprint want = get_fp(body.data() + 2 + tl);
                    const std::string mm = describe_mismatch(want, mine);
                    if (!mm.empty()) { ok = false; err = "configuration mismatch:" + mm; }
                }
            }
        } else if (err.empty()) err = "expected HELLO";
        if (!ok) {
            fprintf(stderr, "worker: refused %s: %s\n", pip, err.c_str());
            send_error(fd, err);
            ::close(fd);
            continue;
        }
        uint8_t fpb[kFpBytes];
        put_fp(fpb, mine);
        send_frame(fd, MSG_HELLO_OK, fpb, sizeof fpb);
        fprintf(stderr, "worker: serving coordinator %s\n", pip);

        // Serve RUN requests until BYE / disconnect. v1 sets no read timeout
        // after the authenticated handshake (a person may pause between prompts),
        // so a connected coordinator holds this worker until it disconnects.
        set_recv_timeout(fd, 0);
        const uint32_t max_run = kMaxPayload;
        while (true) {
            if (!recv_frame(fd, &type, body, max_run, &err)) break;
            if (type == MSG_BYE) break;
            if (type != MSG_RUN || body.size() < kRunHead) { send_error(fd, "expected RUN"); break; }
            const uint32_t b = get32(body.data()), e = get32(body.data() + 4);
            const uint64_t sp = get64(body.data() + 8);
            const uint32_t n = get32(body.data() + 16), d = get32(body.data() + 20);
            const uint64_t want_len = (uint64_t)kRunHead + (uint64_t)n * d * sizeof(float);
            if (b != mine.layer_begin || e != mine.layer_end) { send_error(fd, "RUN for layers this worker does not own"); break; }
            if (d != dim || n == 0 || want_len != body.size()) { send_error(fd, "RUN size/shape mismatch"); break; }
            if (sp + n > mine.max_ctx) { send_error(fd, "RUN positions exceed the context window"); break; }
            float* resid = reinterpret_cast<float*>(body.data() + kRunHead);   // 24-byte offset: 4-aligned
            try {
                rt.run_layers(b, e, resid, (int64_t)n, (int64_t)sp);
            } catch (const std::exception& ex) {
                send_error(fd, std::string("worker error: ") + ex.what());
                break;
            }
            if (!send_frame(fd, MSG_RESULT, resid, n * d * (uint32_t)sizeof(float))) break;
        }
        fprintf(stderr, "worker: coordinator %s disconnected; peak RSS so far %.0f MB, now %.0f MB\n",
                pip, peak_rss_mb(), current_rss_bytes() / 1e6);
        ::close(fd);
    }
    ::close(lfd);
}

// ---- coordinator -----------------------------------------------------------

RemoteExecutor::RemoteExecutor(const NodeSpec& node, const std::string& token, const Fingerprint& expect)
    : node_(node), dim_(expect.dim) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    const std::string port = std::to_string(node.port);
    LLM_CHECK(getaddrinfo(node.host.c_str(), port.c_str(), &hints, &res) == 0 && res,
              "cannot resolve worker " + node.host);
    fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    const bool conn = fd_ >= 0 && ::connect(fd_, res->ai_addr, res->ai_addrlen) == 0;
    freeaddrinfo(res);
    const std::string who = node.host + ":" + port;
    if (!conn) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        throw Error("cannot connect to worker " + who + " (" + std::strerror(errno) + ")");
    }
    socket_opts(fd_);
    set_recv_timeout(fd_, 10);

    Fingerprint want = expect;
    want.layer_begin = (uint32_t)node.begin;
    want.layer_end = (uint32_t)node.end;
    LLM_CHECK(token.size() <= 256, "token too long");
    std::vector<uint8_t> hello(2 + token.size() + kFpBytes);
    hello[0] = (uint8_t)token.size(); hello[1] = (uint8_t)(token.size() >> 8);
    std::memcpy(hello.data() + 2, token.data(), token.size());
    put_fp(hello.data() + 2 + token.size(), want);
    std::string err;
    uint16_t type = 0;
    std::vector<uint8_t> body;
    if (!send_frame(fd_, MSG_HELLO, hello.data(), (uint32_t)hello.size()) ||
        !recv_frame(fd_, &type, body, 1024, &err)) {
        ::close(fd_); fd_ = -1;
        throw Error("worker " + who + ": handshake failed (" + (err.empty() ? "send" : err) + ")");
    }
    if (type == MSG_ERROR) {
        ::close(fd_); fd_ = -1;
        throw Error("worker " + who + " refused: " + std::string(body.begin(), body.end()));
    }
    if (type != MSG_HELLO_OK || body.size() != kFpBytes) {
        ::close(fd_); fd_ = -1;
        throw Error("worker " + who + ": unexpected handshake reply");
    }
    set_recv_timeout(fd_, 300);   // generous: a big remote prefill can take a while
    fprintf(stderr, "coordinator: worker %s serves layers [%d, %d)\n", who.c_str(), node.begin, node.end);
}

RemoteExecutor::~RemoteExecutor() {
    if (fd_ >= 0) {
        send_frame(fd_, MSG_BYE, nullptr, 0);
        ::close(fd_);
    }
}

void RemoteExecutor::run(int64_t begin, int64_t end, float* resid, int64_t n, int64_t start_pos) {
    LLM_CHECK(fd_ >= 0, "worker connection is closed");
    LLM_CHECK(begin == node_.begin && end == node_.end, "RemoteExecutor: wrong layer range");
    const uint64_t payload = (uint64_t)n * dim_ * sizeof(float);
    LLM_CHECK(24 + payload <= kMaxPayload, "batch too large for one LAN frame; lower the prompt chunk");
    buf_.resize(24);
    put32(buf_.data(), (uint32_t)begin); put32(buf_.data() + 4, (uint32_t)end);
    put64(buf_.data() + 8, (uint64_t)start_pos);
    put32(buf_.data() + 16, (uint32_t)n); put32(buf_.data() + 20, dim_);
    uint8_t h[kHeaderSize];
    write_header(h, MSG_RUN, (uint32_t)(24 + payload));
    const auto t0 = std::chrono::steady_clock::now();
    const std::string who = node_.host + ":" + std::to_string(node_.port);
    if (!send_all(fd_, h, sizeof h) || !send_all(fd_, buf_.data(), 24) || !send_all(fd_, resid, payload))
        throw Error("lost connection to worker " + who);
    uint16_t type = 0;
    std::string err;
    std::vector<uint8_t> body;
    if (!recv_frame(fd_, &type, body, (uint32_t)payload + 1024, &err))
        throw Error("worker " + who + ": " + err);
    if (type == MSG_ERROR) throw Error("worker " + who + ": " + std::string(body.begin(), body.end()));
    if (type != MSG_RESULT || body.size() != payload) throw Error("worker " + who + ": bad RESULT");
    std::memcpy(resid, body.data(), payload);
    wait_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    sent_ += kHeaderSize + 24 + payload;
    recv_ += kHeaderSize + payload;
    ++calls_;
}

} // namespace lan
} // namespace llm
