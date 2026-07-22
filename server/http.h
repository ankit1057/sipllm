// http.h — a tiny, dependency-free HTTP/1.1 server with SSE support.
//
// Thread-per-connection, Connection: close, just enough to serve the SPA and
// stream tokens as Server-Sent Events. No external libraries (project rule).
#pragma once

#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <functional>
#include <map>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <cstdio>
#include <csignal>

namespace http {

inline std::string url_decode(const std::string& s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            int v = std::stoi(s.substr(i + 1, 2), nullptr, 16); o += (char)v; i += 2;
        } else if (s[i] == '+') o += ' ';
        else o += s[i];
    }
    return o;
}

struct Request {
    std::string method, path, query, body;
    std::map<std::string, std::string> headers;
    std::map<std::string, std::string> query_params;
    std::string q(const std::string& k, const std::string& def = "") const {
        auto it = query_params.find(k);
        return it == query_params.end() ? def : it->second;
    }
};

class Response {
public:
    explicit Response(int fd) : fd_(fd) {}

    void send(int status, const std::string& ctype, const std::string& body) {
        std::string h = "HTTP/1.1 " + std::to_string(status) + " OK\r\n";
        h += "Content-Type: " + ctype + "\r\n";
        h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        h += "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        write_all(h); write_all(body);
    }
    void begin_sse() {
        std::string h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n"
                        "Connection: close\r\n\r\n";
        write_all(h); sse_ = true;
    }
    // Returns false if the client has disconnected (so generation can stop).
    bool sse_event(const std::string& data, const std::string& event = "") {
        std::string m;
        if (!event.empty()) m += "event: " + event + "\n";
        // split data on newlines into multiple data: lines
        size_t start = 0;
        for (;;) {
            size_t nl = data.find('\n', start);
            m += "data: " + data.substr(start, nl == std::string::npos ? std::string::npos : nl - start) + "\n";
            if (nl == std::string::npos) break;
            start = nl + 1;
        }
        m += "\n";
        return write_all(m);
    }

private:
    bool write_all(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            ssize_t n = ::write(fd_, s.data() + off, s.size() - off);
            if (n <= 0) { if (errno == EINTR) continue; return false; }
            off += (size_t)n;
        }
        return true;
    }
    int fd_; bool sse_ = false;
};

class Server {
public:
    using Handler = std::function<void(Request&, Response&)>;
    explicit Server(int port) : port_(port) { signal(SIGPIPE, SIG_IGN); }

    void route(const std::string& method, const std::string& path, Handler h) {
        routes_[method + " " + path] = std::move(h);
    }
    void get(const std::string& p, Handler h) { route("GET", p, std::move(h)); }
    void post(const std::string& p, Handler h) { route("POST", p, std::move(h)); }

    int run() {
        int s = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1; ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{}; addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port_);
        if (::bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
        if (::listen(s, 16) < 0) { perror("listen"); return 1; }
        printf("listening on http://127.0.0.1:%d\n", port_);
        fflush(stdout);
        for (;;) {
            int c = ::accept(s, nullptr, nullptr);
            if (c < 0) continue;
            std::thread([this, c] { handle(c); ::close(c); }).detach();
        }
    }

private:
    void handle(int c) {
        std::string buf;
        char tmp[4096];
        // read headers
        size_t header_end = std::string::npos;
        while (header_end == std::string::npos) {
            ssize_t n = ::read(c, tmp, sizeof(tmp));
            if (n <= 0) return;
            buf.append(tmp, n);
            header_end = buf.find("\r\n\r\n");
            if (buf.size() > (1u << 20)) return;
        }
        Request req;
        parse(buf.substr(0, header_end), req);
        size_t body_start = header_end + 4;
        size_t clen = 0;
        auto it = req.headers.find("content-length");
        if (it != req.headers.end()) clen = std::stoul(it->second);
        std::string body = buf.substr(body_start);
        while (body.size() < clen) {
            ssize_t n = ::read(c, tmp, sizeof(tmp));
            if (n <= 0) break; body.append(tmp, n);
        }
        req.body = body;

        Response res(c);
        auto rit = routes_.find(req.method + " " + req.path);
        if (rit != routes_.end()) rit->second(req, res);
        else res.send(404, "text/plain", "not found");
    }

    static void parse(const std::string& head, Request& req) {
        size_t p = head.find("\r\n");
        std::string line = head.substr(0, p);
        size_t s1 = line.find(' '), s2 = line.find(' ', s1 + 1);
        req.method = line.substr(0, s1);
        std::string target = line.substr(s1 + 1, s2 - s1 - 1);
        size_t qm = target.find('?');
        if (qm == std::string::npos) req.path = target;
        else {
            req.path = target.substr(0, qm);
            req.query = target.substr(qm + 1);
            size_t i = 0;
            while (i < req.query.size()) {
                size_t amp = req.query.find('&', i);
                std::string kv = req.query.substr(i, amp == std::string::npos ? std::string::npos : amp - i);
                size_t eq = kv.find('=');
                if (eq != std::string::npos)
                    req.query_params[kv.substr(0, eq)] = url_decode(kv.substr(eq + 1));
                if (amp == std::string::npos) break; i = amp + 1;
            }
        }
        // headers
        size_t pos = p + 2;
        while (pos < head.size()) {
            size_t nl = head.find("\r\n", pos);
            std::string h = head.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
            size_t colon = h.find(':');
            if (colon != std::string::npos) {
                std::string k = h.substr(0, colon), v = h.substr(colon + 1);
                for (auto& ch : k) ch = (char)tolower(ch);
                while (!v.empty() && v[0] == ' ') v.erase(v.begin());
                req.headers[k] = v;
            }
            if (nl == std::string::npos) break; pos = nl + 2;
        }
    }

    int port_;
    std::map<std::string, Handler> routes_;
};

} // namespace http
