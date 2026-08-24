#include "llm/runtime.h"
#include "llm/remote_protocol.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <vector>

using namespace llm;
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

static std::string read_str(int fd, uint32_t len) {
    std::string s(len, '\0');
    if (len > 0) read_all(fd, &s[0], len);
    return s;
}

void handle_client(int client_sock, const WeightSource* src) {
    try {
        while (true) {
            Cmd cmd;
            ssize_t n = read(client_sock, &cmd, sizeof(cmd));
            if (n == 0) break; // Client disconnected
            if (n < 0 || n != sizeof(cmd)) throw std::runtime_error("Invalid command read");

            if (cmd.type == CMD_FILE_SIZE) {
                uint64_t sz = src->file_size();
                write_all(client_sock, &sz, sizeof(sz));
            } else if (cmd.type == CMD_TENSORS) {
                const auto& tensors = src->tensors();
                uint32_t num_tensors = tensors.size();
                write_all(client_sock, &num_tensors, sizeof(num_tensors));
                for (const auto& t : tensors) {
                    write_str(client_sock, t.name);
                    write_all(client_sock, &t.dtype, sizeof(t.dtype));
                    uint32_t rank = t.shape.size();
                    write_all(client_sock, &rank, sizeof(rank));
                    if (rank > 0) {
                        write_all(client_sock, t.shape.data(), rank * sizeof(int64_t));
                    }
                    write_all(client_sock, &t.offset, sizeof(t.offset));
                    write_all(client_sock, &t.nbytes, sizeof(t.nbytes));
                }
            } else if (cmd.type == CMD_HAS_META) {
                std::string key = read_str(client_sock, cmd.arg_len);
                uint8_t has = src->has_meta(key) ? 1 : 0;
                write_all(client_sock, &has, sizeof(has));
            } else if (cmd.type == CMD_GET_META) {
                std::string key = read_str(client_sock, cmd.arg_len);
                const MetaValue* m = src->meta(key);
                uint8_t has = m ? 1 : 0;
                write_all(client_sock, &has, sizeof(has));
                if (m) {
                    write_all(client_sock, &m->kind, sizeof(m->kind));
                    write_all(client_sock, &m->i, sizeof(m->i));
                    write_all(client_sock, &m->f, sizeof(m->f));
                    write_str(client_sock, m->s);
                    
                    uint32_t ia_len = m->ia.size();
                    write_all(client_sock, &ia_len, sizeof(ia_len));
                    if (ia_len > 0) write_all(client_sock, m->ia.data(), ia_len * sizeof(int64_t));
                    
                    uint32_t fa_len = m->fa.size();
                    write_all(client_sock, &fa_len, sizeof(fa_len));
                    if (fa_len > 0) write_all(client_sock, m->fa.data(), fa_len * sizeof(double));
                    
                    uint32_t sa_len = m->sa.size();
                    write_all(client_sock, &sa_len, sizeof(sa_len));
                    for (const auto& s : m->sa) {
                        write_str(client_sock, s);
                    }
                }
            } else if (cmd.type == CMD_READ_RAW) {
                std::vector<char> buf(cmd.size);
                src->read_raw_at(cmd.offset, buf.data(), cmd.size);
                write_all(client_sock, buf.data(), cmd.size);
            } else {
                throw std::runtime_error("Unknown command type");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Client error: " << e.what() << "\n";
    }
    close(client_sock);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <port>\n";
        return 1;
    }
    std::string path = argv[1];
    int port = std::stoi(argv[2]);

    auto src = llm::open_model(path, false);
    if (!src) {
        std::cerr << "Failed to open model " << path << "\n";
        return 1;
    }

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        std::cerr << "Failed to create socket\n";
        return 1;
    }

    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Failed to bind to port " << port << "\n";
        return 1;
    }

    if (listen(server_sock, 10) < 0) {
        std::cerr << "Listen failed\n";
        return 1;
    }

    std::cout << "Serving " << path << " on port " << port << "...\n";

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &client_len);
        if (client_sock < 0) {
            std::cerr << "Accept failed\n";
            continue;
        }
        std::cout << "Client connected\n";
        std::thread t(handle_client, client_sock, src.get());
        t.detach();
    }

    close(server_sock);
    return 0;
}
