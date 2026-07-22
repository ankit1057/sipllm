// format.cpp — .llmw reader/writer implementation (see format.h).
#include "llm/format.h"

#include <cstdio>
#include <vector>

namespace llm {

// ---- little-endian byte cursor -------------------------------------------
namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    void need(size_t n) { LLM_CHECK(p + n <= end, "llmw: truncated header"); }
    template <class T> T pod() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint32_t len = pod<uint32_t>();
        need(len);
        std::string s(reinterpret_cast<const char*>(p), len);
        p += len;
        return s;
    }
};

struct Writer {
    std::vector<uint8_t> buf;
    template <class T> void pod(T v) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), b, b + sizeof(T));
    }
    void str(const std::string& s) {
        pod<uint32_t>(static_cast<uint32_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }
    void pad_to(size_t align) {
        while (buf.size() % align) buf.push_back(0);
    }
};

} // namespace

// ---- reader ---------------------------------------------------------------
ModelFile::ModelFile(const std::string& path, bool use_mmap) {
    file_ = std::make_unique<FileBacking>(path, use_mmap);

    // Read the fixed header first, then the (variable) metadata+directory.
    // We read a generous prefix so a single pread covers the whole directory
    // for small models; if not, we grow and re-read.
    uint64_t prefix = std::min<uint64_t>(file_->size(), 1u << 20);  // up to 1 MB
    std::vector<uint8_t> hdr(prefix);
    file_->pread_exact(0, hdr.data(), prefix);

    Reader r{hdr.data(), hdr.data() + hdr.size()};
    uint32_t magic = r.pod<uint32_t>();
    LLM_CHECK(magic == kLLMWMagic, "not an .llmw file (bad magic)");
    uint32_t version = r.pod<uint32_t>();
    LLM_CHECK(version == kLLMWVersion, "unsupported .llmw version");
    uint32_t n_meta = r.pod<uint32_t>();
    uint32_t n_tensor = r.pod<uint32_t>();

    for (uint32_t i = 0; i < n_meta; ++i) {
        std::string key = r.str();
        uint8_t type = r.pod<uint8_t>();
        MetaValue mv;
        if (type == 0)      { mv.kind = MetaValue::Kind::Int;   mv.i = r.pod<int64_t>(); }
        else if (type == 1) { mv.kind = MetaValue::Kind::Float; mv.f = r.pod<double>(); }
        else                { mv.kind = MetaValue::Kind::Str;   mv.s = r.str(); }
        meta_.emplace(std::move(key), std::move(mv));
    }

    tensors_.reserve(n_tensor);
    for (uint32_t i = 0; i < n_tensor; ++i) {
        TensorInfo ti;
        ti.name = r.str();
        ti.dtype = static_cast<DType>(r.pod<int32_t>());
        uint32_t ndim = r.pod<uint32_t>();
        ti.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) ti.shape[d] = r.pod<int64_t>();
        ti.offset = r.pod<uint64_t>();
        ti.nbytes = r.pod<uint64_t>();
        LLM_CHECK(ti.offset + ti.nbytes <= file_->size(),
                  "llmw: tensor '" + ti.name + "' extends past EOF");
        index_[ti.name] = static_cast<int>(tensors_.size());
        tensors_.push_back(std::move(ti));
    }
}

const TensorInfo* ModelFile::find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

void ModelFile::read_raw(const TensorInfo& t, void* dst) const {
    read_raw_at(t.offset, dst, t.nbytes);
}

void ModelFile::read_raw_at(uint64_t offset, void* dst, uint64_t n) const {
    if (const uint8_t* base = file_->map_base())
        std::memcpy(dst, base + offset, n);
    else
        file_->pread_exact(offset, dst, n);
}

// ---- writer ---------------------------------------------------------------
void ModelWriter::set_meta(const std::string& key, int64_t v) {
    MetaValue mv; mv.kind = MetaValue::Kind::Int; mv.i = v; meta_[key] = mv;
}
void ModelWriter::set_meta(const std::string& key, double v) {
    MetaValue mv; mv.kind = MetaValue::Kind::Float; mv.f = v; meta_[key] = mv;
}
void ModelWriter::set_meta(const std::string& key, const std::string& v) {
    MetaValue mv; mv.kind = MetaValue::Kind::Str; mv.s = v; meta_[key] = mv;
}

void ModelWriter::add_tensor(const std::string& name, DType dtype,
                             std::vector<int64_t> shape, const void* data) {
    Pending pd;
    pd.info.name = name;
    pd.info.dtype = dtype;
    pd.info.shape = std::move(shape);
    int64_t nbytes = type_nbytes(dtype, pd.info.numel());
    pd.info.nbytes = static_cast<uint64_t>(nbytes);
    pd.bytes.resize(nbytes);
    std::memcpy(pd.bytes.data(), data, nbytes);
    pending_.push_back(std::move(pd));
}

void ModelWriter::add_f32(const std::string& name, std::vector<int64_t> shape,
                          const float* data) {
    add_tensor(name, DType::F32, std::move(shape), data);
}

void ModelWriter::write(const std::string& path) const {
    Writer w;
    w.pod<uint32_t>(kLLMWMagic);
    w.pod<uint32_t>(kLLMWVersion);
    w.pod<uint32_t>(static_cast<uint32_t>(meta_.size()));
    w.pod<uint32_t>(static_cast<uint32_t>(pending_.size()));

    for (auto& kv : meta_) {
        w.str(kv.first);
        const MetaValue& mv = kv.second;
        if (mv.kind == MetaValue::Kind::Int)        { w.pod<uint8_t>(0); w.pod<int64_t>(mv.i); }
        else if (mv.kind == MetaValue::Kind::Float) { w.pod<uint8_t>(1); w.pod<double>(mv.f); }
        else                                        { w.pod<uint8_t>(2); w.str(mv.s); }
    }

    // First pass: emit directory with placeholder offsets so we know the size
    // of the header region, then compute each tensor's aligned data offset.
    size_t dir_start = w.buf.size();
    for (auto& pd : pending_) {
        w.str(pd.info.name);
        w.pod<int32_t>(static_cast<int32_t>(pd.info.dtype));
        w.pod<uint32_t>(static_cast<uint32_t>(pd.info.shape.size()));
        for (auto d : pd.info.shape) w.pod<int64_t>(d);
        w.pod<uint64_t>(0);   // offset placeholder
        w.pod<uint64_t>(pd.info.nbytes);
    }
    (void)dir_start;
    w.pad_to(64);

    // Compute data offsets and patch the directory. We re-walk the directory
    // records to find each offset field. Simpler: record field positions.
    // Recompute positions by re-serializing lengths.
    std::vector<uint64_t> offsets(pending_.size());
    uint64_t cur = w.buf.size();
    for (size_t i = 0; i < pending_.size(); ++i) {
        offsets[i] = cur;
        cur += round_up(pending_[i].info.nbytes, 64);
    }

    // Patch offsets in the directory: reconstruct the byte position of each
    // offset field. Walk from the directory start mirroring the writer above.
    {
        // Rebuild directory positions.
        // Header consumed dir_start bytes already. Now step through records.
        size_t pos = dir_start;
        auto skip_str = [&](const std::string& s) {
            pos += sizeof(uint32_t) + s.size();
        };
        for (size_t i = 0; i < pending_.size(); ++i) {
            const auto& info = pending_[i].info;
            skip_str(info.name);
            pos += sizeof(int32_t);                  // dtype
            pos += sizeof(uint32_t);                 // ndim
            pos += sizeof(int64_t) * info.shape.size();
            // offset field is here:
            std::memcpy(&w.buf[pos], &offsets[i], sizeof(uint64_t));
            pos += sizeof(uint64_t);                 // offset
            pos += sizeof(uint64_t);                 // nbytes
        }
    }

    // Append tensor data with 64-byte alignment.
    for (size_t i = 0; i < pending_.size(); ++i) {
        while (w.buf.size() < offsets[i]) w.buf.push_back(0);
        const auto& b = pending_[i].bytes;
        w.buf.insert(w.buf.end(), b.begin(), b.end());
    }

    FILE* fp = std::fopen(path.c_str(), "wb");
    LLM_CHECK(fp != nullptr, "cannot open for write: " + path);
    size_t wrote = std::fwrite(w.buf.data(), 1, w.buf.size(), fp);
    std::fclose(fp);
    LLM_CHECK(wrote == w.buf.size(), "short write to " + path);
}

} // namespace llm
