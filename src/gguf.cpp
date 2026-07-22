// gguf.cpp — GGUF v2/v3 parser (see gguf.h).
#include "llm/gguf.h"
#include "llm/common.h"

#include <vector>

namespace llm {

namespace {

// Forward-only buffered reader over a FileBacking. GGUF headers are parsed
// strictly in order, so a refilling window beats millions of tiny preads for
// the big tokenizer arrays while never loading the whole file.
class BufReader {
public:
    explicit BufReader(const FileBacking& fb) : fb_(fb), file_size_(fb.size()) {
        buf_.resize(1u << 20);   // 1 MB window
    }
    uint64_t tell() const { return win_start_ + cursor_; }

    void read(void* dst, uint64_t n) {
        uint8_t* out = static_cast<uint8_t*>(dst);
        while (n > 0) {
            if (cursor_ == filled_) refill();
            uint64_t avail = filled_ - cursor_;
            uint64_t take = n < avail ? n : avail;
            std::memcpy(out, buf_.data() + cursor_, take);
            out += take; cursor_ += take; n -= take;
        }
    }
    template <class T> T pod() { T v; read(&v, sizeof(T)); return v; }

    std::string gstr() {
        uint64_t len = pod<uint64_t>();
        LLM_CHECK(len <= file_size_, "gguf: absurd string length");
        std::string s;
        s.resize(len);
        if (len) read(&s[0], len);
        return s;
    }

private:
    void refill() {
        win_start_ += filled_;
        LLM_CHECK(win_start_ < file_size_, "gguf: unexpected EOF while parsing header");
        uint64_t want = std::min<uint64_t>(buf_.size(), file_size_ - win_start_);
        fb_.pread_exact(win_start_, buf_.data(), want);
        filled_ = want;
        cursor_ = 0;
    }
    const FileBacking& fb_;
    uint64_t file_size_;
    std::vector<uint8_t> buf_;
    uint64_t win_start_ = 0, filled_ = 0, cursor_ = 0;
};

// Read one metadata value of the given type into MetaValue. Arrays collapse to
// the matching typed vector (ia/fa/sa).
MetaValue read_value(BufReader& r, GgufType type);

int64_t read_scalar_int(BufReader& r, GgufType t) {
    switch (t) {
        case GgufType::UINT8:  return r.pod<uint8_t>();
        case GgufType::INT8:   return r.pod<int8_t>();
        case GgufType::UINT16: return r.pod<uint16_t>();
        case GgufType::INT16:  return r.pod<int16_t>();
        case GgufType::UINT32: return r.pod<uint32_t>();
        case GgufType::INT32:  return r.pod<int32_t>();
        case GgufType::BOOL:   return r.pod<uint8_t>() ? 1 : 0;
        case GgufType::UINT64: return (int64_t)r.pod<uint64_t>();
        case GgufType::INT64:  return r.pod<int64_t>();
        default: throw Error("gguf: not an int type");
    }
}
double read_scalar_float(BufReader& r, GgufType t) {
    if (t == GgufType::FLOAT32) return r.pod<float>();
    if (t == GgufType::FLOAT64) return r.pod<double>();
    throw Error("gguf: not a float type");
}
bool is_int_type(GgufType t) {
    switch (t) {
        case GgufType::UINT8: case GgufType::INT8: case GgufType::UINT16:
        case GgufType::INT16: case GgufType::UINT32: case GgufType::INT32:
        case GgufType::BOOL:  case GgufType::UINT64: case GgufType::INT64:
            return true;
        default: return false;
    }
}

MetaValue read_value(BufReader& r, GgufType type) {
    MetaValue mv;
    if (is_int_type(type)) { mv.kind = MetaValue::Kind::Int; mv.i = read_scalar_int(r, type); }
    else if (type == GgufType::FLOAT32 || type == GgufType::FLOAT64) {
        mv.kind = MetaValue::Kind::Float; mv.f = read_scalar_float(r, type);
    } else if (type == GgufType::STRING) {
        mv.kind = MetaValue::Kind::Str; mv.s = r.gstr();
    } else if (type == GgufType::ARRAY) {
        GgufType elem = (GgufType)r.pod<uint32_t>();
        uint64_t count = r.pod<uint64_t>();
        if (elem == GgufType::STRING) {
            mv.kind = MetaValue::Kind::StrArr; mv.sa.reserve(count);
            for (uint64_t i = 0; i < count; ++i) mv.sa.push_back(r.gstr());
        } else if (elem == GgufType::FLOAT32 || elem == GgufType::FLOAT64) {
            mv.kind = MetaValue::Kind::FloatArr; mv.fa.reserve(count);
            for (uint64_t i = 0; i < count; ++i) mv.fa.push_back(read_scalar_float(r, elem));
        } else if (is_int_type(elem)) {
            mv.kind = MetaValue::Kind::IntArr; mv.ia.reserve(count);
            for (uint64_t i = 0; i < count; ++i) mv.ia.push_back(read_scalar_int(r, elem));
        } else {
            throw Error("gguf: nested arrays unsupported");
        }
    } else {
        throw Error("gguf: unknown value type " + std::to_string((uint32_t)type));
    }
    return mv;
}

} // namespace

GgufFile::GgufFile(const std::string& path, bool use_mmap) {
    file_ = std::make_unique<FileBacking>(path, use_mmap);
    BufReader r(*file_);

    uint32_t magic = r.pod<uint32_t>();
    LLM_CHECK(magic == kGGUFMagic, "not a GGUF file (bad magic)");
    version_ = r.pod<uint32_t>();
    LLM_CHECK(version_ == 2 || version_ == 3,
              "unsupported GGUF version " + std::to_string(version_));
    uint64_t n_tensors = r.pod<uint64_t>();
    uint64_t n_meta = r.pod<uint64_t>();

    for (uint64_t i = 0; i < n_meta; ++i) {
        std::string key = r.gstr();
        GgufType type = (GgufType)r.pod<uint32_t>();
        meta_.emplace(std::move(key), read_value(r, type));
    }
    if (auto* a = meta("general.alignment"); a && a->kind == MetaValue::Kind::Int)
        alignment_ = (uint32_t)a->i;

    // Tensor directory. dims are stored fastest-first (ne[0]=n_in); we reverse
    // so shape[0] is the outer/output dim to match our row-major [n_out, n_in].
    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo ti;
        ti.name = r.gstr();
        uint32_t ndim = r.pod<uint32_t>();
        std::vector<int64_t> dims(ndim);
        for (uint32_t d = 0; d < ndim; ++d) dims[d] = (int64_t)r.pod<uint64_t>();
        ti.shape.assign(dims.rbegin(), dims.rend());
        ti.dtype = (DType)r.pod<uint32_t>();
        ti.offset = r.pod<uint64_t>();   // relative to data section
        ti.nbytes = (uint64_t)type_nbytes(ti.dtype, ti.numel());
        index_[ti.name] = (int)tensors_.size();
        tensors_.push_back(std::move(ti));
    }

    // Data section begins at the aligned end of the directory.
    data_offset_ = round_up(r.tell(), alignment_);
    for (auto& t : tensors_) {
        t.offset += data_offset_;    // make offsets absolute
        LLM_CHECK(t.offset + t.nbytes <= file_->size(),
                  "gguf: tensor '" + t.name + "' extends past EOF");
    }
}

const TensorInfo* GgufFile::find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

void GgufFile::read_raw(const TensorInfo& t, void* dst) const {
    read_raw_at(t.offset, dst, t.nbytes);
}
void GgufFile::read_raw_at(uint64_t offset, void* dst, uint64_t n) const {
    if (const uint8_t* base = file_->map_base())
        std::memcpy(dst, base + offset, n);
    else
        file_->pread_exact(offset, dst, n);
}

} // namespace llm
