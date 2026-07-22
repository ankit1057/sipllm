// gguf_to_f16 — transcode any GGUF into an all-F16 GGUF.
//
// The public TinyLlama GGUF repos ship every K-quant but no plain F16 file, and
// llama.cpp's `llama-quantize` pulls in a web-server dependency that will not
// build in this sandbox. So we mint the F16 baseline ourselves: read the source
// with our own GGUF parser, dequantize every 2-D weight back to fp32, re-encode
// it as IEEE half precision, and write a byte-valid GGUF v3 that BOTH our engine
// and llama.cpp load. 1-D tensors (RMSNorm weights) stay fp32, mirroring exactly
// what llama.cpp's own F16 conversion does.
//
// The file-level metadata blob (hyperparameters + the full tokenizer) is copied
// through verbatim; only the tensor directory (types + offsets) and the tensor
// data are rebuilt. This gives a genuine F16 model that exercises our F16 decode
// path on every projection weight — the one format the download matrix lacks.
//
//   gguf_to_f16 <in.gguf> <out.gguf>
#include "llm/gguf.h"
#include "llm/quant.h"
#include "llm/dtype.h"
#include "llm/file_backing.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace llm;

namespace {

// Minimal little-endian cursor over an in-memory buffer.
struct Cursor {
    const uint8_t* p;
    size_t n, off = 0;
    template <class T> T pod() {
        T v;
        LLM_CHECK(off + sizeof(T) <= n, "gguf_to_f16: metadata truncated");
        std::memcpy(&v, p + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
    void skip(size_t k) {
        LLM_CHECK(off + k <= n, "gguf_to_f16: metadata truncated");
        off += k;
    }
};

// Byte width of a fixed-size GGUF scalar value type (0 for STRING/ARRAY).
size_t scalar_width(uint32_t t) {
    switch (static_cast<GgufType>(t)) {
        case GgufType::UINT8: case GgufType::INT8: case GgufType::BOOL:   return 1;
        case GgufType::UINT16: case GgufType::INT16:                      return 2;
        case GgufType::UINT32: case GgufType::INT32: case GgufType::FLOAT32: return 4;
        case GgufType::UINT64: case GgufType::INT64: case GgufType::FLOAT64: return 8;
        default: return 0;   // STRING / ARRAY handled separately
    }
}

// Advance the cursor past one metadata value of the given type.
void skip_value(Cursor& c, uint32_t type) {
    auto gt = static_cast<GgufType>(type);
    if (gt == GgufType::STRING) {
        uint64_t len = c.pod<uint64_t>();
        c.skip(len);
    } else if (gt == GgufType::ARRAY) {
        uint32_t elem = c.pod<uint32_t>();
        uint64_t count = c.pod<uint64_t>();
        if (static_cast<GgufType>(elem) == GgufType::STRING) {
            for (uint64_t i = 0; i < count; ++i) { uint64_t l = c.pod<uint64_t>(); c.skip(l); }
        } else {
            c.skip(scalar_width(elem) * count);
        }
    } else {
        c.skip(scalar_width(type));
    }
}

uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const uint8_t* c = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), c, c + sizeof(T));
}
void put_bytes(std::vector<uint8_t>& b, const void* p, size_t n) {
    const uint8_t* c = static_cast<const uint8_t*>(p);
    b.insert(b.end(), c, c + n);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: gguf_to_f16 <in.gguf> <out.gguf>\n"); return 2; }
    const std::string in = argv[1], out = argv[2];
    try {
        GgufFile g(in);
        const uint64_t align = (uint64_t)std::max<int64_t>(1, g.meta_int("general.alignment", 32));
        const uint64_t data_off = g.data_offset();

        // The region [0, data_off) holds: 24-byte header, the metadata blob, the
        // tensor directory, then alignment padding. We copy the header+metadata
        // verbatim and rebuild the directory, so first locate where the metadata
        // ends by skip-parsing exactly the KV pairs the header advertises.
        FileBacking fb(in, false);
        std::vector<uint8_t> head(data_off);
        fb.pread_exact(0, head.data(), data_off);

        Cursor c{head.data(), head.size(), 0};
        uint32_t magic = c.pod<uint32_t>();
        LLM_CHECK(magic == kGGUFMagic, "gguf_to_f16: not a GGUF file");
        c.pod<uint32_t>();                       // version
        c.pod<uint64_t>();                       // tensor count (unchanged)
        uint64_t n_meta = c.pod<uint64_t>();
        for (uint64_t i = 0; i < n_meta; ++i) {
            uint64_t klen = c.pod<uint64_t>();
            c.skip(klen);                        // key
            uint32_t vtype = c.pod<uint32_t>();
            skip_value(c, vtype);
        }
        const size_t meta_end = c.off;           // == start of tensor directory

        const std::vector<TensorInfo>& T = g.tensors();

        // Decide each tensor's output type and compute relative data offsets.
        // 2-D+ weights -> F16; 1-D tensors (norms) stay F32, as llama.cpp does.
        std::vector<DType>    otype(T.size());
        std::vector<uint64_t> onbytes(T.size()), rel(T.size());
        uint64_t cur = 0;
        for (size_t i = 0; i < T.size(); ++i) {
            otype[i]   = (T[i].shape.size() >= 2) ? DType::F16 : DType::F32;
            onbytes[i] = type_nbytes(otype[i], T[i].numel());
            rel[i]     = cur;
            cur        = align_up(cur + onbytes[i], align);
        }

        // Build the new tensor directory (same names/dims/order, new type+offset).
        std::vector<uint8_t> dir;
        for (size_t i = 0; i < T.size(); ++i) {
            const TensorInfo& t = T[i];
            put<uint64_t>(dir, t.name.size());
            put_bytes(dir, t.name.data(), t.name.size());
            put<uint32_t>(dir, (uint32_t)t.shape.size());
            for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it)   // ne-order
                put<uint64_t>(dir, (uint64_t)*it);
            put<uint32_t>(dir, (uint32_t)otype[i]);
            put<uint64_t>(dir, rel[i]);
        }

        const uint64_t data_start = align_up(meta_end + dir.size(), align);

        FILE* f = std::fopen(out.c_str(), "wb");
        LLM_CHECK(f, "gguf_to_f16: cannot open output " + out);
        std::fwrite(head.data(), 1, meta_end, f);        // header + metadata (verbatim)
        std::fwrite(dir.data(), 1, dir.size(), f);        // rebuilt directory
        for (uint64_t pos = meta_end + dir.size(); pos < data_start; ++pos)
            std::fputc(0, f);                             // pad to data_start

        // Stream each tensor: read raw -> fp32 -> (F16 | F32) -> write, honoring
        // per-tensor alignment padding so offsets match the directory exactly.
        std::vector<uint8_t> raw;
        std::vector<float>   f32;
        std::vector<uint16_t> f16;
        uint64_t written = 0;
        for (size_t i = 0; i < T.size(); ++i) {
            const TensorInfo& t = T[i];
            const int64_t ne = t.numel();
            raw.resize(t.nbytes);
            fb.pread_exact(t.offset, raw.data(), t.nbytes);
            f32.resize(ne);
            if (t.dtype == DType::F32)      std::memcpy(f32.data(), raw.data(), ne * sizeof(float));
            else if (t.dtype == DType::F16) for (int64_t j = 0; j < ne; ++j)
                                                f32[j] = fp16_to_fp32(reinterpret_cast<uint16_t*>(raw.data())[j]);
            else                            dequantize_row(t.dtype, raw.data(), f32.data(), ne);

            while (written < rel[i]) { std::fputc(0, f); ++written; }   // inter-tensor pad
            if (otype[i] == DType::F16) {
                f16.resize(ne);
                for (int64_t j = 0; j < ne; ++j) f16[j] = fp32_to_fp16(f32[j]);
                std::fwrite(f16.data(), sizeof(uint16_t), ne, f);
            } else {
                std::fwrite(f32.data(), sizeof(float), ne, f);
            }
            written += onbytes[i];
        }
        std::fclose(f);

        size_t n_f16 = 0;
        for (size_t i = 0; i < T.size(); ++i) if (otype[i] == DType::F16) ++n_f16;
        std::printf("wrote %s: %zu tensors (%zu -> F16, %zu kept F32), align %llu\n",
                    out.c_str(), T.size(), n_f16, T.size() - n_f16,
                    (unsigned long long)align);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gguf_to_f16 error: %s\n", e.what());
        return 1;
    }
}
