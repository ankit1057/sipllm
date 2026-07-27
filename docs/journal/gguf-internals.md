# GGUF internals

**GGUF is the container llama.cpp popularized: a flat, self-describing file with a header, a key-value metadata table, a tensor directory, and one big aligned blob of tensor bytes. SipLLM parses only the first three — magic, metadata, and directory — then leaves the blob on disk and streams each tensor by positional `pread` when a layer needs it. This is the tour of `src/gguf.cpp` and `include/llm/gguf.h`: the byte layout, how the metadata KV and tensor directory are decoded, the 32-byte alignment rule, and why parsing never loads the file whole.**

## The byte layout

Everything is little-endian, and the header is fixed:

```text
magic     "GGUF"  = 0x46554747   (u32)
version                          (u32, must be 2 or 3)
tensor_count                     (u64)
metadata_kv_count                (u64)
metadata_kv_count × { gguf_string key ; u32 value_type ; value }
tensor_count      × { gguf_string name ; u32 n_dims ; u64 dims[] ; u32 type ; u64 offset }
padding to general.alignment (default 32)
tensor data blob   — each tensor at data_start + its relative offset
```

A `gguf_string` is a `u64` length followed by that many raw bytes. `GgufFile`'s constructor checks the magic against `kGGUFMagic = 0x46554747` and rejects any version that is not 2 or 3 — those are the only two on-disk revisions the parser guarantees.

## Reading the header without reading the file

GGUF headers are parsed strictly front-to-back, and a real model's tokenizer arrays can hold tens of thousands of strings. Doing that with millions of tiny `pread`s would be miserable, and slurping the whole file would defeat the entire streaming premise. So `gguf.cpp` uses a `BufReader`: a **1 MB refilling window** over the `FileBacking`. It hands out `pod<T>()` and `gstr()` reads from the buffer, refilling by `pread_exact` only when the cursor hits the end. The header parse touches at most a handful of megabytes regardless of model size.

## Metadata: scalar and array KV

Each metadata entry is a key string, a `u32` type code (`GgufType`), and a value. `read_value` decodes it into a tagged `MetaValue`:

- **Scalars** — `UINT8..INT64`, `BOOL`, `FLOAT32/64` collapse to `MetaValue::Kind::Int` or `Float`; `STRING` to `Str`.
- **Arrays** (`GgufType::ARRAY`) carry an element type and a `u64` count, then collapse to the matching typed vector — `ia` (ints), `fa` (floats), or `sa` (strings). Nested arrays are rejected.

That is how every hyperparameter and the entire tokenizer vocabulary arrive: `n_layers`, RoPE base, the vocab and merges arrays, and the architecture string are all just typed KV entries the [tokenizer](tokenizer.html) and runtime read back with `meta_int` / `meta_float` / `meta_str`. One special key, `general.alignment`, is pulled out early because it changes where the data blob starts.

## The tensor directory

For each tensor the directory records a name, a dimension count, the dims, a `u32` dtype code, and a `u64` offset *relative to the data section*. Two subtleties in the loop:

- **Dims are stored fastest-first** (`ne[0]` is the inner/input dim). SipLLM reverses them — `ti.shape.assign(dims.rbegin(), dims.rend())` — so `shape[0]` is the outer/output dim, matching the engine's row-major `[n_out, n_in]` convention.
- **The dtype code is stored verbatim.** `DType`'s numeric values deliberately mirror ggml's `enum ggml_type`, so the parser casts the on-disk `u32` straight into `DType` and the quant dispatcher switches on the same enum. `nbytes` is then computed from the type's block geometry via `type_nbytes` — never assumed.

## Alignment and the data region

The data blob does not start right after the directory; it starts at the next `general.alignment` boundary (default **32**):

```text
data_offset_ = round_up(r.tell(), alignment_)
for each tensor:  t.offset += data_offset_        // relative -> absolute
                  check t.offset + t.nbytes <= file size
```

Every tensor offset is rebased from data-section-relative to an absolute file offset, and each is bounds-checked against the file size so a truncated or malformed file fails loudly at parse time rather than mid-inference.

## Streaming a tensor: positional `pread`

After the constructor returns, `GgufFile` holds only the directory and metadata — **not one byte of weight data**. When a layer needs a weight, the loader calls `read_raw` / `read_raw_at`:

```text
read_raw_at(offset, dst, n):
    if mmap base present:  memcpy(dst, base + offset, n)   // OS page cache buffers
    else:                  file_->pread_exact(offset, dst, n)
```

It is a positional read — thread-safe, targeted at one tensor's byte range, and it never materializes the whole file. Both `GgufFile` and the toy `.llmw` `ModelFile` implement the same [`WeightSource`](gguf-parser.html) seam (directory + typed metadata + positional reads), which is why the transformer never learns whether its weights came from a real GGUF or a test fixture.

> [!NOTE] mmap is not zero-copy here
> Even with the `mmap` backend, `read_raw_at` `memcpy`s from mapped pages into a resident buffer — the win is OS page-cache buffering, not a zero-copy view. The `pread` path is the default.

That directory-plus-positional-reads design is the foundation of the whole runtime: parse the map once, then [stream layers](streaming-loader.html) off disk one block at a time. For the full parser reference see the [GGUF parser](gguf-parser.html) chapter; for how the decoded vocabulary becomes a working tokenizer, see [Tokenizer](tokenizer.html).
