# GGUF parser

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** The parser is the concrete implementation of the [`WeightSource`](streaming-loader.html) seam for unmodified Hugging Face GGUF files. It reads the header, metadata, and tensor directory eagerly, but never the tensor data — that streams later through `read_raw`.

## Problem

To run stock models we must read GGUF, the container llama.cpp writes, without a dependency on ggml. Two hard constraints shape the parser. First, the tensor *data* — the entire point of streaming — must never be loaded during parsing; only the tiny header and directory. Second, the parser must present its findings through the same `WeightSource` interface the toy `.llmw` format uses, so the transformer cannot tell them apart. A third wrinkle: GGUF stores tensor dimensions fastest-first (`ne[0] = n_in`), the opposite of the row-major `[n_out, n_in]` layout the rest of the engine assumes.

## Design

`GgufFile : WeightSource` (`include/llm/gguf.h`, `src/gguf.cpp`) parses the header strictly in order through a small forward-only `BufReader` — a refilling window over the `FileBacking` that beats millions of tiny `pread`s for the big tokenizer arrays while still never mapping the whole file.

The header is `magic "GGUF" | u32 version | u64 tensor_count | u64 metadata_kv_count`. The magic is checked against `kGGUFMagic` (`0x46554747`), and the version against **2 or 3** — anything else throws. Then:

1. **Metadata.** `metadata_kv_count` entries of `gguf_string key · u32 value_type · value`. `read_value` maps each `GgufType` (13 codes: `UINT8`..`FLOAT64`, plus `STRING` and `ARRAY`) into a tagged `MetaValue`. Scalars collapse to `i`/`f`/`s`; an `ARRAY` collapses to the matching typed vector (`ia`/`fa`/`sa`). `general.alignment`, if present, overrides the default alignment of **32**.
2. **Tensor directory.** `tensor_count` entries of `gguf_string name · u32 n_dims · u64 dims[] · u32 type · u64 offset`. The dims are read fastest-first and then **reversed** (`shape.assign(dims.rbegin(), dims.rend())`) so `shape[0]` is the outer/output dim — the row-major `[n_out, n_in]` the loader and `matmul` expect. `nbytes` is computed from the dtype and element count (`type_nbytes`), so the on-disk quantized size is known without touching the data.
3. **Absolute offsets.** The stored `offset` is relative to the data section. The data section begins at `round_up(directory_end, alignment)`; every tensor's `offset` is then made absolute by adding that base, and bounds-checked against the file size so a truncated file fails at open, not mid-inference.

Reads go through two positional entry points that satisfy the seam:

```text
read_raw(t, dst)         -> read_raw_at(t.offset, dst, t.nbytes)
read_raw_at(off, dst, n) -> if mmap:  memcpy(dst, map_base + off, n)
                            else:      file->pread_exact(off, dst, n)
```

Both are thread-safe positional reads. Note the mmap path **memcpys from mapped pages into the caller's resident buffer** — it is not zero-copy; the win is that the OS page cache buffers the file. (`FileBacking::prefetch()` for `fadvise`/`madvise` exists but is unused.)

### The `.llmw` sibling and content sniffing

There is a second, parallel `WeightSource`: `ModelFile` for the hand-rolled `.llmw` format (`include/llm/format.h`, `src/format.cpp`), magic `"LLMW"` = `kLLMWMagic` (`0x574D4C4C`). Its layout differs — 32-bit counts, a simpler three-type metadata section (`i64`/`f64`/`str`), directory offsets that are already absolute, and 64-byte tensor alignment — but it exposes the exact same `tensors()`/`find()`/`read_raw`/`meta` interface. It exists for toy models and the golden-oracle unit tests (`tools/make_toy_model`, `ModelWriter`).

Callers never pick a parser by extension. `open_model()` (`src/runtime.cpp`) sniffs the first four bytes and dispatches on magic:

```text
open_model(path):
    pread 4 bytes at offset 0 -> magic
    magic == 0x46554747 ('GGUF') -> GgufFile
    magic == 0x574D4C4C ('LLMW') -> ModelFile
    else                         -> throw "unrecognized file magic"
```

Downstream — `ModelConfig::from_source`, `Tokenizer::from_source`, `LayerLoader` — only ever sees a `WeightSource&`, so the same code path serves stock GGUF and toy fixtures.

## Alternatives considered

| Approach | Why not |
|:---------|:--------|
| Link ggml / llama.cpp to parse GGUF | Adds the exact dependency SipLLM exists to avoid; the engine ships with zero runtime deps. |
| `mmap` the file and point tensors into it (true zero-copy) | Gives up the bounded-memory guarantee — the OS decides residency and can fault the whole model in. mmap is offered as a *backend*, but `read_raw_at` still copies into a budgeted resident buffer. |
| Load tensor data during parse | Defeats streaming entirely; parsing must stay O(header + directory). |
| Keep GGUF's fastest-first dim order | Would force every matmul and role mapping to transpose mentally; reversing once at parse time keeps the math row-major. |

## Tradeoffs

The parser is deliberately strict: only GGUF **v2/v3**, alignment honored but assumed sane, offsets bounds-checked eagerly. It reads metadata fully (including multi-megabyte tokenizer arrays) into RAM — acceptable because metadata is tiny relative to weights and is needed anyway for config and tokenizer. Tensor *data* stays on disk until the loader asks for a specific tensor or byte range, preserving the streaming contract end to end.

## Source files

| File | Role |
|:-----|:-----|
| `include/llm/gguf.h` | `GgufFile : WeightSource` declaration, `GgufType` codes, `kGGUFMagic` |
| `src/gguf.cpp` | header/metadata/directory parse, dim reversal, absolute-offset fixup, `read_raw`/`read_raw_at` |
| `include/llm/weight_source.h` | the `WeightSource` interface, `TensorInfo`, `MetaValue`, typed `meta_int/float/str` accessors |
| `include/llm/format.h` / `src/format.cpp` | `ModelFile` (`.llmw`) sibling reader + `ModelWriter` |
| `include/llm/file_backing.h` | POSIX `pread_exact` / optional `mmap` shared by both sources |
| `src/runtime.cpp` | `open_model()` — 4-byte magic sniff dispatching to `GgufFile` or `ModelFile` |

## Future work

- **`posix_fadvise`/`madvise(WILLNEED)` readahead** — `FileBacking::prefetch()` is implemented but not wired into the parser or loader.
- **Streaming metadata for pathological tokenizer arrays** — today the arrays are read fully into RAM; a lazy accessor would trim the parse-time footprint further.
- **GGUF v1 / newer versions** — the parser rejects anything outside v2/v3 by design; broadening it is a compatibility, not a correctness, question.
