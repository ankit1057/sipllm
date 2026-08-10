# Sip IR Binary Format Specification

**Sip IR (SIPR)** is SipLLM's stable, backend-agnostic model representation binary format. It acts as the seam that makes the runtime a platform. Importers (e.g. GGUF today; Hugging Face, ONNX, PyTorch tomorrow) all produce the same `SipModel` in this format, and the executor and backends consume only that. 

Nothing downstream of the Sip IR knows what original file format the model came from.

## Design Principles

- **Additive and Read-Only**: Importing a model into Sip IR does not change a single number in the forward pass. It formalizes what the engine does so that platform layers have one stable contract.
- **Streaming Ready**: The format natively supports memory-mapped I/O and stream processing (positional `pread`), just like GGUF.
- **Zero Dependencies**: Standard C++17 implementations can write and parse the format with no extra JSON or parser libraries.

## On-Disk Layout

The binary format consists of four contiguous sections:
1. Header
2. Metadata Key-Value pairs
3. Tensor Directory (Descriptors)
4. Tensor Payload Data

All integers are little-endian.

---

### 1. Header

The file begins immediately with a 32-byte header.

```c
struct SipIRHeader {
    uint32_t magic;          // Must be 0x53495052 ('SIPR' in little-endian ASCII)
    uint32_t version;        // Format version (currently 1)
    uint64_t n_tensors;      // Number of tensors
    uint64_t metadata_bytes; // Size of the metadata block following the header (bytes)
    uint64_t payload_offset; // Absolute file offset where tensor payloads begin
};
```

---

### 2. Metadata Section

Immediately following the header is the metadata section. Its size is exactly `metadata_bytes`.
It consists of a sequence of key-value pairs encoded sequentially exactly as they are in GGUF.

For each pair, the encoding is:
1. `uint64_t key_len`: Length of the key string.
2. `char key[key_len]`: The key string (no null terminator).
3. `uint32_t value_type`: The data type of the value (mapped to `GgufType` enum).
4. `value_data`: The value encoded exactly as the GGUF `value_type`.

**GgufType enum mapping (for `value_type`)**:
- UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5
- FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9
- UINT64 = 10, INT64 = 11, FLOAT64 = 12

Note: `STRING` is encoded as `uint64_t len` + `char str[len]`. `ARRAY` is encoded as `uint32_t elem_type` + `uint64_t count` + elements.

---

### 3. Tensor Directory

Immediately following the metadata section is the tensor directory, which consists of `n_tensors` records. Each record is a fixed-size `SipIRTensorDescriptor` (104 bytes), aligned to 8 bytes.

```c
struct SipIRTensorDescriptor {
    char name[64];      // 64-character fixed tensor name (null terminated).
    uint32_t dtype;     // Data type enum (from llm::DType).
    uint32_t pad;       // Padding to ensure 8-byte alignment for shape.
    int64_t shape[4];   // 4D shape (row-major). Unused dimensions are set to 1.
    uint64_t offset;    // Absolute file offset to the tensor payload.
    uint64_t nbytes;    // Size of the payload in bytes.
};
```
The fixed 64-character name array allows random access scanning of the directory without string length parsing overhead.

---

### 4. Padding and Tensor Payloads

After the tensor directory, padding may be inserted to align the first tensor payload (usually to a 32-byte boundary). 
The `payload_offset` in the `SipIRHeader` points exactly to the first tensor payload.

Tensor payloads are written sequentially. Each payload length matches its descriptor's `nbytes`. Padding bytes may be inserted between tensor payloads to maintain 32-byte alignment. Each descriptor's `offset` specifies the absolute file offset where its payload begins.
