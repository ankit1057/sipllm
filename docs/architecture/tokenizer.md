# Tokenizer

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** One `Tokenizer` class covers three vocab kinds, all built from the same GGUF `tokenizer.ggml.*` metadata through the [`WeightSource`](gguf-parser.html) seam. It turns text into token ids and back — nothing more. Conversation formatting is explicitly *not* its job.

## Problem

Stock models ship their vocabulary inside the GGUF, but in three incompatible schemes: Llama-1/2 use SentencePiece (score-driven merges, `▁` for space, `<0xNN>` byte fallback); Llama-3 and GPT-2 use byte-level BPE (a byte→unicode remap then rank-ordered pair merges); and toy models have no vocab at all. The engine needs one interface that reads whichever scheme a file declares and produces the *same* ids llama.cpp would — because a single mis-tokenized prompt invalidates the golden logit comparison.

## Design

`Tokenizer::from_source(const WeightSource&)` (`include/llm/tokenizer.h`, `src/tokenizer.cpp`) builds the right variant from metadata. If `tokenizer.ggml.tokens` is absent or empty it returns `byte_tokenizer()` — an identity byte-level map (256 symbols). Otherwise it loads the vocab, then reads `tokenizer.ggml.scores` (SPM merge scores) and `tokenizer.ggml.token_type` if present, and picks a `Kind`:

- **`SentencePiece`** — the default for a scored `"llama"`/SPM vocab.
- **`BPE`** — chosen only when `tokenizer.ggml.model == "gpt2"`, or when the model string is empty but a non-empty `merges` list exists *and there are no scores*.
- **`Byte`** — the no-vocab fallback.

> [!KEY] The declared model wins
> A `"llama"` GGUF is SentencePiece even when it also ships a merges list (used there only for byte-fallback). Treating it as merge-BPE would mis-tokenize. Only `"gpt2"` (the Llama-3 family) is routed to BPE. This is a deliberate precedence rule in `from_source`, not a guess.

BOS/EOS come from `tokenizer.ggml.bos_token_id`/`eos_token_id`; end-of-generation is `eos` plus `<|eot_id|>` when the vocab contains it (Llama-3 ends turns on it). `is_eog(id)` checks that set.

### encode

`encode(text, add_bos)` prepends `bos_` only when `add_bos` and `bos_ >= 0`, then dispatches on kind:

**`encode_spm`** — SentencePiece greedy merge. It prepends a dummy `▁` prefix and maps every space to `▁` (`U+2581`), splits into UTF-8 symbols, then repeatedly merges the adjacent pair whose concatenation has the **highest vocab score** until no pair improves. Leftover symbols with no id fall back to `<0xNN>` byte tokens — one per raw byte — so any input is representable.

**`encode_bpe`** — byte-level BPE. A pragmatic pre-tokenizer splits text into ASCII-class chunks (letters / ≤3-digit runs / whitespace / punctuation, with GPT-2-style contractions and leading-space attachment), each chunk's bytes are remapped through the GPT-2 **byte↔unicode table** (`build_byte_unicode`: printable bytes map to themselves, the rest to codepoints ≥256 so every byte is a visible char), and pairs are merged by **lowest rank** from the `merges` list until none remain.

**`Byte`** — each byte becomes its own id, exactly.

### decode

`decode_token(id)` is the inverse and is safe to call **incrementally on one new id** during streaming generation. For SPM it turns `<0xNN>` back into the raw byte and `▁` back into a space; for BPE it reverses the byte→unicode table to recover raw bytes; for Byte it is the identity. `decode(ids)` just concatenates.

## The engine applies no chat template

> [!WARNING] Raw-text inference — templating lives in the app
> The engine tokenizes and runs **raw text**. It does not wrap a prompt in any chat/instruction template, and BOS is added only at the start of a fresh sequence — not before every turn. Conversation structure (system/user/assistant turns, special tokens) is the caller's responsibility. In the Flutter SDK a separate `PromptTemplate` (chatml / llama3 / zephyr / raw) formats the conversation *before* it reaches `encode`. Do not expect the C++ engine to add role markers for you.

## Alternatives considered

| Approach | Why not |
|:---------|:--------|
| Ship SentencePiece / tokenizers as a dependency | Reintroduces the exact third-party weight the project avoids; the tokenizer is a few hundred lines instead. |
| Detect scheme purely from the merges list | Mis-classifies SPM Llama files that carry merges for byte fallback — hence the "declared model wins" precedence. |
| Bake a chat template into `encode` | Couples the engine to one chat convention; keeping it raw lets the app choose per-model formatting. |
| Separate tokenizer classes per kind | Three kinds share vocab loading, BOS/EOG handling, and decode plumbing; one class with a `Kind` tag is less duplication. |

## Tradeoffs

The pre-tokenizer is "pragmatic" — ASCII-class heuristics rather than a full regex spec — which is enough to reproduce llama.cpp ids on the validation prompts but is not a byte-for-byte port of every edge case. Correctness is anchored by the golden comparison: the tokenizer only has to agree with llama.cpp closely enough that the final argmax matches (it does, across F16/Q8_0/Q5_K_M/Q4_K_M — see [Benchmarks](benchmarks.html)). SPM's greedy score merge is O(symbols²) per step but inputs are short prompts, so it is not a hot path next to the matmuls.

## Source files

| File | Role |
|:-----|:-----|
| `include/llm/tokenizer.h` | `Tokenizer` class, `Kind` enum, `from_source`/`byte_tokenizer`, encode/decode API |
| `src/tokenizer.cpp` | `from_source` scheme selection, `encode_spm`, `encode_bpe`, `pretokenize`, byte↔unicode table, `decode_token` |
| `include/llm/weight_source.h` | `tokenizer.ggml.*` metadata read via `meta_str`/`meta_int` and typed `MetaValue` arrays |
| `src/runtime.cpp` | builds the tokenizer once from the model's `WeightSource` in the generate loop |

## Future work

- **A stricter BPE pre-tokenizer** matching each family's exact regex, to close remaining divergences on adversarial inputs.
- **Faster SPM merge** (priority-queue instead of full rescan) — only worth it for very long prompts.
- **Added-token / special-token tables** surfaced to the API so apps can address them without hardcoding strings like `<|eot_id|>`.
