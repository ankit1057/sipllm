<!-- Thanks for contributing to sipllm! -->

## What & why

<!-- What does this change do, and why? Link any issue: "Closes #123". -->

## How I tested

- [ ] `make test` is green
- [ ] If I touched the math (forward pass / quant / RoPE / attention / loader),
      the golden matrix still passes:
      `python3 golden/validate_matrix.py` → all formats PASS
- [ ] Added/updated tests for the change

## Checklist

- [ ] No new third-party runtime dependency
- [ ] Matches the surrounding code style
- [ ] CI passes (build + tests on gcc and clang)
