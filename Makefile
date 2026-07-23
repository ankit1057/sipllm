# Makefile — miniature streaming LLM inference engine.
#
# Targets:
#   make            build the engine binary + tools
#   make test       build & run all unit tests
#   make bench      build the CLI benchmark
#   make server     build the web-GUI HTTP server
#   make clean
#
# ARM64 tuned; falls back gracefully on other hosts. No external libraries.

CXX      ?= clang++
CXXSTD   := -std=c++17
WARN     := -Wall -Wextra -Wno-unused-parameter
OPT      := -O3 -funroll-loops -fno-math-errno
INCLUDE  := -Iinclude -I.
LDLIBS   := -lpthread

# Per-arch tuning. By default -march=native lights up the host's SIMD:
#   aarch64 -> asimddp/i8mm/sve2/bf16 (the NEON kernels in simd.h/neon.cpp)
#   x86_64  -> AVX2+FMA when present, which enables the AVX2 path in simd.h
#              (falls back to scalar on pre-AVX2 hosts, so it always builds).
# Portable/release builds override this to avoid baking in the builder's exact
# ISA (which could SIGILL on older download targets), e.g.
#   make ARCHFLAGS="-mavx2 -mfma"   (AVX2 baseline)   or   make ARCHFLAGS=""
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
  CXXFLAGS_ARCH := -march=native
else ifeq ($(ARCH),x86_64)
  CXXFLAGS_ARCH := -march=native
else
  CXXFLAGS_ARCH :=
endif
ARCHFLAGS ?= $(CXXFLAGS_ARCH)

# ---- release hardening ----------------------------------------------------
# Force deterministic zero-initialization of every automatic (stack) variable.
# This is a class-elimination fix for an intermittent uninitialized-stack read
# that only surfaced under -O3 with reused (non-zeroed) stack pages and turned a
# garbage local into a near-null memmove destination (SIGSEGV). Cost is
# near-zero (a few extra stores); it makes any such latent read return 0 (the
# benign value fresh/demand-zeroed pages already had) rather than heap garbage.
# Kept SEPARATE from OPT so it survives `make OPT="-O3"` overrides (the exact
# invocation the bug was reproduced under). The Linux sanitizer/valgrind CI
# gates build with HARDEN= (empty) so they can still DETECT uninitialized reads.
HARDEN ?= -ftrivial-auto-var-init=zero

CXXFLAGS := $(CXXSTD) $(WARN) $(OPT) $(ARCHFLAGS) $(HARDEN) $(INCLUDE)

BUILD := build
SRC   := $(wildcard src/*.cpp)
OBJ   := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))

# ---- optional Vulkan backend ---------------------------------------------
# `make VULKAN=1` links libvulkan and compiles the backend (device enumeration
# + info + CPU-fallback matmul). If a GLSL->SPIR-V compiler is also present, the
# compute shader is built and embedded so matmul can offload to the GPU.
GLSLC := $(shell command -v glslc 2>/dev/null)
ifdef VULKAN
  CXXFLAGS += -DLLM_HAVE_VULKAN=1
  LDLIBS   += -lvulkan
endif

# Generate shaders/matmul_spv.h from the GLSL source when glslc is available.
.PHONY: shader
shader:
	@if [ -n "$(GLSLC)" ]; then \
	  echo "compiling shader with $(GLSLC)"; \
	  $(GLSLC) -O -o $(BUILD)/matmul.spv shaders/matmul.comp && \
	  ( echo "// auto-generated from shaders/matmul.comp"; \
	    echo "static const unsigned char matmul_spv[] = {"; \
	    od -An -v -tu1 $(BUILD)/matmul.spv | sed 's/[0-9]\{1,\}/&,/g'; \
	    echo "};"; \
	    echo "static const unsigned int matmul_spv_len = sizeof(matmul_spv);" ) \
	    > shaders/matmul_spv.h && echo "wrote shaders/matmul_spv.h"; \
	else echo "glslc not found; GPU offload will use CPU fallback"; fi

.PHONY: all test bench server clean tools

all: $(BUILD)/llm tools

# ---- core objects ---------------------------------------------------------
$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---- main CLI --------------------------------------------------------------
$(BUILD)/llm: $(OBJ) $(BUILD)/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/main.o: main.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---- tools -----------------------------------------------------------------
TOOL_SRC := $(wildcard tools/*.cpp)
TOOL_BIN := $(patsubst tools/%.cpp,$(BUILD)/%,$(TOOL_SRC))

tools: $(TOOL_BIN)

$(BUILD)/%: tools/%.cpp $(OBJ)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(OBJ) -o $@ $(LDLIBS)

# ---- web server ------------------------------------------------------------
server: $(BUILD)/llm_server

$(BUILD)/llm_server: server/server.cpp $(OBJ)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(OBJ) -o $@ $(LDLIBS)

# ---- tests -----------------------------------------------------------------
TEST_SRC := $(wildcard tests/*.cpp)
TEST_BIN := $(patsubst tests/%.cpp,$(BUILD)/%,$(TEST_SRC))

test: $(TEST_BIN)
	@echo; echo "Running unit tests:"; \
	fail=0; \
	for t in $(TEST_BIN); do echo "--- $$t"; $$t || fail=1; done; \
	echo; if [ $$fail -eq 0 ]; then echo "ALL TEST BINARIES PASSED"; else echo "SOME TESTS FAILED"; fi; \
	exit $$fail

$(BUILD)/%: tests/%.cpp $(OBJ)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< $(OBJ) -o $@ $(LDLIBS)

clean:
	rm -rf $(BUILD)
