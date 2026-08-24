#!/bin/sh
# sipllm installer — one line to a working streaming LLM engine:
#
#   curl -fsSL https://raw.githubusercontent.com/ankit1057/sipllm/main/install.sh | sh
#
# Downloads the prebuilt release for your OS/arch if one exists; otherwise builds
# from source (needs git + make + a C++17 compiler). Installs the `sipllm` CLI
# and engine into ~/.sipllm/bin and points your PATH at it. Env overrides:
#   SIPLLM_PREFIX=/custom/dir      SIPLLM_BUILD=1 (force source build)
set -eu

REPO="ankit1057/sipllm"
VERSION="0.6.0"
PREFIX="${SIPLLM_PREFIX:-$HOME/.sipllm}"
BIN="$PREFIX/bin"
FORCE_BUILD="${SIPLLM_BUILD:-0}"

say()  { printf '\033[1;36m>>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# ---- detect platform ------------------------------------------------------
os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
case "$arch" in
    aarch64|arm64) arch="aarch64" ;;
    x86_64|amd64)  arch="x86_64"  ;;
    *) warn "unknown arch '$arch' — will build from source" ; FORCE_BUILD=1 ;;
esac
ASSET="sipllm-${VERSION}-${os}-${arch}.tar.gz"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ASSET}"

fetch() {  # fetch <url> <dest>  (curl or wget)
    if command -v curl >/dev/null 2>&1; then curl -fSL -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then wget -qO "$2" "$1"
    else die "need curl or wget"; fi
}

mkdir -p "$BIN"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

install_prebuilt() {
    say "fetching prebuilt: $ASSET"
    fetch "$URL" "$TMP/pkg.tar.gz" 2>/dev/null || return 1
    tar -xzf "$TMP/pkg.tar.gz" -C "$TMP" || return 1
    src="$(find "$TMP" -maxdepth 2 -name sipllm -type f | head -1)"
    [ -n "$src" ] || return 1
    cp -f "$(dirname "$src")"/* "$BIN"/ 2>/dev/null || true

    chmod +x "$BIN"/* 2>/dev/null || true
    if [ "$os" = "darwin" ] && command -v codesign >/dev/null 2>&1; then
        codesign --force -s - "$BIN"/* 2>/dev/null || true
    fi

    # verify the compiled binary can execute on this system (catches libc/arch mismatch)
    local ret=0
    "$BIN/llm-engine" >/dev/null 2>&1 || ret=$?
    if [ "$ret" = "126" ] || [ "$ret" = "127" ] || [ "$ret" -gt 128 ]; then
        warn "prebuilt binary is incompatible with this system (libc or architecture mismatch)"
        return 1
    fi
    return 0
}


install_deps() {
    local need_git=0 need_make=0 need_cxx=0
    command -v git >/dev/null 2>&1 || need_git=1
    command -v make >/dev/null 2>&1 || need_make=1
    if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        need_cxx=1
    fi
    if [ "$need_git" = "0" ] && [ "$need_make" = "0" ] && [ "$need_cxx" = "0" ]; then
        return 0
    fi
    say "Installing missing build dependencies (git, make, C++ compiler)..."
    local sudo_cmd=""
    [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && sudo_cmd="sudo"

    if command -v apk >/dev/null 2>&1; then
        $sudo_cmd apk add --no-cache git make g++
    elif command -v pkg >/dev/null 2>&1 && uname -o 2>/dev/null | grep -iq android; then
        pkg install -y git make clang
    elif command -v apt-get >/dev/null 2>&1; then
        $sudo_cmd apt-get update -qq || true
        $sudo_cmd apt-get install -y git make g++ build-essential
    elif command -v dnf >/dev/null 2>&1; then
        $sudo_cmd dnf install -y git make gcc-c++
    elif command -v yum >/dev/null 2>&1; then
        $sudo_cmd yum install -y git make gcc-c++
    elif command -v pacman >/dev/null 2>&1; then
        $sudo_cmd pacman -Sy --noconfirm git make gcc
    elif command -v brew >/dev/null 2>&1; then
        brew install git make gcc
    else
        warn "Could not detect package manager. Build may fail if git, make, or C++17 compiler are missing."
    fi
}

build_from_source() {
    say "building from source"
    install_deps
    command -v git  >/dev/null 2>&1 || die "git required for source build"
    command -v make >/dev/null 2>&1 || die "make required for source build"
    git clone --depth 1 "https://github.com/${REPO}.git" "$TMP/src" >/dev/null 2>&1         || die "git clone failed"
    ( cd "$TMP/src" && make -j"$(nproc 2>/dev/null || echo 2)" all && make server )         || die "build failed (need a C++17 compiler)"
    cp -f "$TMP/src/sipllm" "$BIN/sipllm"
    cp -f "$TMP/src/build/llm"        "$BIN/llm-engine"
    cp -f "$TMP/src/build/llm_server" "$BIN/llm_server" 2>/dev/null || true
    for t in dump_logits bench inspect_gguf make_toy_model gguf_to_f16; do
        cp -f "$TMP/src/build/$t" "$BIN/$t" 2>/dev/null || true
    done

    chmod +x "$BIN"/* 2>/dev/null || true
    if [ "$os" = "darwin" ] && command -v codesign >/dev/null 2>&1; then
        codesign --force -s - "$BIN"/* 2>/dev/null || true
    fi
}

if [ "$FORCE_BUILD" = "1" ] || ! install_prebuilt; then
    [ "$FORCE_BUILD" = "1" ] || warn "no prebuilt for ${os}-${arch} (or download failed)"
    build_from_source
fi

[ -x "$BIN/sipllm" ] || die "install failed: $BIN/sipllm missing"

# ---- wire up PATH ---------------------------------------------------------
linked=0
for d in /usr/local/bin /data/data/com.termux/files/usr/bin "$HOME/.local/bin" "$HOME/bin"; do
    [ -d "$d" ] || mkdir -p "$d" 2>/dev/null || true
    if [ -d "$d" ] && [ -w "$d" ]; then
        ln -sf "$BIN/sipllm" "$d/sipllm" && linked=1 && say "linked sipllm into $d" && break
    fi
done
if [ "$linked" = "0" ]; then
    case ":$PATH:" in
        *":$BIN:"*) : ;;
        *) profile="${HOME}/.profile"; [ -n "${ZSH_VERSION:-}" ] && profile="${HOME}/.zshrc"
           printf '\nexport PATH="%s:$PATH"\n' "$BIN" >> "$profile"
           say "added $BIN to PATH in $profile (open a new shell, or: export PATH=\"$BIN:\$PATH\")" ;;
    esac
fi

cat <<EOF

$(say "sipllm $VERSION installed to $BIN")

Quick start:
  sipllm run tinyllama -p "The capital of France is" -n 40
  sipllm serve tinyllama --port 8080
  sipllm registry          # list available models

The engine streams model weights off disk, so peak RAM stays flat (~200-400 MB)
regardless of model size. Models cache in $PREFIX/models.
EOF
