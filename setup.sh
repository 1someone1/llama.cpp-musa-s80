#!/usr/bin/env bash
# One-shot build for llama.cpp (b12600) + S80 patches on MUSA 4.0.1.
set -euo pipefail

PINNED_COMMIT="29b52d398b294509746ec0a3f28948a3107c0213"
UI_VERSION="b10726"

echo "==> Checking MUSA SDK"
if [ ! -d /usr/local/musa ]; then
    echo "ERROR: /usr/local/musa not found. Install MUSA SDK first." >&2
    exit 1
fi

echo "==> Verifying this is the patched source snapshot"
for f in fattn-musa.cu mul-mat-q4-mma.cu musa-fuse.cu; do
    if [ ! -f "ggml/src/ggml-cuda/$f" ]; then
        echo "ERROR: S80 patch file ggml/src/ggml-cuda/$f not found." >&2
        exit 1
    fi
done
echo "    (upstream snapshot: $PINNED_COMMIT)"

echo "==> Fetching WebUI dist ($UI_VERSION, matches pinned commit)"
BASE="https://hf-mirror.com/buckets/ggml-org/llama-ui/resolve/$UI_VERSION"
if [ ! -f ui/apps/chat/dist/index.html ]; then
    wget -q -O /tmp/llama-ui-dist.tar.gz "$BASE/dist.tar.gz?download=true"
    wget -q -O /tmp/llama-ui-dist.tar.gz.sha256 "$BASE/dist.tar.gz.sha256?download=true"
    expected=$(grep -oE '^[0-9a-fA-F]+' /tmp/llama-ui-dist.tar.gz.sha256 | head -1)
    actual=$(sha256sum /tmp/llama-ui-dist.tar.gz | awk '{print $1}')
    if [ "$expected" != "$actual" ]; then
        echo "ERROR: WebUI dist checksum mismatch" >&2
        exit 1
    fi
    mkdir -p ui/apps/chat/dist
    tar -xzf /tmp/llama-ui-dist.tar.gz -C ui/apps/chat/dist
fi

echo "==> Configuring cmake"
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_MUSA=ON -DMUSA_ARCHITECTURES="21" \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=ON

echo "==> Building llama-server"
cmake --build build --target llama-server -j "$(nproc)"

echo "==> Done. Binary: build/bin/llama-server"
echo "Next: ./run.sh (see README.md for model download and settings)"
