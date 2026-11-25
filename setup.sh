#!/usr/bin/env bash
# One-shot build for llama.cpp + S80 (MUSA) optimizations.
# Based on patches by pcalau12i (https://www.foleosoft.com/software/S80-Optimization).
set -euo pipefail

# Source is a snapshot of upstream llama.cpp pinned at:
PINNED_COMMIT="85c55223caf0a2ad0d1d88e5a73ab3fe36107867"
UI_VERSION="b10726"

echo "==> Checking MUSA SDK"
if [ ! -d /usr/local/musa ]; then
    echo "ERROR: /usr/local/musa not found. Install MUSA SDK first." >&2
    exit 1
fi

echo "==> Verifying this is the patched source snapshot"
if [ ! -f ggml/src/ggml-cuda/fattn-musa.cu ] || [ ! -f ggml/src/ggml-cuda/mul-mat-q4-mma.cu ]; then
    echo "ERROR: S80 patch files not found. Are you running from the repo root?" >&2
    exit 1
fi
echo "    (upstream snapshot: $PINNED_COMMIT)"

echo "==> Fetching WebUI dist ($UI_VERSION, matches pinned commit)"
BASE="https://hf-mirror.com/buckets/ggml-org/llama-ui/resolve/$UI_VERSION"
if [ ! -f tools/ui/dist/index.html ]; then
    wget -q -O /tmp/llama-ui-dist.tar.gz "$BASE/dist.tar.gz?download=true"
    wget -q -O /tmp/llama-ui-dist.tar.gz.sha256 "$BASE/dist.tar.gz.sha256?download=true"
    expected=$(grep -oE '^[0-9a-fA-F]+' /tmp/llama-ui-dist.tar.gz.sha256 | head -1)
    actual=$(sha256sum /tmp/llama-ui-dist.tar.gz | awk '{print $1}')
    if [ "$expected" != "$actual" ]; then
        echo "ERROR: WebUI dist checksum mismatch" >&2
        exit 1
    fi
    mkdir -p tools/ui/dist
    tar -xzf /tmp/llama-ui-dist.tar.gz -C tools/ui/dist
fi

echo "==> Configuring cmake"
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_MUSA=ON -DMUSA_ARCHITECTURES="21" \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=ON

echo "==> Building llama-server"
cmake --build build --target llama-server -j "$(nproc)"

echo "==> Done. Binary: build/bin/llama-server"
echo "Next: edit run.sh with your model path, LAN IP and API key, then run ./run.sh"
