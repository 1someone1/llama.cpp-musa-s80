#!/usr/bin/env bash
# Start llama-server with the tested MoE + MTP settings for MTT S80.
# Do NOT add --load-mode none: it corrupts output with split CPU/GPU experts.
set -euo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.8-35B-A3B-Distill-Abliterated-APEX.gguf}"
DRAFT_MODEL="${DRAFT_MODEL:-$HOME/models/mtp-Qwen3.8-35B-A3B-Q8_0.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"                 # 8192 is what was tested; 64K not verified on this branch
N_CPU_MOE="${N_CPU_MOE:-16}"       # 24 expert layers stay on GPU (fastest safe value)
DRAFTS="${DRAFTS:-3}"

# Bind address: default localhost only. Set LAN_ACCESS=1 to expose to LAN.
LAN_IP="127.0.0.1"
if [ "${LAN_ACCESS:-0}" = "1" ] || [ -n "${LAN_IP_OVERRIDE:-}" ]; then
    detected="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)' | head -1)"
    LAN_IP="${LAN_IP_OVERRIDE:-${detected:-127.0.0.1}}"
fi

# API key: only needed for LAN access; use API_KEY env, .api-key file, or generate.
KEY_FILE="$(dirname "$0")/.api-key"
API_KEY=""
if [ "$LAN_IP" != "127.0.0.1" ]; then
    if [ -n "${API_KEY:-}" ]; then
        :
    elif [ -f "$KEY_FILE" ]; then
        API_KEY="$(cat "$KEY_FILE")"
    else
        API_KEY="$(openssl rand -hex 16)"
        printf '%s\n' "$API_KEY" > "$KEY_FILE"
        chmod 600 "$KEY_FILE"
    fi
fi

ARGS=(
    -m "$MODEL"
    -ngl 999 -c "$CTX" -np 1
    --n-cpu-moe "$N_CPU_MOE" -t 16 -tb 16
    --host "$LAN_IP" --port "$PORT"
    --no-warmup
)
if [ -n "$API_KEY" ]; then
    ARGS+=(--api-key "$API_KEY")
fi
if [ -n "$DRAFT_MODEL" ]; then
    ARGS+=(--model-draft "$DRAFT_MODEL")
    ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$DRAFTS")
fi

echo "Starting llama-server:"
echo "  model:  $MODEL"
echo "  draft:  ${DRAFT_MODEL:-none}"
echo "  bind:   $LAN_IP:$PORT"
echo "  ctx:    $CTX"
echo "  n-cpu-moe: $N_CPU_MOE (GPU expert layers: $((40 - N_CPU_MOE)))"
if [ -n "$API_KEY" ]; then
    echo "  LAN mode: key in $KEY_FILE"
else
    echo "  localhost mode: no API key (use LAN_ACCESS=1 to expose)"
fi

exec ./build/bin/llama-server "${ARGS[@]}"
