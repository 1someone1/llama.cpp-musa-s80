#!/usr/bin/env bash
# Start llama-server with S80-optimized settings.
# Binds to 127.0.0.1 by default. Set LAN_ACCESS=1 to auto-detect LAN IP.
set -euo pipefail

# --- user config (override via env vars) ---
MODEL="${MODEL:-/path/to/your/model.gguf}"
MMPROJ="${MMPROJ:-}"                       # optional: vision projector .gguf
PORT="${PORT:-8080}"
CTX="${CTX:-65536}"                        # 64K is safe with a graphical desktop
DRAFT_MODEL="${DRAFT_MODEL:-}"             # optional: standalone MTP draft .gguf
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

if [ "$MODEL" = "/path/to/your/model.gguf" ]; then
    echo "ERROR: set MODEL to your GGUF file (edit run.sh or use env var)." >&2
    exit 1
fi

ARGS=(
    -m "$MODEL"
    -ngl 999 -c "$CTX" -np 1
    --host "$LAN_IP" --port "$PORT"
    --no-warmup
)
if [ -n "$API_KEY" ]; then
    ARGS+=(--api-key "$API_KEY")
fi
if [ -n "$MMPROJ" ]; then
    ARGS+=(--mmproj "$MMPROJ")
fi
# Two MTP forms:
#  1) model has a built-in nextn head: --spec-type draft-mtp with no draft model
#  2) model ships a standalone draft .gguf: set DRAFT_MODEL and add --model-draft
if [ -n "$DRAFT_MODEL" ]; then
    ARGS+=(--model-draft "$DRAFT_MODEL")
fi
ARGS+=(--spec-type draft-mtp --spec-draft-n-max "$DRAFTS")

echo "Starting llama-server:"
echo "  model: $MODEL"
echo "  bind:  $LAN_IP:$PORT"
echo "  ctx:   $CTX"
if [ -n "$API_KEY" ]; then
    echo "  LAN mode: key in $KEY_FILE"
else
    echo "  localhost mode: no API key (use LAN_ACCESS=1 to expose)"
fi

exec ./build/bin/llama-server "${ARGS[@]}"
