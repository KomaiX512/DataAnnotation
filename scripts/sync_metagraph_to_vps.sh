#!/usr/bin/env bash
set -euo pipefail

# This script queries the live metagraph on Bittensor testnet (NetUID 498)
# and synchronizes the clean cache to the production VPS (/var/www/canopymrv/artifacts/metagraph_cache.json).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="${WEB_DIR:-$HOME/data-annotation-web}"
VPS_HOST="root@209.74.66.135"
VPS_DEST="/var/www/canopymrv/artifacts/metagraph_cache.json"
SSH_SOCK="/tmp/ssh-canopy-root@209.74.66.135:22"

TMP_FILE="/tmp/canopy_metagraph_cache_tmp.json"
FINAL_FILE="/tmp/canopy_metagraph_cache.json"

"$REPO_DIR/.venv-neurons/bin/python3" "$WEB_DIR/scripts/get_subnet_state.py" > "$TMP_FILE" 2>/dev/null || true

if [ -s "$TMP_FILE" ] && grep -q '"success": true' "$TMP_FILE" && grep -q '"active_miners": [1-9]' "$TMP_FILE"; then
    mv "$TMP_FILE" "$FINAL_FILE"
    mkdir -p "$WEB_DIR/artifacts"
    cp "$FINAL_FILE" "$WEB_DIR/artifacts/metagraph_cache.json"

    # Fast SSH sync using multiplexed socket
    scp -o StrictHostKeyChecking=no \
        -o ControlMaster=auto \
        -o ControlPath="$SSH_SOCK" \
        -o ControlPersist=10m \
        -o ConnectTimeout=5 \
        "$FINAL_FILE" "$VPS_HOST:$VPS_DEST" >/dev/null 2>&1 || true
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Metagraph sync complete: $(grep -o '"active_miners": [0-9]*' "$FINAL_FILE")"
else
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Metagraph sync skipped: output incomplete or 0 active miners"
    rm -f "$TMP_FILE"
fi
