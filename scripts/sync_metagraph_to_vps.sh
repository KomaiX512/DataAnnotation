#!/usr/bin/env bash
set -euo pipefail

# This script queries the live metagraph on Bittensor testnet (NetUID 498)
# and synchronizes the clean cache to the production VPS (/var/www/canopymrv/artifacts/metagraph_cache.json).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WEB_DIR="/home/komail/data-annotation-web"
VPS_HOST="root@209.74.66.135"
VPS_DEST="/var/www/canopymrv/artifacts/metagraph_cache.json"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Querying live metagraph for Subnet 498..."

"$REPO_DIR/.venv-neurons/bin/python3" "$WEB_DIR/scripts/get_subnet_state.py" > /tmp/canopy_metagraph_cache.json 2>/dev/null || true

if [ -s /tmp/canopy_metagraph_cache.json ]; then
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Syncing cache to $VPS_HOST:$VPS_DEST..."
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$VPS_HOST" "mkdir -p /var/www/canopymrv/artifacts"
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=5 /tmp/canopy_metagraph_cache.json "$VPS_HOST:$VPS_DEST"
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Metagraph sync complete."
else
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Warning: /tmp/canopy_metagraph_cache.json was empty or failed."
fi
