#!/usr/bin/env bash
set -euo pipefail

# Continuous 15-second metagraph synchronization daemon.
# Mirrors live Bittensor Testnet Subnet 498 state to local and production VPS every 15 seconds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_metagraph_to_vps.sh"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting 15s Subnet Metagraph Sync Daemon..."

while true; do
    bash "$SYNC_SCRIPT" || true
    sleep 15
done
