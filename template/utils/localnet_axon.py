"""Local multi-miner helpers: chain axon info often disagrees with real listen ports."""

from __future__ import annotations

import os
from typing import FrozenSet, Optional


def localnet_port_map_hotkeys() -> FrozenSet[str]:
    """Hotkeys that have an explicit ``ss58=port`` entry (non-empty map => filter sampling)."""

    raw = os.getenv("LOCALNET_MINER_PORT_BY_SS58", "").strip()
    if not raw:
        return frozenset()
    keys: set[str] = set()
    for part in raw.split(","):
        piece = part.strip()
        if "=" not in piece:
            continue
        key, _ = piece.split("=", 1)
        k = key.strip()
        if k:
            keys.add(k)
    return frozenset(keys)


def localnet_miner_endpoint_override(hotkey_ss58: str) -> Optional[tuple[str, int]]:
    """If ``LOCALNET_MINER_PORT_BY_SS58`` is set, return (ip, port) for this hotkey.

    Supports either:
      - 'hotkey=port' -> ('127.0.0.1', port)
      - 'hotkey=ip:port' -> ('ip', port)
    """

    raw = os.getenv("LOCALNET_MINER_PORT_BY_SS58", "").strip()
    if not raw:
        return None
    hk = (hotkey_ss58 or "").strip()
    for part in raw.split(","):
        piece = part.strip()
        if "=" not in piece:
            continue
        key, val = piece.split("=", 1)
        if key.strip() == hk:
            val_str = val.strip()
            if ":" in val_str:
                host, port_str = val_str.rsplit(":", 1)
                return (host.strip(), int(port_str.strip()))
            return ("127.0.0.1", int(val_str))
    return None


def localnet_miner_port_override(hotkey_ss58: str) -> Optional[int]:
    """If ``LOCALNET_MINER_PORT_BY_SS58`` is set, return the port for this hotkey.

    Format: comma-separated ``ss58=port`` pairs, e.g.
    ``5ABC...=8091,5DEF...=8093``
    """

    ep = localnet_miner_endpoint_override(hotkey_ss58)
    return ep[1] if ep is not None else None

