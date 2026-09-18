import os

from template.utils.localnet_axon import localnet_miner_port_override, localnet_port_map_hotkeys


def test_localnet_miner_port_override_parses_ss58_map(monkeypatch):
    monkeypatch.setenv(
        "LOCALNET_MINER_PORT_BY_SS58",
        "5ABCdefGH=8091, 5XYZuvwQR=8093 ",
    )
    assert localnet_miner_port_override("5ABCdefGH") == 8091
    assert localnet_miner_port_override("5XYZuvwQR") == 8093
    assert localnet_miner_port_override("5Other") is None


def test_localnet_miner_port_override_empty(monkeypatch):
    monkeypatch.delenv("LOCALNET_MINER_PORT_BY_SS58", raising=False)
    assert localnet_miner_port_override("5Anything") is None


def test_localnet_port_map_hotkeys(monkeypatch):
    monkeypatch.setenv("LOCALNET_MINER_PORT_BY_SS58", "5AAA=1,5BBB=2")
    assert localnet_port_map_hotkeys() == frozenset({"5AAA", "5BBB"})
