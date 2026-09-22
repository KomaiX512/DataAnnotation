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


def test_resolve_target_axon_external_vs_local(monkeypatch):
    from types import SimpleNamespace
    from template.validator.dual_forward import _resolve_target_axon

    class MockAxon:
        def __init__(self, ip, port):
            self.ip = ip
            self.port = port

    class MockMetagraph:
        def __init__(self):
            self.axons = {
                6: MockAxon("182.176.222.243", 8091),
                13: MockAxon("184.174.33.251", 8091),
            }
            self.hotkeys = {
                6: "5CtC7DQivd6coTohRk1dxjiXdvopq6q7az9BRQxV3qy31271",
                13: "5Hg6ADYHcn7jThwRBQAvwyvmiwyTChTjyhsDNq8hH3pQDsu5",
            }

    mock_self = SimpleNamespace(
        metagraph=MockMetagraph(),
        uid=5,
        config=SimpleNamespace(
            subtensor=SimpleNamespace(
                chain_endpoint="wss://test.finney.opentensor.ai:443",
                network="test",
            )
        ),
    )

    # Set local override for UID 6 only
    monkeypatch.setenv(
        "LOCALNET_MINER_PORT_BY_SS58",
        "5CtC7DQivd6coTohRk1dxjiXdvopq6q7az9BRQxV3qy31271=8091",
    )
    monkeypatch.delenv("LOCALNET_MINER_PORT", raising=False)

    # Local miner 6 routes to 127.0.0.1
    res6 = _resolve_target_axon(mock_self, 6)
    assert res6.ip == "127.0.0.1"
    assert res6.port == 8091

    # External miner 13 routes to their real public IP
    res13 = _resolve_target_axon(mock_self, 13)
    assert res13.ip == "184.174.33.251"
    assert res13.port == 8091


def test_resolve_target_axon_lan_endpoint(monkeypatch):
    from types import SimpleNamespace
    from template.validator.dual_forward import _resolve_target_axon

    class MockAxon:
        def __init__(self, ip, port):
            self.ip = ip
            self.port = port

    class MockMetagraph:
        def __init__(self):
            self.axons = {
                14: MockAxon("182.176.222.243", 8091),
            }
            self.hotkeys = {
                14: "5FHgYbUVesTVhHh3yQjAQm9cEAthaTpZ4DsUj7BC1mRao43e",
            }

    mock_self = SimpleNamespace(
        metagraph=MockMetagraph(),
        uid=5,
        config=SimpleNamespace(
            subtensor=SimpleNamespace(
                chain_endpoint="wss://test.finney.opentensor.ai:443",
                network="test",
            )
        ),
    )

    monkeypatch.setenv(
        "LOCALNET_MINER_PORT_BY_SS58",
        "5FHgYbUVesTVhHh3yQjAQm9cEAthaTpZ4DsUj7BC1mRao43e=10.1.207.169:8091",
    )

    res14 = _resolve_target_axon(mock_self, 14)
    assert res14.ip == "10.1.207.169"
    assert res14.port == 8091


