import asyncio
import random
from dataclasses import dataclass

import numpy as np

from template.miner import HazardMinerEngine


@dataclass
class _MockKey:
    ss58_address: str


class MockWallet:
    def __init__(self, hotkey: str = "mock-hotkey-0", coldkey: str = "mock-coldkey-0"):
        self.hotkey = _MockKey(hotkey)
        self.coldkey = _MockKey(coldkey)


@dataclass
class MockAxonInfo:
    uid: int
    hotkey: str
    ip: str = "127.0.0.1"
    port: int = 8091
    is_serving: bool = True


class MockSubtensor:
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        self.netuid = netuid
        self.network = network
        self.chain_endpoint = "mock_endpoint"
        self._block = 1
        owner_hotkey = wallet.hotkey.ss58_address if wallet else "mock-hotkey-0"
        self._hotkeys = [owner_hotkey] + [f"miner-hotkey-{i}" for i in range(1, n + 1)]

    def subnet_exists(self, netuid):
        return netuid == self.netuid

    def is_hotkey_registered(self, netuid, hotkey_ss58):
        return netuid == self.netuid and hotkey_ss58 in self._hotkeys

    def metagraph(self, netuid):
        return MockMetagraph(netuid=netuid, subtensor=self)

    def serve_axon(self, netuid, axon):
        return True

    def set_weights(self, **kwargs):
        return True, "ok"

    def get_current_block(self):
        self._block += 1
        return self._block


class MockMetagraph:
    def __init__(self, netuid=1, network="mock", subtensor=None):
        self.netuid = netuid
        self.network = network
        self.subtensor = subtensor
        self.sync(subtensor=subtensor)

    def sync(self, subtensor=None):
        if subtensor is not None:
            self.subtensor = subtensor
        hotkeys = list(getattr(self.subtensor, "_hotkeys", ["mock-hotkey-0"]))
        self.hotkeys = hotkeys
        self.n = np.array(len(hotkeys))
        self.uids = np.arange(len(hotkeys))
        self.S = np.array([0.0] + [1.0] * (len(hotkeys) - 1), dtype=np.float32)
        self.validator_permit = np.array([False] * len(hotkeys))
        self.last_update = np.array([0] * len(hotkeys))
        self.axons = [MockAxonInfo(uid=i, hotkey=hk) for i, hk in enumerate(hotkeys)]


class MockAxon:
    def __init__(self, wallet, config):
        self.wallet = wallet
        self.config = config
        self._forward = None

    def attach(self, forward_fn, blacklist_fn=None, priority_fn=None):
        self._forward = forward_fn
        return self

    def serve(self, netuid, subtensor):
        return self

    def start(self):
        return self

    def stop(self):
        return self

    def __repr__(self):
        return f"MockAxon(hotkey={self.wallet.hotkey.ss58_address})"


class MockDendrite:
    def __init__(self, wallet):
        self.wallet = wallet
        self.engines = {}

    async def __call__(self, axons, synapse, timeout=12, deserialize=True, **kwargs):
        async def single_response(axon):
            response = synapse.model_copy(deep=True)
            engine = self.engines.setdefault(axon.uid, HazardMinerEngine())
            process_time = random.random()
            if process_time < timeout:
                response = engine.run(response)
            else:
                response.error_message = "Timeout"
            return response.deserialize() if deserialize else response

        return await asyncio.gather(*(single_response(axon) for axon in axons))

    def __str__(self):
        return f"MockDendrite({self.wallet.hotkey.ss58_address})"
