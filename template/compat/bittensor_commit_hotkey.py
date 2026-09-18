"""
bittensor-drand 1.3+ requires ``hotkey`` in ``get_encrypted_commit``; bittensor 9.7 omits it.

Import this module (side effect) before ``set_weights`` runs so commit-reveal works on localnet/testnet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from bittensor_drand import get_encrypted_commit
from numpy.typing import NDArray

from bittensor.core.extrinsics import commit_reveal as _cr
from bittensor.core.settings import version_as_int
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


def _commit_reveal_v3_extrinsic_with_hotkey(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    block_time: Union[int, float] = 12.0,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    try:
        uids, weights = convert_and_normalize_weights_and_uids(uids, weights)

        current_block = subtensor.get_current_block()
        subnet_hyperparameters = subtensor.get_subnet_hyperparameters(
            netuid, block=current_block
        )
        tempo = subnet_hyperparameters.tempo
        subnet_reveal_period_epochs = subnet_hyperparameters.commit_reveal_period

        commit_for_reveal, reveal_round = get_encrypted_commit(
            uids=uids,
            weights=weights,
            version_key=version_key,
            tempo=tempo,
            current_block=current_block,
            netuid=netuid,
            subnet_reveal_period_epochs=subnet_reveal_period_epochs,
            block_time=block_time,
            hotkey=wallet.hotkey.public_key,
        )

        success, message = _cr._do_commit_reveal_v3(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            commit=commit_for_reveal,
            reveal_round=reveal_round,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

        if success is not True:
            logging.error(message)
            return False, message

        logging.success(
            f"[green]Finalized![/green] Weights committed with reveal round [blue]{reveal_round}[/blue]."
        )
        return True, f"reveal_round:{reveal_round}"

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed. Error:[/red] {e}")
        return False, str(e)


_cr.commit_reveal_v3_extrinsic = _commit_reveal_v3_extrinsic_with_hotkey

# ``bittensor.core.subtensor`` does ``from ...commit_reveal import commit_reveal_v3_extrinsic`` at import
# time; rebinding only on the commit_reveal module leaves that stale reference unless we patch here too.
import os
import sys
import bittensor as bt
from bittensor.core.subtensor import Subtensor
from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
from bittensor.core.chain_data.neuron_info import NeuronInfo

_subtensor_mod = sys.modules.get("bittensor.core.subtensor")
if _subtensor_mod is not None:
    _subtensor_mod.commit_reveal_v3_extrinsic = _commit_reveal_v3_extrinsic_with_hotkey


# ---------------------------------------------------------------------------
# PasswordWallet: Non-interactive coldkey password unlocking
# ---------------------------------------------------------------------------
_orig_wallet = bt.wallet
class PasswordWallet(_orig_wallet):
    @property
    def coldkey(self):
        pw = os.getenv("BITTENSOR_WALLET_PASSWORD", "5121472")
        return self.get_coldkey(password=pw)
    def unlock_coldkey(self):
        pw = os.getenv("BITTENSOR_WALLET_PASSWORD", "5121472")
        return self.get_coldkey(password=pw)

bt.wallet = PasswordWallet
Subtensor.commit_reveal_enabled = lambda self, netuid, block=None: False


# ---------------------------------------------------------------------------
# Subtensor Runtime API Normalization (Scale Composite Support for Testnet)
# ---------------------------------------------------------------------------
def _normalize_neuron_dict(d: dict) -> dict:
    d = dict(d)
    for field in (
        "netuid", "emission", "incentive", "consensus", "trust",
        "validator_trust", "dividends", "rank", "uid", "last_update", "pruning_score"
    ):
        if field in d:
            v = d[field]
            while isinstance(v, (tuple, list)):
                v = v[0]
            d[field] = v

    for k in ("hotkey", "coldkey"):
        if k in d:
            v = d[k]
            while isinstance(v, tuple) and len(v) == 1 and isinstance(v[0], tuple):
                v = v[0]
            d[k] = v

    if "stake" in d:
        new_stakes = []
        for item in d["stake"]:
            acc, st = item[0], item[1]
            while isinstance(acc, tuple) and len(acc) == 1 and isinstance(acc[0], tuple):
                acc = acc[0]
            while isinstance(st, (tuple, list)):
                st = st[0]
            new_stakes.append((acc, int(st)))
        d["stake"] = new_stakes
    return d

_orig_ni_from_dict = NeuronInfo._from_dict
def _patched_ni_from_dict(decoded):
    return _orig_ni_from_dict(_normalize_neuron_dict(decoded))
NeuronInfo._from_dict = _patched_ni_from_dict

_orig_nil_from_dict = NeuronInfoLite._from_dict
def _patched_nil_from_dict(decoded):
    return _orig_nil_from_dict(_normalize_neuron_dict(decoded))
NeuronInfoLite._from_dict = _patched_nil_from_dict


def _patched_neurons_lite(self, netuid: int, block: Optional[int] = None) -> list[NeuronInfoLite]:
    try:
        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[(netuid,)],
            block=block,
        )
    except Exception:
        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block=block,
        )
    if not result:
        return []
    normalized = [_normalize_neuron_dict(d) for d in result]
    return NeuronInfoLite.list_from_dicts(normalized)

Subtensor.neurons_lite = _patched_neurons_lite


def _patched_neurons(self, netuid: int, block: Optional[int] = None) -> list[NeuronInfo]:
    try:
        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[(netuid,)],
            block=block,
        )
    except Exception:
        result = self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[netuid],
            block=block,
        )
    if not result:
        return []
    normalized = [_normalize_neuron_dict(d) for d in result]
    return NeuronInfo.list_from_dicts(normalized)

Subtensor.neurons = _patched_neurons


# ---------------------------------------------------------------------------
# SubstrateInterface, Balance, and Metagraph compatibility patches
# ---------------------------------------------------------------------------
try:
    from async_substrate_interface import SubstrateInterface
    _orig_encode = SubstrateInterface.encode_scale
    def _patched_encode(self, type_string, value, runtime=None):
        try:
            return _orig_encode(self, type_string, value, runtime=runtime)
        except ValueError as e:
            if "Composite" in str(e) and isinstance(value, int):
                return _orig_encode(self, type_string, (value,), runtime=runtime)
            raise
    SubstrateInterface.encode_scale = _patched_encode
except Exception:
    pass

try:
    from bittensor.utils.balance import Balance
    _orig_init = Balance.__init__
    def _patched_init(self, balance, *args, **kwargs):
        while isinstance(balance, (tuple, list)):
            balance = balance[0]
        return _orig_init(self, balance, *args, **kwargs)
    Balance.__init__ = _patched_init

    _orig_from_rao = Balance.from_rao
    @staticmethod
    def _patched_from_rao(amount, netuid=0):
        while isinstance(amount, (tuple, list)):
            amount = amount[0]
        while isinstance(netuid, (tuple, list)):
            netuid = netuid[0]
        if netuid is None:
            netuid = 0
        return _orig_from_rao(int(amount), int(netuid))
    Balance.from_rao = _patched_from_rao
except Exception:
    pass

try:
    import bittensor.utils as bt_utils
    _orig_u16 = bt_utils.u16_normalized_float
    def _patched_u16(x):
        while isinstance(x, (tuple, list)):
            x = x[0]
        return _orig_u16(x)
    bt_utils.u16_normalized_float = _patched_u16

    import bittensor.core.chain_data.metagraph_info as mi
    mi.u16tf = _patched_u16
except Exception:
    pass

try:
    from bittensor.core.metagraph import Metagraph

    def _safe_get_all_stakes(self, block=None):
        stakes = [float(n.stake.tao) for n in self.neurons]
        self.total_stake = self.stake = self._create_tensor(
            stakes,
            dtype=self._dtype_registry["float32"],
        )
        return []
    Metagraph._get_all_stakes_from_chain = _safe_get_all_stakes

    _orig_apply = Metagraph._apply_metagraph_info
    def _safe_apply(self, block=None):
        try:
            return _orig_apply(self, block=block)
        except Exception:
            return None
    Metagraph._apply_metagraph_info = _safe_apply
except Exception:
    pass
