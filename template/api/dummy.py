# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import base64
from typing import List, Optional, Union, Any, Dict
from template.protocol import HazardDetection
from bittensor.subnets import SubnetsAPI


class HazardAPI(SubnetsAPI):
    def __init__(self, wallet: "bt.wallet"):
        super().__init__(wallet)
        self.netuid = 33
        self.name = "hazard"

    def prepare_synapse(
        self,
        *,
        task_id: str,
        site_id: str,
        challenge_nonce: str,
        image_bytes: bytes,
        dataset_partition: str = "hidden_eval",
        task_type: str = "inference",
    ) -> HazardDetection:
        return HazardDetection(
            task_type=task_type,
            dataset_partition=dataset_partition,
            task_id=task_id,
            site_id=site_id,
            challenge_nonce=challenge_nonce,
            image_b64=base64.b64encode(image_bytes).decode("ascii"),
        )

    def process_responses(
        self, responses: List[Union["bt.Synapse", Any]]
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for response in responses:
            if response.dendrite.status_code != 200:
                continue
            outputs.append(
                {
                    "hazard_detected": response.hazard_detected,
                    "severity": response.severity,
                    "confidence": response.confidence,
                    "osha_refs": response.osha_refs,
                    "model_hash": response.model_hash,
                }
            )
        return outputs
