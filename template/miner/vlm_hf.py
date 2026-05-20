"""
In-process vision-language inference via Hugging Face Transformers (Qwen2-VL family).

Used when ``--miner.vlm_hf_model`` or ``MINER_VLM_HF_MODEL`` is set instead of an
OpenAI-compatible HTTP endpoint. Loads weights from the Hub (or a local path),
runs ``generate`` on each detector crop, and parses the same JSON contract as
:class:`OpenAICompatVlm` in :mod:`template.miner.vlm_client`.
"""

from __future__ import annotations

import json
import logging
import re
from threading import Lock
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from template.miner.vlm_client import parse_vlm_json_object

_LOG = logging.getLogger(__name__)

_CACHE: Dict[Tuple[str, str, str, int], HuggingFaceQwen2Vlm] = {}
_CACHE_LOCK = Lock()


def _build_instruction(
    *,
    hazard_class: str,
    detector_confidence: float,
    full_size: tuple[int, int],
) -> str:
    w, h = full_size
    return (
        "You are a certified safety director. The attached image is a crop of a construction photo "
        f"(full frame {w}x{h}px) centered on a region flagged by an object detector.\n"
        f"The detector proposes hazard_class={hazard_class!r} with confidence={detector_confidence:.4f}. "
        "Treat the class name as a weak hint only.\n"
        "Analyze the visible evidence and produce:\n"
        "(1) the best hazard class label for the detected object;\n"
        "(2) a severity tier you assign from visual evidence alone.\n"
        "Return a single JSON object with exactly these keys:\n"
        '  "hazard_class" (string),\n'
        '  "severity" (string, exactly one of: none, low, medium, high, critical).\n'
        "No markdown fences, no extra keys, no prose outside the JSON object."
    )


def _resolve_device(explicit: str) -> str:
    e = (explicit or "").strip().lower()
    if e in ("cuda", "cpu", "mps"):
        if e == "cuda" and not torch.cuda.is_available():
            _LOG.warning("vlm_hf_device=cuda requested but CUDA unavailable; using cpu.")
            return "cpu"
        if e == "mps" and not torch.backends.mps.is_available():
            _LOG.warning("vlm_hf_device=mps requested but MPS unavailable; using cpu.")
            return "cpu"
        return e
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str, explicit: str) -> torch.dtype:
    d = (explicit or "auto").strip().lower()
    if d in ("float32", "fp32"):
        return torch.float32
    if d in ("float16", "fp16"):
        return torch.float16
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


class HuggingFaceQwen2Vlm:
    """Qwen2-VL instruct models loaded with Transformers + ``qwen_vl_utils``."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        dtype_name: str,
        max_new_tokens: int,
    ) -> None:
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "Hugging Face VLM requires: pip install 'transformers>=4.46' accelerate qwen-vl-utils"
            ) from exc

        self._model_id = model_id
        self._device_s = _resolve_device(device)
        self._dtype = _resolve_dtype(self._device_s, dtype_name)
        self._max_new_tokens = max(32, int(max_new_tokens))
        self._process_vision_info = process_vision_info

        _LOG.info(
            "event=vlm_hf_load_start model_id=%s device=%s dtype=%s",
            model_id,
            self._device_s,
            self._dtype,
        )
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        load_kw: Dict[str, Any] = {"torch_dtype": self._dtype, "trust_remote_code": True}
        if self._device_s == "cuda":
            load_kw["device_map"] = "auto"
        else:
            load_kw["device_map"] = None
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kw)
        if load_kw.get("device_map") is None:
            self._model.to(self._device_s)
        self._model.eval()
        _LOG.info("event=vlm_hf_load_done model_id=%s", model_id)

    def complete_safety_json(
        self,
        *,
        crop: Image.Image,
        full_size: tuple[int, int],
        hazard_class: str,
        detector_confidence: float,
    ) -> dict[str, Any]:
        user_text = _build_instruction(
            hazard_class=hazard_class,
            detector_confidence=detector_confidence,
            full_size=full_size,
        )
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop.convert("RGB")},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device_s)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        in_ids = inputs["input_ids"]
        trimmed = [out[len(inp) :] for inp, out in zip(in_ids, generated_ids)]
        raw = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        raw = str(raw).strip()
        # Model sometimes wraps JSON in whitespace or a single fence; reuse client parser
        try:
            out = parse_vlm_json_object(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                raise ValueError(f"VLM output is not valid JSON: {raw[:500]!r}") from None
            out = parse_vlm_json_object(m.group(0))
        for k in ("hazard_class", "severity"):
            if k not in out:
                raise ValueError(f"VLM JSON missing required key {k!r}: {out!r}")
        return out


def get_hf_qwen_vlm_singleton(
    *,
    model_id: str,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
) -> HuggingFaceQwen2Vlm:
    """One shared model instance per (model, device, dtype, max_new_tokens)."""

    resolved = _resolve_device(device)
    key = (model_id, resolved, (dtype_name or "auto").strip().lower(), int(max_new_tokens))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is None:
            hit = HuggingFaceQwen2Vlm(
                model_id=model_id,
                device=device,
                dtype_name=dtype_name,
                max_new_tokens=max_new_tokens,
            )
            _CACHE[key] = hit
        return hit
