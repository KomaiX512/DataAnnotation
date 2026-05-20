from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Protocol

import requests
from PIL import Image


def _miner_str_attr(miner: object, name: str) -> str:
    """Safe for MagicMock-based tests: only real ``str`` values count as set."""

    if not hasattr(miner, name):
        return ""
    v = getattr(miner, name)
    if isinstance(v, str):
        return v.strip()
    return ""


class VlmClient(Protocol):
    def complete_safety_json(
        self,
        *,
        crop: Image.Image,
        full_size: tuple[int, int],
        hazard_class: str,
        detector_confidence: float,
    ) -> dict[str, str]:
        ...


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_vlm_json_object(text: str) -> dict[str, Any]:
    """Parse a single JSON object from model output; tolerate optional markdown fences."""
    raw = text.strip()
    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    return json.loads(raw)


class OpenAICompatVlm:
    """
    Vision-language calls via an OpenAI-compatible HTTP API (LM Studio, vLLM, Ollama OpenAI shim, etc.).

    Default model id targets the smallest Qwen2-VL instruct family member for staging (2B).
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
    ):
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    def complete_safety_json(
        self,
        *,
        crop: Image.Image,
        full_size: tuple[int, int],
        hazard_class: str,
        detector_confidence: float,
    ) -> dict[str, Any]:
        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="JPEG", quality=88)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        w, h = full_size
        user_text = (
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
        url = f"{self._base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout_s)
        resp.raise_for_status()
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected VLM response schema: {data!r}") from exc
        out = parse_vlm_json_object(str(text))
        for k in ("hazard_class", "severity"):
            if k not in out:
                raise ValueError(f"VLM JSON missing required key {k!r}: {out!r}")
        return out


def build_vlm_client(config: Any | None) -> VlmClient:
    miner = getattr(config, "miner", object()) if config is not None else object()
    hf_model = _miner_str_attr(miner, "vlm_hf_model") or os.environ.get("MINER_VLM_HF_MODEL", "").strip()
    if hf_model:
        from template.miner.vlm_hf import get_hf_qwen_vlm_singleton

        device = _miner_str_attr(miner, "vlm_hf_device") or os.environ.get("MINER_VLM_HF_DEVICE", "").strip()
        dtype_name = (
            _miner_str_attr(miner, "vlm_hf_dtype")
            or os.environ.get("MINER_VLM_HF_DTYPE", "auto").strip()
        )
        mx = getattr(miner, "vlm_hf_max_new_tokens", None)
        if isinstance(mx, int) and mx > 0:
            max_new_tokens = mx
        else:
            raw_max = os.environ.get("MINER_VLM_HF_MAX_NEW_TOKENS", "").strip()
            max_new_tokens = int(raw_max) if raw_max else 384

        return get_hf_qwen_vlm_singleton(
            model_id=hf_model,
            device=device,
            dtype_name=dtype_name,
            max_new_tokens=max_new_tokens,
        )

    base = _miner_str_attr(miner, "vlm_openai_base_url") or os.environ.get(
        "MINER_VLM_OPENAI_BASE_URL", ""
    ).strip()
    if not base:
        raise ValueError(
            "Configure a real vision-language model for annotations: either "
            "(1) set --miner.vlm_hf_model or MINER_VLM_HF_MODEL to a Hugging Face Qwen2-VL instruct "
            "model id (e.g. Qwen/Qwen2-VL-2B-Instruct) for in-process inference, or "
            "(2) set --miner.vlm_openai_base_url / MINER_VLM_OPENAI_BASE_URL to an OpenAI-compatible "
            "server (vLLM, LM Studio, etc.). Mock VLM endpoints are not supported."
        )
    api_key = _miner_str_attr(miner, "vlm_openai_api_key") or os.environ.get(
        "MINER_VLM_OPENAI_API_KEY", ""
    ).strip()
    model = (
        _miner_str_attr(miner, "vlm_openai_model")
        or os.environ.get("MINER_VLM_OPENAI_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    ).strip()
    timeout_s = float(getattr(miner, "vlm_request_timeout_s", 120.0) or 120.0)
    return OpenAICompatVlm(base_url=base, api_key=api_key, model=model, timeout_s=timeout_s)
