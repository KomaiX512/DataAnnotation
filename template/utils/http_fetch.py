"""Bounded HTTP(S) GET for validator-side artifact downloads."""

from __future__ import annotations

from urllib.parse import urlparse

DEFAULT_MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


def fetch_url_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    max_bytes: int | None = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> bytes:
    """GET an approved public R2 URL, bounding redirects and response bytes."""
    import requests

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or not (hostname.endswith(".r2.dev") or hostname.endswith(".r2.cloudflarestorage.com"))
    ):
        raise ValueError("Artifact URL must use HTTPS on a Cloudflare R2 host.")

    # Redirects are rejected. Following an attacker-controlled Location header
    # would turn a public object fetch into an SSRF primitive.
    with requests.get(
        url, timeout=timeout, stream=True, allow_redirects=False
    ) as resp:
        if 300 <= resp.status_code < 400:
            raise ValueError("Artifact URL redirects are not permitted.")
        resp.raise_for_status()
        content_length = resp.headers.get("Content-Length")
        if max_bytes is not None and content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                # Invalid Content-Length is ignored; the streamed byte limit
                # below remains authoritative.
                declared_size = None
            if declared_size is not None and declared_size > max_bytes:
                raise ValueError("Artifact exceeds the maximum allowed size.")
        body = bytearray()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            body.extend(chunk)
            if max_bytes is not None and len(body) > max_bytes:
                raise ValueError("Artifact exceeds the maximum allowed size.")
        return bytes(body)
