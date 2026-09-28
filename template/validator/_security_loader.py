"""Runtime Security & Confidentiality Loader for Proprietary Validator Modules.

Protects proprietary evaluation algorithms, vision adjudicator adapters, and scoring
mechanisms from public exposure while allowing seamless authenticated execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes

MAGIC = b"AGYV15ENC"
DEFAULT_KEY = "komail@321"


def get_security_key() -> str:
    """Retrieve validator security key from environment, .env file, or operator default."""
    # 1. Environment variables
    key = os.environ.get("VALIDATOR_SECURITY_KEY") or os.environ.get("VALIDATOR_ENCRYPTION_KEY")
    if key:
        return key.strip()

    # 2. Check .env file in repo root
    try:
        cur = Path(__file__).resolve().parent
        for _ in range(4):
            env_file = cur / ".env"
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("VALIDATOR_SECURITY_KEY=") or line.startswith("VALIDATOR_ENCRYPTION_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if val:
                                return val
            if cur.parent == cur:
                break
            cur = cur.parent
    except Exception:
        pass

    return DEFAULT_KEY


def encrypt_code(source_code: str, passphrase: Optional[str] = None) -> bytes:
    """Encrypt python source code using AES-256-GCM with PBKDF2 key derivation."""
    pw = (passphrase or get_security_key()).strip()
    salt = get_random_bytes(16)
    key = PBKDF2(pw, salt, dkLen=32, count=100000)
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(source_code.encode("utf-8"))
    # Format: MAGIC (9 bytes) + salt (16) + nonce (16) + tag (16) + ciphertext
    return MAGIC + salt + cipher.nonce + tag + ciphertext


def decrypt_code(payload: bytes, passphrase: Optional[str] = None) -> str:
    """Decrypt authenticated AES-256-GCM payload back to python source code."""
    if not payload.startswith(MAGIC):
        raise ValueError("Invalid encrypted payload header (magic mismatch).")

    pw = (passphrase or get_security_key()).strip()
    salt = payload[9:25]
    nonce = payload[25:41]
    tag = payload[41:57]
    ciphertext = payload[57:]

    key = PBKDF2(pw, salt, dkLen=32, count=100000)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        decrypted = cipher.decrypt_and_verify(ciphertext, tag)
    except Exception as exc:
        raise PermissionError(
            "Validator decryption authentication failed. Invalid security key."
        ) from exc
    return decrypted.decode("utf-8")


def load_encrypted_module(module_name: str, enc_file: Path | str, target_globals: Dict[str, Any]) -> None:
    """Decrypt and execute proprietary module code directly into the target module's globals."""
    enc_path = Path(enc_file)
    if not enc_path.exists():
        raise FileNotFoundError(f"Encrypted proprietary module not found: {enc_path}")

    payload = enc_path.read_bytes()
    code_str = decrypt_code(payload)

    # Compile and execute within target module's namespace
    compiled = compile(code_str, f"<secure:{module_name}>", "exec")
    exec(compiled, target_globals)
