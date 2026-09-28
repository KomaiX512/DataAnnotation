"""Proprietary Validator Module (Confidential v1.5 Protected Module).

This module is protected under proprietary validation architecture licenses.
Runtime execution is authenticated dynamically via in-memory security loader.
"""
from pathlib import Path
from template.validator._security_loader import load_encrypted_module

_ENC_FILE = Path(__file__).with_suffix(".enc")
load_encrypted_module(__name__, _ENC_FILE, globals())
