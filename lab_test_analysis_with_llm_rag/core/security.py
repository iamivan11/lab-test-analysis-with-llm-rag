"""Password-derived encryption for local sensitive app data."""

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import DATA_DIR, MODELS_DIR
from core.file_io import atomic_write_bytes, atomic_write_json, atomic_write_text

SECURITY_FILE = DATA_DIR / "security.json"
_SCHEMA_VERSION = 1
_KDF = "pbkdf2-sha256"
_CIPHER = "aes-256-gcm"
_ITERATIONS = 600_000
_SALT_BYTES = 16
_NONCE_BYTES = 12
_KEY_BYTES = 32
_VERIFIER = b"lab-analyzer-password-verifier"
_DATA_KEY_FIELD = "encrypted_data_key"
_session_key: bytes | None = None


class SecurityError(RuntimeError):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _derive_key(password: str, salt: bytes, iterations: int) -> bytes:
    if not password:
        raise SecurityError("Password cannot be empty.")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, _KEY_BYTES)


def _security_payload(password: str, data_key: bytes) -> dict[str, Any]:
    salt = os.urandom(_SALT_BYTES)
    password_key = _derive_key(password, salt, _ITERATIONS)
    return {
        "schema_version": _SCHEMA_VERSION,
        "kdf": _KDF,
        "iterations": _ITERATIONS,
        "salt": _b64(salt),
        _DATA_KEY_FIELD: _encrypt_bytes(data_key, password_key),
        "verifier": _encrypt_bytes(_VERIFIER, password_key),
    }


def _encrypt_bytes(data: bytes, key: bytes) -> dict[str, Any]:
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return {
        "schema_version": _SCHEMA_VERSION,
        "cipher": _CIPHER,
        "nonce": _b64(nonce),
        "ciphertext": _b64(ciphertext),
    }


def _decrypt_payload(payload: dict[str, Any], key: bytes) -> bytes:
    if payload.get("cipher") != _CIPHER:
        raise SecurityError("Unsupported encrypted file format.")
    return AESGCM(key).decrypt(
        _unb64(str(payload["nonce"])),
        _unb64(str(payload["ciphertext"])),
        None,
    )


def is_security_configured() -> bool:
    return SECURITY_FILE.exists()


def is_unlocked() -> bool:
    return _session_key is not None


def setup_password(password: str) -> None:
    global _session_key
    data_key = os.urandom(_KEY_BYTES)
    atomic_write_json(SECURITY_FILE, _security_payload(password, data_key))
    _session_key = data_key


def _data_key_from_password(password: str) -> bytes:
    # Disambiguate user-facing failures: wrong password is the common
    # case (and recoverable by retyping); file/format errors are rare
    # and need different remediation, so don't conflate them.
    try:
        data = json.loads(SECURITY_FILE.read_text(encoding="utf-8"))
    except OSError as e:
        raise SecurityError(f"Cannot read security metadata: {e}") from e
    except json.JSONDecodeError as e:
        raise SecurityError("Security metadata is corrupted (not valid JSON).") from e

    try:
        password_key = _derive_key(
            password,
            _unb64(str(data["salt"])),
            int(data["iterations"]),
        )
    except (KeyError, ValueError) as e:
        raise SecurityError("Security metadata is missing required fields.") from e

    try:
        verifier = _decrypt_payload(data["verifier"], password_key)
        data_key = (
            _decrypt_payload(data[_DATA_KEY_FIELD], password_key)
            if _DATA_KEY_FIELD in data
            else password_key
        )
    except InvalidTag as e:
        raise SecurityError("Invalid password.") from e
    except (KeyError, ValueError) as e:
        raise SecurityError("Security metadata is missing required fields.") from e

    if not hmac.compare_digest(verifier, _VERIFIER):
        raise SecurityError("Invalid password.")
    return data_key


def unlock(password: str) -> None:
    global _session_key
    _session_key = _data_key_from_password(password)


def change_password(current_password: str, new_password: str) -> None:
    global _session_key
    data_key = _data_key_from_password(current_password)
    atomic_write_json(SECURITY_FILE, _security_payload(new_password, data_key))
    _session_key = data_key


def disable_password(current_password: str) -> None:
    global _session_key
    _session_key = _data_key_from_password(current_password)
    # Recursive sweep, not the curated list: any file added to the
    # encryption flow but missed in `decrypt_known_sensitive_files` would
    # otherwise be orphaned (encrypted on disk after the security file is
    # gone, with no key to recover it).
    _decrypt_all_under_data_dir()
    SECURITY_FILE.unlink(missing_ok=True)
    _session_key = None


def _decrypt_all_under_data_dir() -> None:
    skip = {MODELS_DIR.resolve()}
    for root, dirs, files in os.walk(DATA_DIR):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if (root_path / d).resolve() not in skip]
        for fname in files:
            decrypt_file_in_place(root_path / fname)


def lock() -> None:
    global _session_key
    _session_key = None


def _require_key() -> bytes:
    if _session_key is None:
        raise SecurityError("Protected data is locked.")
    return _session_key


def encrypt_bytes(data: bytes) -> bytes:
    return json.dumps(_encrypt_bytes(data, _require_key()), ensure_ascii=False).encode("utf-8")


def decrypt_bytes(data: bytes) -> bytes:
    payload = _encrypted_payload(data)
    if payload is None:
        return data
    return _decrypt_payload(payload, _require_key())


def _encrypted_payload(data: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("cipher") != _CIPHER:
        return None
    return payload


def write_protected_bytes(path: Path, data: bytes) -> None:
    if is_security_configured():
        data = encrypt_bytes(data)
    atomic_write_bytes(path, data)


def read_protected_bytes(path: Path) -> bytes:
    return decrypt_bytes(path.read_bytes())


def write_protected_text(path: Path, text: str) -> None:
    if is_security_configured():
        text = encrypt_bytes(text.encode("utf-8")).decode("utf-8")
    atomic_write_text(path, text)


def read_protected_text(path: Path) -> str:
    return decrypt_bytes(path.read_bytes()).decode("utf-8")


def write_protected_json(path: Path, data: Any) -> None:
    write_protected_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def read_protected_json(path: Path) -> Any:
    return json.loads(read_protected_text(path))


def encrypt_file_in_place(path: Path) -> None:
    if not is_security_configured() or not path.exists():
        return
    data = path.read_bytes()
    if _encrypted_payload(data) is not None:
        return
    write_protected_bytes(path, data)


def decrypt_file_in_place(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if _encrypted_payload(data) is None:
        return
    atomic_write_bytes(path, decrypt_bytes(data))


def migrate_known_sensitive_files() -> None:
    from config import (
        DOCS_DIR,
        FILTERING_OUTPUT_DIR,
        PARSING_OUTPUT_DIR,
        PROFILE_FILE,
        REPORTS_DIR,
        SETTINGS_FILE,
    )
    from core.biomarkers import BIOMARKERS_FILE
    from core.chat_store import CHATS_DIR

    for path in [
        PROFILE_FILE,
        *(path for path in DOCS_DIR.glob("*") if path.is_file()),
        BIOMARKERS_FILE,
        *CHATS_DIR.glob("*.json"),
        *PARSING_OUTPUT_DIR.glob("*.md"),
        *FILTERING_OUTPUT_DIR.glob("*.md"),
        *FILTERING_OUTPUT_DIR.glob("*.meta.json"),
        *REPORTS_DIR.glob("health_report.*"),
    ]:
        if path == SETTINGS_FILE:
            continue
        encrypt_file_in_place(path)


def decrypt_known_sensitive_files() -> None:
    from config import (
        DOCS_DIR,
        FILTERING_OUTPUT_DIR,
        PARSING_OUTPUT_DIR,
        PROFILE_FILE,
        REPORTS_DIR,
        SETTINGS_FILE,
    )
    from core.biomarkers import BIOMARKERS_FILE
    from core.chat_store import CHATS_DIR

    for path in [
        PROFILE_FILE,
        *(path for path in DOCS_DIR.glob("*") if path.is_file()),
        BIOMARKERS_FILE,
        *CHATS_DIR.glob("*.json"),
        *PARSING_OUTPUT_DIR.glob("*.md"),
        *FILTERING_OUTPUT_DIR.glob("*.md"),
        *FILTERING_OUTPUT_DIR.glob("*.meta.json"),
        *REPORTS_DIR.glob("health_report.*"),
    ]:
        if path == SETTINGS_FILE:
            continue
        decrypt_file_in_place(path)
