"""Read model metadata (name, context length) from GGUF files."""

import re
import struct
from dataclasses import dataclass
from pathlib import Path

from app.core.logger import log

# ── Fast name-only reader (no mmap, no numpy) ──────────────────────────

_GGUF_MAGIC = b"GGUF"
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_str(data: bytes, off: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, off)[0]
    off += 8
    return data[off : off + length].decode("utf-8", errors="replace"), off + length


def read_model_name(model_path: str) -> str | None:
    """Extract general.name from a GGUF file by reading only the first 64 KB."""
    try:
        with Path(model_path).open("rb") as f:
            header = f.read(65536)

        if len(header) < 24 or header[:4] != _GGUF_MAGIC:
            return None

        n_kv = struct.unpack_from("<Q", header, 16)[0]
        off = 24

        for _ in range(n_kv):
            if off + 8 >= len(header):
                break
            key, off = _read_str(header, off)
            if off + 4 >= len(header):
                break
            vtype = struct.unpack_from("<I", header, off)[0]
            off += 4

            if vtype == _TYPE_STRING:
                value, off = _read_str(header, off)
                if key == "general.name":
                    return _clean_name(value)
            elif vtype in _TYPE_SIZES:
                off += _TYPE_SIZES[vtype]
            elif vtype == _TYPE_ARRAY:
                if off + 12 >= len(header):
                    break
                atype = struct.unpack_from("<I", header, off)[0]
                count = struct.unpack_from("<Q", header, off + 4)[0]
                off += 12
                if atype == _TYPE_STRING:
                    for _ in range(count):
                        if off >= len(header):
                            break
                        _, off = _read_str(header, off)
                elif atype in _TYPE_SIZES:
                    off += count * _TYPE_SIZES[atype]
                else:
                    break
            else:
                break
    except Exception:
        pass
    return None


def _clean_name(raw: str) -> str:
    """Normalize a model name: lowercase, keep letters/digits/dots, deduplicate."""
    name = re.sub(r"[^a-zA-Z0-9.]", "-", raw)
    name = re.sub(r"-{2,}", "-", name).strip("-").lower()
    words = [w for w in name.split("-") if w]
    # Drop any word that is a prefix of another (e.g. "qwen" if "qwen3.5" exists)
    filtered: list[str] = []
    for word in words:
        is_prefix = any(other.startswith(word) and other != word for other in words)
        if is_prefix or word in filtered:
            continue
        filtered.append(word)
    return "-".join(filtered)


@dataclass
class ModelMeta:
    name: str
    context_length: int


def read_model_meta(model_path: str) -> ModelMeta:
    """Return metadata for a GGUF model file."""
    try:
        from gguf import GGUFReader

        reader = GGUFReader(model_path, "r")
        return _from_gguf(reader, model_path)
    except Exception as e:
        log("META", f"Failed to read GGUF metadata from {model_path}: {e}")
        return ModelMeta(name=_clean_name(Path(model_path).stem), context_length=4096)


# ── GGUF reading ────────────────────────────────────────────────────────


def _str_field(fields: dict, key: str) -> str | None:
    f = fields.get(key)
    if f is None:
        return None
    try:
        return bytes(f.parts[-1]).decode("utf-8", errors="replace")
    except Exception:
        return None


def _int_field(fields: dict, key: str) -> int | None:
    f = fields.get(key)
    if f is None:
        return None
    try:
        # f.data contains indices into f.parts, not the values themselves
        return int(f.parts[f.data[0]][0])
    except Exception:
        return None


def _from_gguf(reader, model_path: str) -> ModelMeta:
    fields = reader.fields

    arch = _str_field(fields, "general.architecture") or "llama"
    raw_name = _str_field(fields, "general.name") or Path(model_path).stem
    name = _clean_name(raw_name)
    context_length = _int_field(fields, f"{arch}.context_length") or 4096

    log("META", f"GGUF: name={name!r}, arch={arch}, ctx={context_length}")

    return ModelMeta(name=name, context_length=context_length)
