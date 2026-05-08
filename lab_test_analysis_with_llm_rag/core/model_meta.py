"""Read model metadata (name, context length) from GGUF files."""

from dataclasses import dataclass
from pathlib import Path

from core.logger import log


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
        return ModelMeta(name=Path(model_path).stem, context_length=4096)


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
    name = _str_field(fields, "general.name") or Path(model_path).stem
    context_length = _int_field(fields, f"{arch}.context_length") or 4096

    log("META", f"GGUF: name={name!r}, arch={arch}, ctx={context_length}")

    return ModelMeta(name=name, context_length=context_length)
