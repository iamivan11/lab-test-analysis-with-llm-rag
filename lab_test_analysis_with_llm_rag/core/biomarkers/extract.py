"""LLM-driven biomarker extraction worker.

Pipeline (Generate): for every uploaded medical document, prompt the LLM
to emit a strict JSON list of quantitative measurements. We aggregate by
the LLM-emitted canonical name (the LLM is the synonym judge — no
hand-coded table), normalize values to a per-biomarker canonical unit,
and persist everything to disk.

Update: the same pipeline but only over documents whose content hash
isn't already present in the cache.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from PySide6.QtCore import Signal

from core.biomarkers.store import (
    _doc_hash,
    _read_parsed,
    list_uploaded_docs,
    load_cache,
    save_cache,
)
from core.llm_engine import generate_stream
from core.logger import log, log_exception
from core.qthread_utils import StoppableQThread


_EXTRACT_SYSTEM_PROMPT = (
    "You are a clinical quantitative medical measurement extractor. Read "
    "the medical document the user provides and emit a JSON array of every "
    "quantitative medical measurement you find. This includes laboratory "
    "tests, spermograms/semen analyses, ultrasound/imaging measurements, "
    "organ sizes, volumes, dimensions, counts, percentages, rates, indexes, "
    "and other numeric clinical findings.\n\n"
    "Output rules:\n"
    "- Output ONLY valid JSON. No prose, no commentary, no markdown fences.\n"
    "- Be exhaustive. Read every heading, paragraph, table row, table "
    "column, and footnote. Do not emit only abnormal or clinically "
    "important values: emit every quantitative measurement that is present.\n"
    "- For rows containing several values or dimensions, emit a separate "
    "JSON object for each value. For example, ultrasound dimensions such as "
    "left testis length, width, thickness, and volume must become separate "
    "measurements with distinct canonical names.\n"
    "- If the document is not in English, translate biomarker names, panels, "
    "units, and dates into the English/schema forms below before emitting "
    "JSON. Do not preserve local-language names when an English clinical "
    "equivalent exists.\n"
    "- Each array element has these keys exactly:\n"
    "  - \"name\": canonical biomarker name in English. Use a short, "
    "standardized name (e.g. \"Total testosterone\", \"Free "
    "testosterone\", \"TSH\", \"LH\", \"Glucose\", \"ALT\", \"Left "
    "testis volume\", \"Right thyroid lobe width\"). Treat "
    "synonyms across documents as the same biomarker — emit the same "
    "canonical name. Distinguish forms that are clinically different "
    "(total vs free, LDL vs HDL, fasting vs postprandial, direct vs "
    "total bilirubin, left vs right side, length vs width vs volume).\n"
    "  - \"panel\": one of \"Reproductive Hormones\", \"Thyroid\", "
    "\"Lipid\", \"Glucose\", \"Liver Function\", \"Kidney Function\", "
    "\"CBC\", \"Vitamins & Minerals\", \"Inflammation\", \"Coagulation\", "
    "\"Spermogram\", \"Hormones\", \"Imaging\", \"Other\".\n"
    "  - \"value\": the numeric value as a JSON number (not a string).\n"
    "  - \"unit\": the unit as reported (e.g. \"ng/dL\", \"nmol/L\", "
    "\"mIU/L\", \"g/dL\", \"mm\", \"cm\", \"mL\"). If the document uses a "
    "decimal comma, convert it to a JSON decimal point.\n"
    "  - \"ref_low\": lower bound of the reference range as a number, or "
    "null if unknown.\n"
    "  - \"ref_high\": upper bound, or null.\n"
    "  - \"date\": ISO date YYYY-MM-DD when the sample was taken (or the "
    "report's date if the sample date is missing); null if unknown.\n\n"
    "Skip qualitative results (\"negative\", \"trace\"), diagnoses, "
    "free-text comments, and anything without a numeric value. Do not "
    "invent reference ranges. If the document has no quantitative "
    "measurements, return an empty `measurements` array.\n\n"
    "The response must be a JSON OBJECT of shape "
    "`{\"measurements\": [ ... ]}` (a JSON Schema is enforced server-side; "
    "any deviation will be rejected).\n"
)


# JSON Schema enforced server-side via OpenAI-style response_format. With
# this set, llama-server constrains the decoder to only produce tokens that
# yield a valid match — eliminating malformed-JSON failures and surrounding
# chatter that previously had to be salvaged by `_strip_to_json`.
_EXTRACTION_SCHEMA: dict = {
    "name": "biomarker_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "measurements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "panel": {"type": "string", "minLength": 1},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                        "ref_low": {"type": ["number", "null"]},
                        "ref_high": {"type": ["number", "null"]},
                        "date": {"type": ["string", "null"]},
                    },
                    "required": [
                        "name",
                        "panel",
                        "value",
                        "unit",
                        "ref_low",
                        "ref_high",
                        "date",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["measurements"],
        "additionalProperties": False,
    },
}


_CANONICALIZE_SYSTEM_PROMPT = (
    "You normalize names of quantitative medical measurements after they "
    "were extracted from user documents. The user provides a JSON array of "
    "unique measurement identities. Each object has: id, name, panel, unit.\n\n"
    "Return ONLY a valid JSON object. No prose, no markdown fences. The "
    "object must map each input id to an object with exactly these keys:\n"
    "  - \"canonical_name\": short canonical English name for the same "
    "clinical measurement.\n"
    "  - \"canonical_panel\": one of \"Reproductive Hormones\", \"Thyroid\", "
    "\"Lipid\", \"Glucose\", \"Liver Function\", \"Kidney Function\", "
    "\"CBC\", \"Vitamins & Minerals\", \"Inflammation\", \"Coagulation\", "
    "\"Spermogram\", \"Hormones\", \"Imaging\", \"Other\".\n\n"
    "Merge names only when they clearly refer to the same clinical "
    "measurement. Normalize synonyms, abbreviations, spelling variants, "
    "word-order variants, and translations across languages. Keep distinct "
    "clinical entities separate: total vs free, LDL vs HDL, fasting vs "
    "postprandial, direct vs total, left vs right, length vs width vs "
    "thickness vs volume, progressive vs non-progressive, concentration vs "
    "total count, and percentage vs absolute count.\n\n"
    "If uncertain whether two measurements are the same, do not merge them. "
    "Preserve the most clinically standard English name available."
)


def _strip_to_json(text: str) -> str:
    """Find the first '[' ... matching ']' span and return it. Some models
    wrap output in fences or chatter despite instructions."""
    text = text.strip()
    # Strip ``` fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find array span.
    start = text.find("[")
    if start < 0:
        return "[]"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:] + "]"  # best-effort close


def _strip_to_json_object(text: str) -> str:
    """Find the first JSON object span and return it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return "{}"
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:] + "}"


def _measurement_identity_key(m: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(m.get("name") or "").strip(),
        str(m.get("panel") or "Other").strip() or "Other",
        str(m.get("unit") or "").strip(),
    )


def _collect_measurement_identities(
    by_doc_hash: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, tuple[str, str, str]]]:
    key_to_id: dict[tuple[str, str, str], str] = {}
    identities: list[dict[str, str]] = []
    id_to_key: dict[str, tuple[str, str, str]] = {}

    for entry in by_doc_hash.values():
        for m in entry.get("measurements", []):
            name, panel, unit = _measurement_identity_key(m)
            if not name:
                continue
            key = (name, panel, unit)
            if key in key_to_id:
                continue
            item_id = f"m{len(identities) + 1}"
            key_to_id[key] = item_id
            id_to_key[item_id] = key
            identities.append(
                {"id": item_id, "name": name, "panel": panel, "unit": unit}
            )
    return identities, id_to_key


def _apply_canonical_measurement_mapping(
    by_doc_hash: dict[str, Any],
    mapping: dict[tuple[str, str, str], tuple[str, str]],
) -> None:
    for entry in by_doc_hash.values():
        for m in entry.get("measurements", []):
            key = _measurement_identity_key(m)
            canonical = mapping.get(key)
            if canonical is None:
                continue
            canonical_name, canonical_panel = canonical
            if canonical_name:
                m["name"] = canonical_name
            if canonical_panel:
                m["panel"] = canonical_panel


def _canonicalize_measurement_names(
    by_doc_hash: dict[str, Any],
    *,
    stop_event: threading.Event,
    max_tokens: int | None,
) -> None:
    identities, id_to_key = _collect_measurement_identities(by_doc_hash)
    if len(identities) < 2:
        return

    history = [{"role": "user", "content": json.dumps(identities, ensure_ascii=False)}]
    collected: list[str] = []
    for kind, tok in generate_stream(
        history,
        context="",
        stop_event=stop_event,
        max_tokens=max_tokens,
        use_rag=False,
        enable_thinking=False,
        system_prompt_override=_CANONICALIZE_SYSTEM_PROMPT,
    ):
        if stop_event.is_set():
            return
        if kind == "thinking":
            continue
        collected.append(tok)

    raw = "".join(collected)
    try:
        payload = json.loads(_strip_to_json_object(raw))
    except json.JSONDecodeError as e:
        log("BIO", f"canonicalization JSON parse failed (first 200 chars: {raw[:200]!r}): {e}")
        return
    if not isinstance(payload, dict):
        return

    mapping: dict[tuple[str, str, str], tuple[str, str]] = {}
    for item_id, key in id_to_key.items():
        item = payload.get(item_id)
        if not isinstance(item, dict):
            continue
        canonical_name = str(item.get("canonical_name") or "").strip()
        canonical_panel = str(item.get("canonical_panel") or "").strip()
        if canonical_name:
            mapping[key] = (canonical_name, canonical_panel)
    _apply_canonical_measurement_mapping(by_doc_hash, mapping)


class BiomarkerExtractionWorker(StoppableQThread):
    """Iterates documents, prompts the LLM per-doc, accumulates JSON.

    Modes:
      MODE_GENERATE: extract every uploaded document from scratch (drops the
                   existing cache before starting).
      MODE_UPDATE: extract only documents whose content hash isn't yet in
                   the cache.
    """

    progress = Signal(str)            # status text
    doc_progress = Signal(int, int)   # done, total
    # (added_doc_count, removed_doc_count). Generate always emits
    # (N, 0); Refresh emits any combination.
    finished_ok = Signal(int, int)
    error_occurred = Signal(str)
    cancelled = Signal()

    MODE_GENERATE = "generate"
    MODE_UPDATE = "update"

    def __init__(self, *, mode: str, max_tokens: int | None = 8192):
        super().__init__()
        self.mode = mode
        # Long lab panels with full reference ranges easily exceed 4 K
        # tokens of JSON. Mid-list truncation drops the tail silently;
        # default raised to 8192 to make truncation rare.
        self.max_tokens = max_tokens

    def run(self) -> None:
        log("BIO", f"BiomarkerExtractionWorker: mode={self.mode}")
        try:
            cache = load_cache() if self.mode == self.MODE_UPDATE else {"by_doc_hash": {}}
            by_doc_hash: dict[str, Any] = dict(cache.get("by_doc_hash", {}))

            uploaded = list_uploaded_docs()

            # In MODE_UPDATE, build the set of hashes that correspond to
            # currently-uploaded documents so we can drop entries from
            # the cache for documents the user has removed. Refresh is
            # meant to sync Trends with the current document set, not
            # only add new docs.
            current_hashes: set[str] = set()
            todo: list[tuple[str, str, str]] = []  # (filename, text, hash)
            for name in uploaded:
                text = _read_parsed(name)
                if not text.strip():
                    log("BIO", f"skip {name}: no parsed markdown")
                    continue
                h = _doc_hash(text)
                current_hashes.add(h)
                if self.mode == self.MODE_UPDATE and h in by_doc_hash:
                    continue
                todo.append((name, text, h))

            removed_count = 0
            if self.mode == self.MODE_UPDATE:
                stale = set(by_doc_hash) - current_hashes
                for h in stale:
                    del by_doc_hash[h]
                removed_count = len(stale)

            total = len(todo)
            if total == 0:
                # No new docs to extract. Still save (cache may have had
                # stale entries pruned above) and report counts.
                save_cache(by_doc_hash)
                self.finished_ok.emit(0, removed_count)
                return

            self.progress.emit(f"Extracting biomarkers from {total} document(s)...")
            self.doc_progress.emit(0, total)

            for i, (name, text, h) in enumerate(todo, start=1):
                if self._stop_event.is_set():
                    self.cancelled.emit()
                    return
                self.progress.emit(f"Reading {name} ({i}/{total})...")
                try:
                    measurements = self._extract_one(text)
                except Exception as e:
                    # A cancel during the per-doc LLM call surfaces as a
                    # generic exception (httpx error from the watcher).
                    # Promote it to cancellation if stop_event is set;
                    # otherwise the UI shows "Extraction failed for X"
                    # instead of finalising the cancel.
                    if self._stop_event.is_set():
                        self.cancelled.emit()
                        return
                    log("BIO", f"extract failed for {name}: {e}")
                    self.error_occurred.emit(f"Extraction failed for {name}: {e}")
                    return
                by_doc_hash[h] = {"source_doc": name, "measurements": measurements}
                self.doc_progress.emit(i, total)

            if self._stop_event.is_set():
                self.cancelled.emit()
                return

            self.progress.emit("Normalizing measurement names...")
            _canonicalize_measurement_names(
                by_doc_hash,
                stop_event=self._stop_event,
                max_tokens=self.max_tokens,
            )
            if self._stop_event.is_set():
                self.cancelled.emit()
                return

            save_cache(by_doc_hash)
            self.finished_ok.emit(total, removed_count)
        except Exception as e:
            log_exception("BIO", "BiomarkerExtractionWorker failed")
            self.error_occurred.emit(str(e))

    def _extract_one(self, doc_text: str) -> list[dict[str, Any]]:
        """One LLM round-trip per document. Returns list of measurements.

        The response is constrained server-side via OpenAI-style
        response_format=json_schema, so the model can only produce tokens
        that match `_EXTRACTION_SCHEMA`. The output shape is
        `{"measurements": [...]}` — we extract the array.
        """
        history = [{"role": "user", "content": doc_text}]
        collected: list[str] = []
        for kind, tok in generate_stream(
            history,
            context="",
            stop_event=self._stop_event,
            max_tokens=self.max_tokens,
            use_rag=False,
            enable_thinking=False,
            system_prompt_override=_EXTRACT_SYSTEM_PROMPT,
            response_format={"type": "json_schema", "json_schema": _EXTRACTION_SCHEMA},
        ):
            if self._stop_event.is_set():
                return []
            if kind == "thinking":
                continue
            collected.append(tok)
        raw = "".join(collected)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            # Schema enforcement should make this unreachable, but keep the
            # salvage path for older llama-server builds that don't support
            # response_format.
            log("BIO", f"JSON parse failed (first 200 chars: {raw[:200]!r}): {e}")
            try:
                payload = json.loads(_strip_to_json_object(raw))
            except (json.JSONDecodeError, ValueError) as e2:
                # Raise instead of returning []: an empty measurements list
                # would be cached by the content hash and the doc never
                # re-extracted, even though we never actually got a valid
                # response from the model.
                raise RuntimeError(
                    f"Model output was not valid JSON ({len(raw)} chars): {e}"
                ) from e2
        if isinstance(payload, list):
            # Pre-schema fallback: treat bare array as the measurements list.
            return payload
        if isinstance(payload, dict):
            if "measurements" not in payload:
                # `_strip_to_json_object` returns "{}" when no JSON object
                # is present in the raw output — caching that as an empty
                # success would silently lose the doc forever.
                raise RuntimeError(
                    "Model output JSON missing 'measurements' field"
                )
            measurements = payload["measurements"]
            if not isinstance(measurements, list):
                raise RuntimeError(
                    f"Model output 'measurements' is not a list: "
                    f"{type(measurements).__name__}"
                )
            return measurements
        raise RuntimeError(
            f"Model output JSON has unexpected shape: {type(payload).__name__}"
        )


__all__ = ["BiomarkerExtractionWorker"]
