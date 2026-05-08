"""Biomarker extraction & storage for the Trends section.

Pipeline (Generate): for every uploaded medical document, prompt the LLM to
emit a strict JSON list of quantitative measurements. We aggregate by the
LLM-emitted canonical name (the LLM is the synonym judge — no hand-coded
table), normalize values to a per-biomarker canonical unit using the
conversion table below, and persist everything to disk.

Update: the same pipeline but only over documents whose content hash isn't
already present in the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from config import DATA_DIR, DOCS_DIR, FILTERING_OUTPUT_DIR
from core.llm_engine import generate_stream
from core.logger import log
from core.security import read_protected_json, read_protected_text, write_protected_json

BIOMARKERS_FILE = DATA_DIR / "biomarkers.json"


# ── Per-biomarker canonical unit + conversion table ───────────────────────
#
# Maps a canonical biomarker name (lower-cased, with form-distinguishing
# prefix preserved — "total testosterone" stays separate from "free
# testosterone") to:
#   (canonical_unit, {alt_unit_lower: factor_to_canonical})
#
# Pattern: a value v reported in `alt_unit` becomes v * factor in
# canonical_unit. Apply the same factor to ref_low / ref_high.
#
# Coverage is intentionally limited to common labs that show up in the kind
# of reports the app handles. For an unknown biomarker we leave the value
# as-reported and only plot points that share a unit string.

_UMOL_L_KEYS = ["umol/l", "µmol/l", "mcmol/l"]


def _expand_synonyms(units: dict[str, float]) -> dict[str, float]:
    """Common spelling/casing variants get the same factor."""
    out: dict[str, float] = {}
    for k, v in units.items():
        out[k] = v
        out[k.replace("/", " / ")] = v
    return out


CANONICAL_UNITS: dict[str, tuple[str, dict[str, float]]] = {
    # Reproductive hormones
    "total testosterone": ("nmol/L", _expand_synonyms({
        "nmol/l": 1.0, "ng/dl": 0.0347, "ng/ml": 3.467, "pg/ml": 0.00347,
    })),
    "free testosterone": ("pmol/L", _expand_synonyms({
        "pmol/l": 1.0, "pg/ml": 3.467, "ng/dl": 34.67,
    })),
    "shbg": ("nmol/L", {"nmol/l": 1.0}),
    "estradiol": ("pmol/L", _expand_synonyms({
        "pmol/l": 1.0, "pg/ml": 3.671,
    })),
    "lh": ("IU/L", {"iu/l": 1.0, "miu/ml": 1.0, "u/l": 1.0}),
    "fsh": ("IU/L", {"iu/l": 1.0, "miu/ml": 1.0, "u/l": 1.0}),
    "prolactin": ("ng/mL", _expand_synonyms({
        "ng/ml": 1.0, "miu/l": 0.0212, "uiu/ml": 0.0212,
    })),
    # Thyroid
    "tsh": ("mIU/L", _expand_synonyms({
        "miu/l": 1.0, "uiu/ml": 1.0, "µiu/ml": 1.0,
    })),
    "free t4": ("pmol/L", _expand_synonyms({
        "pmol/l": 1.0, "ng/dl": 12.87,
    })),
    "free t3": ("pmol/L", _expand_synonyms({
        "pmol/l": 1.0, "pg/ml": 1.536,
    })),
    # Lipids
    "total cholesterol": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.0259,
    })),
    "ldl cholesterol": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.0259,
    })),
    "hdl cholesterol": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.0259,
    })),
    "triglycerides": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.0113,
    })),
    # Glucose / glycemic
    "glucose": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.0555,
    })),
    "fasting glucose": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.0555,
    })),
    "hba1c": ("%", {"%": 1.0, "mmol/mol": 0.0915}),  # rough IFCC→NGSP-ish; flagged
    # Liver
    "alt": ("U/L", {"u/l": 1.0, "iu/l": 1.0}),
    "ast": ("U/L", {"u/l": 1.0, "iu/l": 1.0}),
    "ggt": ("U/L", {"u/l": 1.0, "iu/l": 1.0}),
    "alp": ("U/L", {"u/l": 1.0, "iu/l": 1.0}),
    "total bilirubin": ("umol/L", _expand_synonyms({
        **dict.fromkeys(_UMOL_L_KEYS, 1.0), "mg/dl": 17.1,
    })),
    "direct bilirubin": ("umol/L", _expand_synonyms({
        **dict.fromkeys(_UMOL_L_KEYS, 1.0), "mg/dl": 17.1,
    })),
    # Kidney
    "creatinine": ("umol/L", _expand_synonyms({
        **dict.fromkeys(_UMOL_L_KEYS, 1.0), "mg/dl": 88.4,
    })),
    "urea": ("mmol/L", _expand_synonyms({
        "mmol/l": 1.0, "mg/dl": 0.357,
    })),
    "egfr": ("mL/min/1.73m²", {"ml/min/1.73m2": 1.0, "ml/min/1.73m²": 1.0}),
    # CBC
    "hemoglobin": ("g/L", _expand_synonyms({
        "g/l": 1.0, "g/dl": 10.0, "mmol/l": 16.114,
    })),
    "hematocrit": ("%", {"%": 1.0, "l/l": 100.0}),
    "wbc": ("10⁹/L", {"10^9/l": 1.0, "10⁹/l": 1.0, "k/ul": 1.0, "x10^9/l": 1.0}),
    "rbc": ("10¹²/L", {"10^12/l": 1.0, "10¹²/l": 1.0, "m/ul": 1.0, "x10^12/l": 1.0}),
    "platelets": ("10⁹/L", {"10^9/l": 1.0, "10⁹/l": 1.0, "k/ul": 1.0, "x10^9/l": 1.0}),
    # Vitamins
    "vitamin d": ("nmol/L", _expand_synonyms({
        "nmol/l": 1.0, "ng/ml": 2.496,
    })),
    "vitamin b12": ("pmol/L", _expand_synonyms({
        "pmol/l": 1.0, "pg/ml": 0.738,
    })),
    "ferritin": ("ng/mL", _expand_synonyms({
        "ng/ml": 1.0, "ug/l": 1.0, "µg/l": 1.0,
    })),
    "iron": ("umol/L", _expand_synonyms({
        **dict.fromkeys(_UMOL_L_KEYS, 1.0), "ug/dl": 0.179,
    })),
    # Spermogram
    "sperm concentration": ("10⁶/mL", {
        "10^6/ml": 1.0, "10⁶/ml": 1.0, "m/ml": 1.0, "mln/ml": 1.0,
    }),
    "sperm motility": ("%", {"%": 1.0}),
    "sperm morphology": ("%", {"%": 1.0}),
}


def _norm_unit(unit: str | None) -> str:
    if not unit:
        return ""
    return unit.strip().lower().replace(" ", "")


def _norm_name(name: str | None) -> str:
    if not name:
        return ""
    return name.strip().lower()


def convert_to_canonical(
    name: str, value: float | None, unit: str | None
) -> tuple[float | None, str]:
    """Return (converted_value, canonical_unit) for a known biomarker.
    Falls back to (value, unit_as_reported) when biomarker unknown."""
    if value is None:
        return None, unit or ""
    entry = CANONICAL_UNITS.get(_norm_name(name))
    if entry is None:
        return value, unit or ""
    canonical_unit, table = entry
    factor = table.get(_norm_unit(unit))
    if factor is None:
        # Unit not in our table → fall back to as-reported.
        return value, unit or ""
    return value * factor, canonical_unit


# ── Storage ────────────────────────────────────────────────────────────────


def _doc_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _list_uploaded_docs() -> list[str]:
    if not DOCS_DIR.exists():
        return []
    return sorted(
        f.name for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    )


def _read_parsed(name: str) -> str:
    """Returns the post-sanitization (filtered) markdown for a document.

    We deliberately read from FILTERING_OUTPUT_DIR — the cleaned-up output
    of the parser's sanitization pass — rather than the raw PARSING_OUTPUT
    text. Filtered text has page-header/footer noise stripped and lab
    tables re-tabulated, which makes structured biomarker extraction
    materially more reliable.
    """
    md = FILTERING_OUTPUT_DIR / f"{Path(name).stem}.md"
    if md.exists():
        try:
            return read_protected_text(md)
        except OSError:
            return ""
    return ""


def load_cache() -> dict[str, Any]:
    if not BIOMARKERS_FILE.exists():
        return {"by_doc_hash": {}, "extracted_at": None}
    try:
        return read_protected_json(BIOMARKERS_FILE)
    except (OSError, json.JSONDecodeError) as e:
        log("BIO", f"cache read failed: {e}")
        return {"by_doc_hash": {}, "extracted_at": None}


def save_cache(by_doc_hash: dict[str, Any]) -> None:
    payload = {
        "extracted_at": datetime.now(UTC).isoformat(),
        "by_doc_hash": by_doc_hash,
    }
    write_protected_json(BIOMARKERS_FILE, payload)


def has_cache() -> bool:
    return BIOMARKERS_FILE.exists() and bool(load_cache().get("by_doc_hash"))


def clear_cache() -> None:
    BIOMARKERS_FILE.unlink(missing_ok=True)


# ── Aggregation ───────────────────────────────────────────────────────────


def _to_iso_date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    # Accept YYYY-MM-DD directly; coerce a couple of common alternatives.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.fullmatch(r"(\d{2})[./](\d{2})[./](\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def aggregate(by_doc_hash: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collapse the per-doc cache into per-biomarker time series.

    Returns a dict keyed by canonical biomarker name. Each entry has:
      panel: clinical-panel string (most-common across measurements)
      unit:  canonical unit after conversion
      points: list of {date, value, ref_low, ref_high, source_doc, in_range}
              sorted by date ascending. Points whose unit couldn't be
              converted to the canonical unit are dropped (mixed-unit
              points within one biomarker would otherwise plot wrong).
      single_point: True if only one measurement (these are filtered out
                    of the dashboard).
    """
    by_biomarker: dict[str, dict[str, Any]] = {}

    for entry in by_doc_hash.values():
        measurements: list[dict[str, Any]] = entry.get("measurements", [])
        source_doc: str = entry.get("source_doc", "")
        for m in measurements:
            name = m.get("name") or ""
            if not name:
                continue
            value = m.get("value")
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            unit = m.get("unit") or ""
            converted, canonical_unit = convert_to_canonical(name, value, unit)
            ref_low_raw = m.get("ref_low")
            ref_high_raw = m.get("ref_high")
            ref_low_conv, _ = convert_to_canonical(name, ref_low_raw, unit)
            ref_high_conv, _ = convert_to_canonical(name, ref_high_raw, unit)
            date = _to_iso_date(m.get("date"))
            if date is None:
                # No date → can't plot on a time axis.
                continue

            slot = by_biomarker.setdefault(
                name,
                {
                    "name": name,
                    "panel": m.get("panel") or "Other",
                    "unit": canonical_unit,
                    "points": [],
                    "_units_seen": set(),
                },
            )
            # If this point's unit didn't match what we've already accepted,
            # drop it rather than mix scales on the same chart.
            if slot["unit"] and canonical_unit and canonical_unit != slot["unit"]:
                continue
            slot["_units_seen"].add(canonical_unit)
            in_range = None
            if ref_low_conv is not None and ref_high_conv is not None:
                in_range = ref_low_conv <= converted <= ref_high_conv
            slot["points"].append(
                {
                    "date": date,
                    "value": converted,
                    "ref_low": ref_low_conv,
                    "ref_high": ref_high_conv,
                    "source_doc": source_doc,
                    "in_range": in_range,
                }
            )

    # Sort by date, mark single-point biomarkers.
    # "Single point" for dashboard purposes means fewer than two distinct
    # dates — multiple measurements on the same date overlap visually as
    # one dot and don't form a trend line, so we hide those too.
    for slot in by_biomarker.values():
        slot["points"].sort(key=lambda p: p["date"])
        unique_dates = {p["date"] for p in slot["points"]}
        slot["single_point"] = len(unique_dates) < 2
        slot.pop("_units_seen", None)

    return by_biomarker


# ── LLM extraction ────────────────────────────────────────────────────────


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


class BiomarkerExtractionWorker(QThread):
    """Iterates documents, prompts the LLM per-doc, accumulates JSON.

    Modes:
      MODE_GENERATE: extract every uploaded document from scratch (drops the
                   existing cache before starting).
      MODE_UPDATE: extract only documents whose content hash isn't yet in
                   the cache.
    """

    progress = Signal(str)            # status text
    doc_progress = Signal(int, int)   # done, total
    finished_ok = Signal()
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
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        log("BIO", f"BiomarkerExtractionWorker: mode={self.mode}")
        try:
            cache = load_cache() if self.mode == self.MODE_UPDATE else {"by_doc_hash": {}}
            by_doc_hash: dict[str, Any] = dict(cache.get("by_doc_hash", {}))

            uploaded = _list_uploaded_docs()
            if not uploaded:
                self.error_occurred.emit("No documents uploaded yet.")
                return

            # Pair each doc with its parsed text + hash.
            todo: list[tuple[str, str, str]] = []  # (filename, text, hash)
            for name in uploaded:
                text = _read_parsed(name)
                if not text.strip():
                    log("BIO", f"skip {name}: no parsed markdown")
                    continue
                h = _doc_hash(text)
                if self.mode == self.MODE_UPDATE and h in by_doc_hash:
                    continue
                todo.append((name, text, h))

            total = len(todo)
            if total == 0:
                # Update with nothing new is a non-error: just save current
                # cache forward so the timestamp updates.
                save_cache(by_doc_hash)
                self.finished_ok.emit()
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
            self.finished_ok.emit()
        except Exception as e:
            log("BIO", f"BiomarkerExtractionWorker: ERROR {e}")
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


# ── Convenience helpers for the UI ────────────────────────────────────────


def aggregate_for_dashboard() -> dict[str, dict[str, Any]]:
    """Load cache + aggregate, drop single-point biomarkers."""
    cache = load_cache()
    grouped = aggregate(cache.get("by_doc_hash", {}))
    return {
        name: slot for name, slot in grouped.items() if not slot.get("single_point", True)
    }


def docs_pending_for_update() -> list[str]:
    """Names of uploaded documents whose hash isn't yet in the cache."""
    cache = load_cache()
    seen = set(cache.get("by_doc_hash", {}).keys())
    pending: list[str] = []
    for name in _list_uploaded_docs():
        text = _read_parsed(name)
        if not text.strip():
            continue
        if _doc_hash(text) not in seen:
            pending.append(name)
    return pending


__all__ = [
    "BIOMARKERS_FILE",
    "BiomarkerExtractionWorker",
    "aggregate",
    "aggregate_for_dashboard",
    "clear_cache",
    "convert_to_canonical",
    "docs_pending_for_update",
    "has_cache",
    "load_cache",
    "save_cache",
]
