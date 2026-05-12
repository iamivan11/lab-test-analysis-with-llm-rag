"""Collapse the per-document biomarker cache into per-biomarker time series."""

from __future__ import annotations

import re
from typing import Any

from core.biomarkers.store import (
    _doc_hash,
    _read_parsed,
    list_uploaded_docs,
    load_cache,
)
from core.biomarkers.units import convert_to_canonical


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
            # Reference ranges can be one-sided (e.g. Anti-TPO "<34" gives
            # only `ref_high`; HDL ">40" gives only `ref_low`). Treat each
            # case so points still get a definite in/out colour — without
            # this, half the Anti-TPO dots fell through to the "unknown"
            # bucket and rendered light-blue despite being clearly out of
            # range.
            in_range = None
            if ref_low_conv is not None and ref_high_conv is not None:
                in_range = ref_low_conv <= converted <= ref_high_conv
            elif ref_high_conv is not None:
                in_range = converted <= ref_high_conv
            elif ref_low_conv is not None:
                in_range = converted >= ref_low_conv
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
    for name in list_uploaded_docs():
        text = _read_parsed(name)
        if not text.strip():
            continue
        if _doc_hash(text) not in seen:
            pending.append(name)
    return pending


def has_pending_refresh() -> bool:
    """True if Refresh would do real work — either new documents to
    extract, or stale cached entries to drop because the user removed
    the underlying documents."""
    cached_hashes = set(load_cache().get("by_doc_hash", {}).keys())
    current_hashes = {
        _doc_hash(t)
        for n in list_uploaded_docs()
        if (t := _read_parsed(n)).strip()
    }
    return cached_hashes != current_hashes


__all__ = [
    "aggregate",
    "aggregate_for_dashboard",
    "docs_pending_for_update",
    "has_pending_refresh",
]
