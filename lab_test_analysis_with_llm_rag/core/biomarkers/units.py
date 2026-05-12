"""Per-biomarker canonical-unit conversion.

Maps a canonical biomarker name (lower-cased, with form-distinguishing
prefix preserved — "total testosterone" stays separate from "free
testosterone") to (canonical_unit, {alt_unit_lower: factor_to_canonical}).
A value v reported in `alt_unit` becomes v * factor in canonical_unit.

Coverage is intentionally limited to common labs that show up in the kind
of reports the app handles. For an unknown biomarker we leave the value
and unit as-reported and let the chart fall back to that scale.
"""

from __future__ import annotations


_UMOL_L_KEYS = ["umol/l", "µmol/l", "mcmol/l"]


def _expand_synonyms(units: dict[str, float]) -> dict[str, float]:
    """Common spelling/casing variants get the same factor."""
    return {**units, **{k.replace("/", " / "): v for k, v in units.items()}}


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


__all__ = ["CANONICAL_UNITS", "convert_to_canonical"]
