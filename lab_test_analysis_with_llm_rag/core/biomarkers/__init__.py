"""Biomarker extraction & storage for the Trends section.

Public API re-exported here so existing callers
(`from core.biomarkers import X`) keep working after the split into a
package. Module breakdown:

- store.py: cache load/save + per-document helpers
- units.py: per-biomarker canonical-unit conversion
- aggregate.py: collapse per-doc cache into per-biomarker time series
- extract.py: BiomarkerExtractionWorker (LLM-driven extraction pipeline)
"""

from core.biomarkers.aggregate import (
    aggregate,
    aggregate_for_dashboard,
    docs_pending_for_update,
    has_pending_refresh,
)
from core.biomarkers.extract import (
    BiomarkerExtractionWorker,
    _apply_canonical_measurement_mapping,
    _CANONICALIZE_SYSTEM_PROMPT,
    _collect_measurement_identities,
    _EXTRACT_SYSTEM_PROMPT,
    _strip_to_json,
    _strip_to_json_object,
)
from core.biomarkers.store import (
    BIOMARKERS_FILE,
    _doc_hash,
    clear_cache,
    has_cache,
    list_uploaded_docs,
    load_cache,
    save_cache,
)
from core.biomarkers.units import convert_to_canonical

__all__ = [
    "BIOMARKERS_FILE",
    "BiomarkerExtractionWorker",
    "aggregate",
    "aggregate_for_dashboard",
    "clear_cache",
    "convert_to_canonical",
    "docs_pending_for_update",
    "has_cache",
    "has_pending_refresh",
    "list_uploaded_docs",
    "load_cache",
    "save_cache",
]
