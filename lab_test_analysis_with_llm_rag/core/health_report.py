"""Health-report generation: gather parsed-document text, prompt the LLM,
and render the result as a PDF.

Layout on disk (under DATA_DIR):
    health_report.pdf   — the rendered report viewed in the UI
    health_report.md    — the markdown source, kept so Update can re-prompt
    health_report.meta.json — { generated_at, used_documents }
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtCore import QMarginsF
from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument

from config import DATA_DIR, DOCS_DIR, PARSING_OUTPUT_DIR
from core.logger import log

HEALTH_REPORT_PDF = DATA_DIR / "health_report.pdf"
HEALTH_REPORT_MD = DATA_DIR / "health_report.md"
HEALTH_REPORT_META = DATA_DIR / "health_report.meta.json"


# ── Disk I/O ───────────────────────────────────────────────────────────────


def report_exists() -> bool:
    return HEALTH_REPORT_PDF.exists() and HEALTH_REPORT_MD.exists()


def load_metadata() -> dict:
    if not HEALTH_REPORT_META.exists():
        return {}
    try:
        return json.loads(HEALTH_REPORT_META.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log("HEALTH", f"Failed to read metadata: {e}")
        return {}


def save_metadata(used_documents: list[str]) -> None:
    HEALTH_REPORT_META.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "used_documents": sorted(used_documents),
            },
            indent=2,
        )
    )


def delete_report() -> None:
    for path in (HEALTH_REPORT_PDF, HEALTH_REPORT_MD, HEALTH_REPORT_META):
        path.unlink(missing_ok=True)


def list_uploaded_documents() -> list[str]:
    """Filenames of all currently-uploaded source documents (DOCS_DIR)."""
    if not DOCS_DIR.exists():
        return []
    return sorted(
        f.name for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    )


def new_documents_since_last_report() -> list[str]:
    """Documents present in DOCS_DIR but not listed in the previous report's
    metadata. Returns [] if no previous report or no new docs."""
    if not report_exists():
        return []
    used = set(load_metadata().get("used_documents", []))
    return [name for name in list_uploaded_documents() if name not in used]


def _read_parsed_markdown(name: str) -> str:
    """Return the full parsed markdown for an uploaded document, or ''.

    Reads from PARSING_OUTPUT_DIR (the unfiltered output of the document
    parser), not from the chunked vector store and not from the trimmed
    FILTERING_OUTPUT_DIR — the report should see the entire source text.
    """
    stem = Path(name).stem
    md_path = PARSING_OUTPUT_DIR / f"{stem}.md"
    if md_path.exists():
        try:
            return md_path.read_text(encoding="utf-8")
        except OSError as e:
            log("HEALTH", f"Failed to read {md_path}: {e}")
    return ""


def gather_document_texts(filenames: list[str]) -> list[tuple[str, str]]:
    """Pair each filename with its full parsed markdown. Filenames whose
    parsed content is missing are skipped."""
    out: list[tuple[str, str]] = []
    for name in filenames:
        text = _read_parsed_markdown(name)
        if text.strip():
            out.append((name, text))
        else:
            log("HEALTH", f"No parsed markdown for {name}, skipping")
    return out


# ── Prompts ────────────────────────────────────────────────────────────────


# Custom system prompt that overrides the chat one — the chat prompt forbids
# lists and asks for concise paragraphs, both of which work against a
# comprehensive multi-document report.
HEALTH_REPORT_SYSTEM_PROMPT = (
    "You are a clinical assistant generating a comprehensive patient "
    "health report from a patient profile and uploaded medical documents.\n\n"
    "OUTPUT RULES:\n"
    "- Markdown only. Use H2 (##) for top-level sections and H3 (###) "
    "for per-document subsections.\n"
    "- Use markdown tables to compare lab values across dates. Every row "
    "starts and ends with `|`. Always leave a blank line before and after "
    "each table.\n"
    "- CHRONOLOGY (mandatory): every table that contains a Date column "
    "must be sorted by date in ascending order — OLDEST first, NEWEST "
    "last. This applies regardless of the order in which the source "
    "documents appear in the input. Use ISO format YYYY-MM-DD for dates. "
    "Rows with no recoverable date go at the end.\n"
    "- When summarizing events in prose (e.g. surgical history), present "
    "them in chronological order as well — oldest first.\n"
    "- Bulleted and numbered lists are allowed where they aid clarity.\n"
    "- Professional clinical tone. No emojis, no exclamation marks.\n"
    "- Be factual: only state what is in the documents or profile. Do not "
    "fabricate values or dates. If a date is unclear, write \"unknown\".\n"
    "- Do not prescribe treatments or medications.\n"
    "- COVERAGE: every uploaded document must be reflected in the report. "
    "Do not omit any document, even if it seems redundant or minor.\n"
)


def _docs_input_blob(docs: list[tuple[str, str]]) -> str:
    return "\n\n---\n\n".join(
        f"### {name}\n\n{content}" for name, content in docs
    )


def _full_report_user_prompt(docs: list[tuple[str, str]]) -> str:
    filenames = "\n".join(f"- {name}" for name, _ in docs)
    return (
        "Generate a comprehensive health report from the patient profile "
        "(in the system context) and ALL of the medical documents listed "
        f"below ({len(docs)} documents).\n\n"
        "## REQUIRED OUTPUT STRUCTURE\n\n"
        "## Patient Overview\n"
        "One short paragraph: name, age, gender, key vitals, and a "
        "one-line summary of presenting concerns based on the documents.\n\n"
        "## Lab Findings Across Dates\n"
        "For every lab marker that appears in two or more documents, "
        "present a markdown table with columns Date | Value | Reference | "
        "Status. **Rows MUST be sorted by Date ascending (oldest at the "
        "top, newest at the bottom).** Use ISO format YYYY-MM-DD. Note "
        "values that fall outside the reference range in the Status "
        "column.\n\n"
        "## Surgical History\n"
        "List surgeries with dates if known, **sorted from oldest to "
        "newest**. State \"None reported\" if no surgeries appear in any "
        "document.\n\n"
        "## Allergies\n"
        "Drug, food, environmental. State \"None reported\" if none.\n\n"
        "## Medications\n"
        "Only those explicitly mentioned in documents.\n\n"
        "## Per-Document Summary\n"
        "For EACH of the documents listed below, write a paragraph "
        "summarizing its key findings. Use the document filename as an "
        "H3 heading. **Do not skip any document.** Order the H3 "
        "subsections chronologically by the document's report date "
        "(oldest first); if a document has no clear date, place it at "
        "the end. Documents to cover:\n\n"
        f"{filenames}\n\n"
        "## Recommendations and Follow-up\n"
        "Brief, factual; no prescriptions.\n\n"
        "---\n\n"
        "## SOURCE DOCUMENTS (input — do not echo verbatim)\n\n"
        + _docs_input_blob(docs)
    )


def _update_report_user_prompt(
    existing_markdown: str, new_docs: list[tuple[str, str]]
) -> str:
    new_filenames = "\n".join(f"- {name}" for name, _ in new_docs)
    return (
        "Below is an existing patient health report followed by NEW "
        "medical documents uploaded since it was written. Produce an "
        "updated report in the same markdown structure as the original.\n\n"
        "Integrate the new findings, extend the lab tables with any new "
        "dated values, and add new surgical history / allergies / "
        "medications. Preserve everything from the original report that is "
        "still accurate. Add an H3 subsection for each new document under "
        "the Per-Document Summary section. New documents to cover:\n\n"
        f"{new_filenames}\n\n"
        "## EXISTING REPORT\n\n"
        + existing_markdown
        + "\n\n---\n\n"
        "## NEW SOURCE DOCUMENTS\n\n"
        + _docs_input_blob(new_docs)
    )


def build_full_report_history(
    profile_context: str, docs: list[tuple[str, str]]
) -> tuple[list[dict], str]:
    return [{"role": "user", "content": _full_report_user_prompt(docs)}], profile_context


def build_update_report_history(
    profile_context: str,
    existing_markdown: str,
    new_docs: list[tuple[str, str]],
) -> tuple[list[dict], str]:
    return (
        [{"role": "user", "content": _update_report_user_prompt(existing_markdown, new_docs)}],
        profile_context,
    )


# ── PDF rendering ──────────────────────────────────────────────────────────


def write_report_pdf(html: str, path: Path = HEALTH_REPORT_PDF) -> None:
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(20, 20, 20, 20))
    doc = QTextDocument()
    doc.setDefaultStyleSheet(
        "body { font-family: -apple-system, Helvetica, Arial, sans-serif; "
        "font-size: 11pt; color: #1e1e2e; }"
        "h1 { font-size: 22pt; margin: 0 0 12px; }"
        "h2 { font-size: 16pt; margin: 18px 0 6px; }"
        "h3 { font-size: 13pt; margin: 12px 0 4px; }"
        "p { margin: 6px 0; line-height: 1.4; }"
        "table { border-collapse: collapse; margin: 8px 0; }"
        "th, td { border: 1px solid #999; padding: 4px 8px; }"
        "th { background: #eaeaea; }"
    )
    doc.setHtml(html)
    doc.print_(writer)


__all__ = [
    "HEALTH_REPORT_MD",
    "HEALTH_REPORT_META",
    "HEALTH_REPORT_PDF",
    "delete_report",
    "list_uploaded_documents",
    "new_documents_since_last_report",
    "report_exists",
]
