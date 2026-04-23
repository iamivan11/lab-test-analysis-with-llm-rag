"""LLM-powered document organizer.

Three-phase pipeline per category:
1. Classify  — one LLM call using extracted test names → JSON category map
2. Synthesize — one LLM call per category → unified trend tables
3. Refine    — one LLM call per category to strip noise and compress
"""

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path

import httpx

from app.core.llm_engine import SERVER_URL
from app.core.logger import log


def _call_llm(messages: list[dict], timeout: int = 300) -> str:
    """Blocking, non-streaming LLM call.

    Attempts to disable Qwen3 thinking mode for speed; falls back gracefully
    if the loaded model doesn't support that parameter.
    """
    body = {"model": "local", "messages": messages, "stream": False}

    # Try with Qwen3 thinking disabled first
    try:
        response = httpx.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={**body, "chat_template_kwargs": {"enable_thinking": False}},
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    # Fall back: no model-specific params (works with any model)
    response = httpx.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _slugify(name: str) -> str:
    """Convert a category name to a safe filename slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap JSON in."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def _extract_test_names(text: str, max_names: int = 25) -> list[str]:
    """Extract test/parameter names from markdown table data rows.

    Handles tables where the first column is a flag/deviation marker (empty
    for most rows) by picking the first non-empty, non-numeric cell per row.
    """
    names = []
    header_skipped = False

    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            header_skipped = False
            continue

        cells = [re.sub(r"[*_<>✓]", "", c).strip() for c in line.strip("|").split("|")]

        # Separator row (e.g. |---|:---|...)
        if all(re.match(r"^[-:\s]*$", c) for c in cells):
            header_skipped = True
            continue

        # Header row — skip it, mark separator as expected
        if not header_skipped:
            continue

        # Data row: pick first cell that looks like a name, not a number/symbol
        for cell in cells:
            cell = cell.strip()
            if cell and len(cell) > 1 and not re.match(r"^[\d\s<>().,%;±+\-*/=]+$", cell):
                names.append(cell)
                break

        if len(names) >= max_names:
            break

    return names


def _doc_signal(filename: str, text: str) -> str:
    """Build a compact classification signal for a document.

    Uses extracted test names when available; falls back to text preview.
    """
    names = _extract_test_names(text)
    if names:
        return f"[{filename}] tests: {', '.join(names)}"
    # Fall back to first 600 chars of text (skipping pure header lines)
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    preview = " ".join(lines)[:600]
    return f"[{filename}] preview: {preview}"


def organize_documents(
    source_dir: Path,
    master_dir: Path,
    on_progress: Callable[[str], None] | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Classify and synthesize lab documents into master category files.

    Returns the number of master files created. If `stop_event` is set
    between steps, the pipeline exits early; any master files already
    written are left in place.
    """

    def emit(msg: str) -> None:
        log("ORGANIZER", msg)
        if on_progress:
            on_progress(msg)

    def cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    md_files = sorted(source_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No .md files found in {source_dir}")

    emit(f"Found {len(md_files)} documents to compile")

    contents: dict[str, str] = {f.name: f.read_text(encoding="utf-8") for f in md_files}

    if cancelled():
        emit("Cancelled before classification")
        return 0

    # ── Phase 1: Classify ────────────────────────────────────────────────
    emit("Classifying documents...")

    signals = "\n".join(_doc_signal(name, text) for name, text in contents.items())
    log("ORGANIZER", f"Classification signal ({len(signals)} chars):\n{signals}")

    classify_messages = [
        {
            "role": "system",
            "content": (
                "You are a medical document classifier. "
                "Return ONLY a raw JSON object "
                "-- no explanation, no code fences, no text before or after."
            ),
        },
        {
            "role": "user",
            "content": (
                "Group these lab report files into the MINIMUM number of broad medical categories. "
                "Merge aggressively — prefer 3-5 categories over many narrow ones. "
                "Every file must appear in exactly one category. "
                "Base your decision on the test names listed for each file. "
                "Category names: short, lowercase, e.g. 'blood tests', 'semen analysis', "
                "'hormones', 'ultrasound', 'microbiology'.\n\n"
                f"Files:\n{signals}\n\n"
                'Return JSON only: {"category name": ["file1.md", "file2.md"], ...}'
            ),
        },
    ]

    raw = _call_llm(classify_messages, timeout=120)
    raw = _strip_fences(raw)
    log("ORGANIZER", f"Raw classification response:\n{raw}")

    try:
        categories: dict[str, list[str]] = json.loads(raw)
    except json.JSONDecodeError as e:
        emit(f"JSON parse failed ({e}) — retrying...")
        retry = _call_llm([
            {
                "role": "system",
                "content": "Return ONLY a valid JSON object. No other text.",
            },
            {"role": "user", "content": f"Fix and return valid JSON:\n{raw}"},
        ], timeout=60)  # fmt: skip
        categories = json.loads(_strip_fences(retry))

    # Deduplicate: each file must appear in exactly one category (first wins)
    seen: set[str] = set()
    for cat_name in list(categories.keys()):
        unique = [f for f in categories[cat_name] if f not in seen]
        seen.update(unique)
        if unique:
            categories[cat_name] = unique
        else:
            del categories[cat_name]
            log("ORGANIZER", f"Dropped empty category '{cat_name}' after deduplication")

    category_summary = ", ".join(f"'{k}' ({len(v)} files)" for k, v in categories.items())
    emit(f"Categories: {category_summary}")

    # ── Phase 2 + 3: Synthesize then Refine per category ────────────────
    master_dir.mkdir(parents=True, exist_ok=True)
    for old in master_dir.glob("master_*.md"):
        old.unlink()

    created = 0
    total = len(categories)

    for i, (category, filenames) in enumerate(categories.items(), 1):
        if cancelled():
            emit(f"Cancelled after {created} master files")
            return created
        emit(f"[{i}/{total}] Synthesizing '{category}'...")

        parts = [
            f"=== {fname} ===\n{contents[fname]}"
            for fname in filenames
            if fname in contents and contents[fname].strip()
        ]
        if not parts:
            log("ORGANIZER", f"Skipping '{category}' — no content")
            continue

        # Cap combined input to avoid context overflow: ~6 000 chars ≈ 2 000 tokens input
        MAX_COMBINED = 6_000
        combined = "\n\n".join(parts)
        if len(combined) > MAX_COMBINED:
            # Trim each part proportionally
            budget = MAX_COMBINED // len(parts)
            parts = [p[:budget] for p in parts]
            combined = "\n\n".join(parts)
            log("ORGANIZER", f"'{category}': trimmed to {len(combined)} chars (was larger)")

        log("ORGANIZER", f"'{category}': {len(filenames)} files, {len(combined)} chars combined")

        synthesis_messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical data synthesizer. "
                    "Preserve all numeric values, units, and reference ranges exactly. "
                    "Do not interpret or add recommendations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"You have {len(parts)} lab reports from different dates, "
                    f"category: {category}.\n\n"
                    "Create a single unified master document showing TRENDS over time.\n\n"
                    "Rules:\n"
                    "- For each test that appears in multiple reports, create ONE pivot table "
                    "where rows = individual tests and columns = dates. Example:\n\n"
                    "| Test | Unit | Ref. Range | 2025-03-01 | 2025-09-22 | 2026-01-20 |\n"
                    "| --- | --- | --- | --- | --- | --- |\n"
                    "| Testosterone Total | ng/mL | 2.64-9.16 | 7.21 | 8.40 | **9.53 H** |\n\n"
                    "- Mark abnormal values bold with H (high) or L (low): **9.53 H**\n"
                    "- Group related tests under headings\n"
                    "- Tests that appear in only one report go in a 'Single Measurement' section\n"
                    "- Do not repeat any test in multiple tables\n"
                    "- Do not add interpretation or recommendations\n\n"
                    f"Reports:\n\n{combined}"
                ),
            },
        ]

        master_text = _call_llm(synthesis_messages, timeout=300)
        log("ORGANIZER", f"'{category}' synthesis: {len(master_text)} chars")

        if not master_text.strip():
            emit(f"[{i}/{total}] WARNING: synthesis returned empty for '{category}', skipping")
            continue

        if cancelled():
            emit(f"Cancelled after {created} master files")
            return created

        # ── Phase 3: Refine ──────────────────────────────────────────────
        emit(f"[{i}/{total}] Refining '{category}'...")

        refine_messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical document editor. "
                    "Your only job is to improve signal-to-noise ratio. "
                    "Never alter any numeric value, unit, date, or reference range. "
                    "When in doubt, keep the content."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Refine this medical document in three steps:\n\n"
                    "1. CLEAN — Remove non-clinical noise: clinic/lab names, addresses, "
                    "phone numbers, doctor names, patient names, license numbers, "
                    "legal disclaimers, software version strings, and administrative filler. "
                    "Keep all test names, values, dates, units, reference ranges.\n\n"
                    "2. AUDIT — Remove any content that clearly does not belong to this "
                    "document's category (e.g. a blood count result in a semen analysis file). "
                    "If it could belong, keep it.\n\n"
                    "3. COMPRESS — Merge duplicate tables covering the same tests. "
                    "Remove exact duplicate rows. Trim redundant prose. "
                    "Do not remove any unique data point.\n\n"
                    "Return only the refined document. No explanation.\n\n"
                    f"Document:\n\n{master_text}"
                ),
            },
        ]

        refined_text = _call_llm(refine_messages, timeout=300)
        log(
            "ORGANIZER",
            f"'{category}' refined: {len(refined_text)} chars (was {len(master_text)})",
        )

        # Safety: if refine wiped most content, fall back to synthesis output
        if len(refined_text.strip()) < max(200, len(master_text) * 0.15):
            log("ORGANIZER", f"'{category}' refine output too small — using synthesis output")
            refined_text = master_text

        slug = _slugify(category)
        out_path = master_dir / f"master_{slug}.md"
        out_path.write_text(refined_text, encoding="utf-8")
        emit(f"[{i}/{total}] Saved {out_path.name}")
        created += 1

    # ── Phase 4: Cross-file reconciliation ──────────────────────────────
    # For each master file, identify content that belongs to a different
    # category, strip it out, and merge it into the correct target file.
    master_files = {p.stem.removeprefix("master_"): p for p in master_dir.glob("master_*.md")}
    category_names = list(master_files.keys())

    # Maps target_slug -> list of markdown snippets to merge in
    orphans: dict[str, list[str]] = {slug: [] for slug in category_names}

    if cancelled():
        emit(f"Cancelled before reconciliation — {created} master files")
        return created

    emit("Reconciling cross-category data...")

    for slug, path in master_files.items():
        if cancelled():
            emit(f"Cancelled during reconciliation — {created} master files")
            return created
        current_text = path.read_text(encoding="utf-8")
        others = [c for c in category_names if c != slug]
        if not others:
            continue

        # Cap document size sent for audit to avoid 500 errors
        audit_text = current_text[:4_000]

        audit_messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical document auditor. Return ONLY valid JSON, no other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"This is the '{slug}' master file. "
                    f"Other categories that exist: {', '.join(others)}.\n\n"
                    "Identify any rows, tables, or sections in this document that CLEARLY belong "
                    f"to a different category — not '{slug}' — and should be moved there.\n\n"
                    'Return JSON: {"target_category": "markdown content to move"}. '
                    "Use the exact category names from the list above as keys. "
                    "If nothing is misplaced, return {}.\n\n"
                    f"Document:\n\n{audit_text}"
                ),
            },
        ]

        try:
            raw = _strip_fences(_call_llm(audit_messages, timeout=120))
        except Exception as e:
            log("ORGANIZER", f"Reconcile '{slug}': LLM call failed ({e}), skipping")
            continue

        log("ORGANIZER", f"Reconcile '{slug}': raw audit = {raw[:200]}")

        try:
            audit: dict[str, str] = json.loads(raw)
        except json.JSONDecodeError:
            log("ORGANIZER", f"Reconcile '{slug}': JSON parse failed, skipping")
            continue

        if not audit:
            log("ORGANIZER", f"Reconcile '{slug}': nothing misplaced")
            continue

        # Remove the misplaced content from the source file
        all_misplaced = "\n\n".join(audit.values())
        remove_messages = [
            {
                "role": "system",
                "content": "You are a medical document editor. Return only the corrected document.",
            },
            {
                "role": "user",
                "content": (
                    f"Remove the following content from this document (it belongs elsewhere). "
                    f"Do not alter anything else.\n\n"
                    f"Content to remove:\n{all_misplaced}\n\n"
                    f"Document:\n\n{current_text}"
                ),
            },
        ]
        try:
            cleaned = _call_llm(remove_messages, timeout=120)
        except Exception as e:
            log("ORGANIZER", f"Reconcile '{slug}': remove call failed ({e}), skipping removal")
            cleaned = ""
        if cleaned and len(cleaned.strip()) >= max(100, len(current_text) * 0.1):
            path.write_text(cleaned, encoding="utf-8")
            log(
                "ORGANIZER",
                f"Reconcile '{slug}': removed misplaced content "
                f"({len(current_text)} -> {len(cleaned)} chars)",
            )

        # Queue orphans for their target files
        for target_slug, snippet in audit.items():
            # Normalize: "semen analysis" -> "semen_analysis"
            normalized = _slugify(target_slug)
            if normalized in orphans:
                orphans[normalized].append(snippet)
                log("ORGANIZER", f"Reconcile: queued {len(snippet)} chars for '{normalized}'")

    # Merge orphaned content into target master files
    for slug, snippets in orphans.items():
        if cancelled():
            emit(f"Cancelled during orphan merge — {created} master files")
            return created
        if not snippets:
            continue
        path = master_files.get(slug)
        if not path or not path.exists():
            log("ORGANIZER", f"Reconcile: target '{slug}' not found, skipping orphan merge")
            continue

        existing = path.read_text(encoding="utf-8")
        incoming = "\n\n".join(snippets)
        emit(f"Merging orphaned data into '{slug}'...")

        merge_messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical document merger. "
                    "Preserve all values exactly. Do not interpret or add recommendations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Merge the following data into the existing '{slug}' master document.\n\n"
                    "Rules:\n"
                    "- Add new rows to existing tables where the test/parameter "
                    "already has a table\n"
                    "- If no matching table exists, add the data in the appropriate section\n"
                    "- Do not duplicate data that is already present\n"
                    "- Maintain the chronological column order\n\n"
                    f"Existing document:\n\n{existing}\n\n"
                    f"Data to merge in:\n\n{incoming}"
                ),
            },
        ]

        try:
            merged = _call_llm(merge_messages, timeout=180)
        except Exception as e:
            log("ORGANIZER", f"Reconcile: merge into '{slug}' failed ({e}), skipping")
            merged = ""
        if merged and len(merged.strip()) >= max(len(existing), 100):
            path.write_text(merged, encoding="utf-8")
            log(
                "ORGANIZER",
                f"Reconcile: merged into '{slug}' ({len(existing)} -> {len(merged)} chars)",
            )

    emit(f"Done — {created} master files created")
    return created
