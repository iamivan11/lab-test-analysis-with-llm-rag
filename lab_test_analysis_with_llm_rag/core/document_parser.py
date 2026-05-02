"""Document parsing via vision LLM.

Sends document page images to the running llama-server (with --mmproj)
to extract structured text using the model's vision capabilities.
"""

import base64
import io
import json
import re
from pathlib import Path

import httpx
from PIL import Image

from config import (
    PARSER_MAX_OUTPUT_TOKENS,
    PARSER_MAX_PARALLEL_PAGES,
    PARSER_METADATA_TIMEOUT_SECONDS,
    PARSER_PDF_DPI,
    PARSER_SANITIZE_TIMEOUT_SECONDS,
    PARSER_VISION_TIMEOUT_SECONDS,
)
from core.logger import log

Image.MAX_IMAGE_PIXELS = None  # Allow high-res medical scans

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpeg", ".jpg"}

_EXTRACTION_PROMPT = (
    "Extract ALL text from this document image exactly as it appears. "
    "Preserve the structure: headings, tables, values, units, reference ranges. "
    "For tables, use plain text columns aligned with | separators. "
    "Do not add commentary or interpretation — only extract what is written."
)
_SANITIZATION_PROMPT = (
    "Clean this parsed medical document before storage.\n\n"
    "Remove only obvious non-medical noise, such as:\n"
    "- lab/clinic branding and marketing text\n"
    "- postal addresses\n"
    "- phone numbers\n"
    "- email addresses\n"
    "- website URLs\n"
    "- legal boilerplate and disclaimers\n"
    "- software/version strings\n"
    "- purely administrative filler that does not affect medical interpretation\n\n"
    "Preserve all medically relevant content exactly, including:\n"
    "- patient names\n"
    "- doctor names\n"
    "- dates\n"
    "- sample type\n"
    "- method\n"
    "- specimen details\n"
    "- test names\n"
    "- results\n"
    "- flags\n"
    "- units\n"
    "- reference ranges\n"
    "- findings\n"
    "- impressions\n"
    "- conclusions\n"
    "- any other clinical context\n\n"
    "Formatting rules:\n"
    "- Preserve the original structure of the document.\n"
    "- If the input contains real markdown tables, keep them as markdown tables.\n"
    "- Preserve table columns, row order, and pipe separators.\n"
    "- Do not flatten tables into prose, lists, or key-value blocks.\n"
    "- Do not convert normal non-table text into fake markdown table rows.\n"
    "- Keep ordinary text as ordinary paragraphs or lines.\n\n"
    "Important:\n"
    "- Do not rewrite, summarize, interpret, or normalize medical content.\n"
    "- Do not delete content unless it is clearly noise.\n"
    "- When unsure, keep the content.\n\n"
    "Return only the cleaned document text."
)
_METADATA_PROMPT = (
    "Extract document metadata from this parsed medical document.\n\n"
    "Return strict JSON with exactly these fields:\n"
    '- "report_date": document/report date in DD/MM/YYYY format, or "" if unknown\n'
    '- "report_type": concise report type like "blood test", "urine test", '
    '"ultrasound", or "" if unknown\n\n'
    "Rules:\n"
    "- Return JSON only. No prose. No markdown.\n"
    "- Use only those 2 fields.\n"
    "- Preserve DD/MM/YYYY format for report_date.\n"
    "- If unsure, use an empty string.\n"
)

# Hidden folders for reviewing raw and filtered parsing results
_RAW_SAVE_DIR: Path | None = None
_FILTERED_SAVE_DIR: Path | None = None


def set_save_dirs(raw_path: Path | None, filtered_path: Path | None) -> None:
    """Set directories where raw and filtered parsing results are saved for review."""
    global _RAW_SAVE_DIR, _FILTERED_SAVE_DIR
    _RAW_SAVE_DIR = raw_path
    _FILTERED_SAVE_DIR = filtered_path
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)
    if filtered_path:
        filtered_path.mkdir(parents=True, exist_ok=True)


def _image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 data URI."""
    buf = io.BytesIO()
    if img.mode == "RGBA" and fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _pdf_to_images(file_path: str) -> list[Image.Image]:
    """Render each PDF page as a PIL Image."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_path)
    images = []
    for page in pdf:
        # Render at configured DPI — preserves small text in medical documents
        bitmap = page.render(scale=PARSER_PDF_DPI / 72)
        images.append(bitmap.to_pil())
    pdf.close()
    return images


def _extract_single_page(
    page_index: int, img: Image.Image, server_url: str, total: int
) -> tuple[int, str]:
    """Extract text from a single page image. Returns (page_index, text)."""
    log("PARSER", f"Sending page {page_index + 1}/{total} to vision model...")
    data_uri = _image_to_base64(img)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": _EXTRACTION_PROMPT},
            ],
        }
    ]
    response = httpx.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "local",
            "messages": messages,
            "stream": False,
            "max_tokens": PARSER_MAX_OUTPUT_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=PARSER_VISION_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    text = message.get("content") or ""
    if not text.strip():
        reasoning = message.get("reasoning_content") or ""
        log(
            "PARSER",
            f"Page {page_index + 1}: empty content "
            f"(reasoning_content={len(reasoning)} chars) — "
            f"model returned nothing usable",
        )
    else:
        log("PARSER", f"Page {page_index + 1}: {len(text)} chars extracted")
    return (page_index, text.strip())


def _extract_text_via_vision(images: list[Image.Image], server_url: str) -> str:
    """Send page images to the LLM server and collect extracted text."""
    total = len(images)
    if total == 1:
        _, text = _extract_single_page(0, images[0], server_url, total)
        return text

    from concurrent.futures import ThreadPoolExecutor

    log("PARSER", f"Processing {total} pages in parallel...")
    with ThreadPoolExecutor(max_workers=min(total, PARSER_MAX_PARALLEL_PAGES)) as pool:
        futures = [
            pool.submit(_extract_single_page, i, img, server_url, total)
            for i, img in enumerate(images)
        ]
        results = [f.result() for f in futures]

    results.sort(key=lambda r: r[0])
    return "\n\n---\n\n".join(text for _, text in results)


def _sanitize_text(text: str, server_url: str) -> str:
    """Remove administrative noise from parsed text while preserving clinical data."""
    log("PARSER", f"Sanitizing parsed text ({len(text)} chars)")
    response = httpx.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": _SANITIZATION_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "max_tokens": PARSER_MAX_OUTPUT_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=PARSER_SANITIZE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    cleaned = (message.get("content") or "").strip()
    if cleaned:
        log("PARSER", f"Sanitization succeeded: {len(cleaned)} chars")
        return cleaned
    raise RuntimeError("Sanitization returned empty text")


def _normalize_report_date(value: str) -> str:
    value = value.strip()
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{day}/{month}/{year}"
    return value if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value) else ""


def extract_document_metadata(text: str, server_url: str) -> dict[str, str]:
    """Extract report metadata from parsed text."""
    log("PARSER", f"Extracting document metadata ({len(text)} chars)")
    response = httpx.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": _METADATA_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=PARSER_METADATA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json").strip()
    data = json.loads(content)
    metadata = {
        "report_date": _normalize_report_date(str(data.get("report_date", ""))),
        "report_type": str(data.get("report_type", "")).strip(),
    }
    log(
        "PARSER",
        "Metadata extracted: "
        f"report_date='{metadata['report_date']}', report_type='{metadata['report_type']}'",
    )
    return metadata


def parse_document(file_path: str | Path, server_url: str) -> str:
    """Parse a document and return extracted text via vision LLM."""
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name
    stem = Path(file_path).stem
    log("PARSER", f"Parsing {name} (ext={ext})")

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    import time

    from core.llm_engine import is_server_running

    if not is_server_running():
        raise RuntimeError("LLM server is not running. Please load a model first.")

    images = _pdf_to_images(file_path) if ext == ".pdf" else [Image.open(file_path)]

    # Try up to 2 times — the server can get stuck after heavy requests
    last_error = None
    for attempt in range(2):
        if attempt > 0:
            log("PARSER", f"Retry {attempt} for {name}, checking server health...")
            for _ in range(10):
                if is_server_running():
                    break
                time.sleep(3)
            else:
                raise RuntimeError(f"Server unresponsive after retry for {name}")

        try:
            text = _extract_text_via_vision(images, server_url)
            if text.strip():
                break
            last_error = RuntimeError(f"Vision model returned empty text for {name}")
        except Exception as e:
            last_error = e
            log("PARSER", f"Attempt {attempt + 1} failed for {name}: {e}")
    else:
        raise last_error  # type: ignore[misc]

    log("PARSER", f"Vision parsing succeeded: {len(text)} chars")
    raw_text = text
    if _RAW_SAVE_DIR:
        raw_save_path = _RAW_SAVE_DIR / f"{stem}.md"
        raw_save_path.write_text(raw_text, encoding="utf-8")
        log("PARSER", f"Saved raw parsing result to {raw_save_path}")

    try:
        text = _sanitize_text(text, server_url)
    except Exception as e:
        log("PARSER", f"Sanitization failed for {name}, using raw parsed text: {e}")

    if _FILTERED_SAVE_DIR:
        filtered_save_path = _FILTERED_SAVE_DIR / f"{stem}.md"
        filtered_save_path.write_text(text, encoding="utf-8")
        log("PARSER", f"Saved filtered parsing result to {filtered_save_path}")

    return text
