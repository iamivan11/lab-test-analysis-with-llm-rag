"""Document parsing via vision LLM.

Sends document page images to the running llama-server (with --mmproj)
to extract structured text using the model's vision capabilities.
"""

import base64
import io
from pathlib import Path

import httpx
from PIL import Image

from app.core.logger import log

Image.MAX_IMAGE_PIXELS = None  # Allow high-res medical scans

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpeg", ".jpg"}

_EXTRACTION_PROMPT = (
    "Extract ALL text from this document image exactly as it appears. "
    "Preserve the structure: headings, tables, values, units, reference ranges. "
    "For tables, use plain text columns aligned with | separators. "
    "Do not add commentary or interpretation — only extract what is written."
)

# Hidden folder for reviewing raw parsing results (set externally if needed)
_SAVE_DIR: Path | None = None


def set_save_dir(path: Path | None) -> None:
    """Set a directory where parsed results are saved for review."""
    global _SAVE_DIR
    _SAVE_DIR = path
    if path:
        path.mkdir(parents=True, exist_ok=True)


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
        # Render at 200 DPI — preserves small text in medical documents
        bitmap = page.render(scale=200 / 72)
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
        json={"model": "local", "messages": messages, "stream": False},
        timeout=600,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
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
    with ThreadPoolExecutor(max_workers=min(total, 2)) as pool:
        futures = [
            pool.submit(_extract_single_page, i, img, server_url, total)
            for i, img in enumerate(images)
        ]
        results = [f.result() for f in futures]

    results.sort(key=lambda r: r[0])
    return "\n\n---\n\n".join(text for _, text in results)


def parse_document(file_path: str | Path) -> str:
    """Parse a document and return extracted text via vision LLM."""
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name
    stem = Path(file_path).stem
    log("PARSER", f"Parsing {name} (ext={ext})")

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    import time

    from app.core.llm_engine import SERVER_URL, is_server_running

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
            text = _extract_text_via_vision(images, SERVER_URL)
            if text.strip():
                break
            last_error = RuntimeError(f"Vision model returned empty text for {name}")
        except Exception as e:
            last_error = e
            log("PARSER", f"Attempt {attempt + 1} failed for {name}: {e}")
    else:
        raise last_error  # type: ignore[misc]

    log("PARSER", f"Vision parsing succeeded: {len(text)} chars")

    if _SAVE_DIR:
        save_path = _SAVE_DIR / f"{stem}.md"
        save_path.write_text(text, encoding="utf-8")
        log("PARSER", f"Saved parsing result to {save_path}")

    return text
