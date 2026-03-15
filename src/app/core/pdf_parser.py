from pathlib import Path

from docling.document_converter import DocumentConverter

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpeg", ".jpg", ".docx"}


def parse_document_to_markdown(file_path: str | Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    return result.document.export_to_markdown()
