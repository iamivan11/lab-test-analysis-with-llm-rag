"""RAG knowledge base for lab test documents.

Pipeline:
1. Chunk documents into ~500-char pieces
2. Embed with BGE-M3 and store in ChromaDB
3. Retrieve via cosine similarity search
"""

import re
import threading
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    DOCS_DIR,
    EMBEDDER_CONFIGS,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_COLLECTION_NAME,
    KB_EMBEDDING_MODEL,
    KB_TOP_K,
)
from core.logger import log

CHROMA_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = KB_COLLECTION_NAME
EMBEDDING_MODEL = KB_EMBEDDING_MODEL
CHUNK_SIZE = KB_CHUNK_SIZE
CHUNK_OVERLAP = KB_CHUNK_OVERLAP
TOP_K = KB_TOP_K

_embedder: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None
_embedder_lock = threading.Lock()
_client_lock = threading.Lock()


# ── Models ─────────────────────────────────────────────────────────────


def _sentence_transformer_kwargs(model_name: str) -> dict:
    config = EMBEDDER_CONFIGS.get(model_name, {})
    return {"device": "cpu", **config.get("loader_kwargs", {})}


def _document_encode_kwargs(model_name: str) -> dict:
    config = EMBEDDER_CONFIGS.get(model_name, {})
    return dict(config.get("document_encode_kwargs", {}))


def _query_encode_kwargs(model_name: str) -> dict:
    config = EMBEDDER_CONFIGS.get(model_name, {})
    return dict(config.get("query_encode_kwargs", {}))


def _get_embedder() -> SentenceTransformer:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            log("KB", f"Loading embedding model {EMBEDDING_MODEL}...")
            _embedder = SentenceTransformer(
                EMBEDDING_MODEL,
                **_sentence_transformer_kwargs(EMBEDDING_MODEL),
            )
            log("KB", "Embedding model loaded")
        return _embedder


def _get_collection() -> chromadb.Collection:
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


# ── Chunking ───────────────────────────────────────────────────────────


def _split_text(text: str, max_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into pieces of ~max_size chars on natural boundaries."""
    if len(text) <= max_size:
        return [text]

    pieces = []
    start = 0
    text_length = len(text)
    while start < text_length:
        remaining = text[start:]
        if len(remaining) <= max_size:
            piece = remaining.strip()
            if piece:
                pieces.append(piece)
            break

        cut = remaining.rfind(". ", 0, max_size)
        if cut > max_size // 3:
            cut += 1
        else:
            cut = remaining.rfind("\n", 0, max_size)
        if cut <= max_size // 3:
            cut = max_size

        end = start + cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        next_start = max(end - CHUNK_OVERLAP, start + 1)
        start = next_start
    return pieces


def chunk_document(
    text: str,
    source: str,
    report_date: str,
    report_type: str,
) -> list[dict]:
    """Split a document into ~500-char chunks with metadata."""
    chunks = []
    for chunk_index, piece in enumerate(_split_text(text, CHUNK_SIZE)):
        if piece.strip():
            chunks.append(
                {
                    "text": piece,
                    "source": source,
                    "report_date": report_date,
                    "report_type": report_type,
                    "chunk_index": chunk_index,
                }
            )
    return chunks


def _extract_date_from_text(text: str) -> str:
    """Extract the report date from labeled date fields in the document."""
    labeled_patterns = [
        r"Collection Date\s*:\s*(.+?)(?:\n|$)",
        r"Received\s*:\s*(.+?)(?:\n|$)",
        r"Verified\s*:\s*(.+?)(?:\n|$)",
        r"(?:Дата регистрации|Дата выполнения|Дата и время|Дата исследования)\s*:?\s*(.+?)(?:\n|$)",
        r"Report Date\s*:\s*(.+?)(?:\n|$)",
        r"(?<!\w)Date\s*:\s*(.+?)(?:\n|$)",
    ]
    for pattern in labeled_patterns:
        m = re.search(pattern, text[:800], re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            date_match = re.match(
                r"(\d{1,2}[/.-]\w{3,}[/.-]\d{4}|\w{3}\s+\d{2}\s+\d{4}|\d{2}[/.]\d{2}[/.]\d{4}|\d{4}-\d{2}-\d{2})",
                val,
            )
            return date_match.group(1) if date_match else val
    return ""


def _extract_date_from_filename(filename: str) -> str:
    """Try to extract date from filename like bt1_20012026.pdf -> 20/01/2026."""
    m = re.search(r"(\d{8})", filename)
    if m:
        digits = m.group(1)
        return f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
    return ""


# ── Indexing ───────────────────────────────────────────────────────────


def index_document(
    filename: str,
    markdown_text: str,
    report_date: str,
    report_type: str,
    on_progress=None,
) -> int:
    """Chunk, embed, and store a document. Returns chunk count."""
    log("KB", f"index_document: {filename}, text length={len(markdown_text)}")
    if on_progress:
        on_progress(f"Chunking {filename}...")

    chunks = chunk_document(
        markdown_text,
        source=filename,
        report_date=report_date,
        report_type=report_type,
    )
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "report_date": c["report_date"],
            "report_type": c["report_type"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    if on_progress:
        on_progress(f"Embedding {len(chunks)} chunks...")

    model = _get_embedder()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        **_document_encode_kwargs(EMBEDDING_MODEL),
    ).tolist()

    if on_progress:
        on_progress(f"Storing {len(chunks)} chunks...")

    collection = _get_collection()
    _remove_by_source(collection, filename)

    ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    log("KB", f"Indexed {filename}: {len(chunks)} chunks")
    return len(chunks)


def remove_document(filename: str) -> None:
    """Remove all chunks for a document from the index."""
    log("KB", f"remove_document: {filename}")
    collection = _get_collection()
    _remove_by_source(collection, filename)


def _remove_by_source(collection: chromadb.Collection, filename: str) -> None:
    """Delete all chunks with the given source filename."""
    try:
        collection.delete(where={"source": filename})
    except ValueError:
        pass
    except Exception as e:
        log("KB", f"Warning: failed to remove chunks for {filename}: {e}")


# ── Retrieval ──────────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve relevant chunks via cosine similarity."""
    log("KB", f"retrieve: query='{query[:80]}', top_k={top_k}")
    collection = _get_collection()
    count = collection.count()
    log("KB", f"Collection has {count} chunks")
    if count == 0:
        return []

    model = _get_embedder()
    query_embedding = model.encode(
        [query],
        show_progress_bar=False,
        **_query_encode_kwargs(EMBEDDING_MODEL),
    ).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i, text in enumerate(results["documents"][0]):
            chunks.append(
                {
                    "text": text,
                    "source": results["metadatas"][0][i]["source"],
                    "report_date": results["metadatas"][0][i].get("report_date", ""),
                    "report_type": results["metadatas"][0][i].get("report_type", ""),
                    "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                    "score": 1 - results["distances"][0][i],
                }
            )

    log(
        "KB", f"Retrieved {len(chunks)} chunks, scores: {[f'{c["score"]:.3f}' for c in chunks[:5]]}"
    )
    return chunks


# ── Queries ────────────────────────────────────────────────────────────


def list_indexed_documents() -> list[str]:
    """Return the set of filenames currently in the index."""
    collection = _get_collection()
    if collection.count() == 0:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    return sorted(set(m["source"] for m in all_meta))


def prune_orphaned_indexed_documents() -> list[str]:
    """Remove indexed sources that no longer exist in DOCS_DIR."""
    indexed = set(list_indexed_documents())
    existing = {
        f.name for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")
    }
    orphaned = sorted(indexed - existing)
    if not orphaned:
        return []

    collection = _get_collection()
    for filename in orphaned:
        _remove_by_source(collection, filename)
        log("KB", f"Pruned orphaned indexed source: {filename}")
    return orphaned


def get_unindexed_documents() -> list[Path]:
    """Return documents in DOCS_DIR that are not yet indexed."""
    indexed = set(list_indexed_documents())
    unindexed = []
    for f in DOCS_DIR.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.name not in indexed:
            unindexed.append(f)
    return sorted(unindexed, key=lambda f: f.name.lower())
