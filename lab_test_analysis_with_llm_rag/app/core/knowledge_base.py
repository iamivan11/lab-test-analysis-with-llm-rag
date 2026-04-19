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

from app.config import DATA_DIR, DOCS_DIR
from app.core.logger import log

CHROMA_DIR = DATA_DIR / "chromadb"
COLLECTION_NAME = "lab_documents"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 500
TOP_K = 10

_embedder: SentenceTransformer | None = None
_client: chromadb.ClientAPI | None = None
_embedder_lock = threading.Lock()
_client_lock = threading.Lock()


# ── Models ─────────────────────────────────────────────────────────────


def _get_embedder() -> SentenceTransformer:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            log("KB", f"Loading embedding model {EMBEDDING_MODEL}...")
            _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
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
    remaining = text
    while len(remaining) > max_size:
        cut = remaining.rfind(". ", 0, max_size)
        if cut > max_size // 3:
            cut += 1
        else:
            cut = remaining.rfind("\n", 0, max_size)
        if cut <= max_size // 3:
            cut = max_size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_document(text: str, source: str) -> list[dict]:
    """Split a document into ~500-char chunks with metadata."""
    doc_date = _extract_date_from_text(text)
    if not doc_date:
        doc_date = _extract_date_from_filename(source)

    chunks = []
    for piece in _split_text(text, CHUNK_SIZE):
        if piece.strip():
            chunks.append(
                {
                    "text": f"[{source}, {doc_date}]\n{piece}",
                    "source": source,
                    "date": doc_date,
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
    on_progress=None,
) -> int:
    """Chunk, embed, and store a document. Returns chunk count."""
    log("KB", f"index_document: {filename}, text length={len(markdown_text)}")
    if on_progress:
        on_progress(f"Chunking {filename}...")

    chunks = chunk_document(markdown_text, source=filename)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "date": c["date"],
            "chunk_index": i,
        }
        for i, c in enumerate(chunks)
    ]

    if on_progress:
        on_progress(f"Embedding {len(chunks)} chunks...")

    model = _get_embedder()
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

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
    query_embedding = model.encode([query], show_progress_bar=False).tolist()
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
    all_meta = collection.get(include=[])["metadatas"]
    return sorted(set(m["source"] for m in all_meta))


def get_unindexed_documents() -> list[Path]:
    """Return documents in DOCS_DIR that are not yet indexed."""
    indexed = set(list_indexed_documents())
    unindexed = []
    for f in DOCS_DIR.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.name not in indexed:
            unindexed.append(f)
    return sorted(unindexed, key=lambda f: f.name.lower())
