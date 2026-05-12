from config import (
    LLM_HYDE_TIMEOUT_SECONDS,
    LLM_QUERY_MODIFICATION_TIMEOUT_SECONDS,
    LLM_RAG_COMPRESSION_TIMEOUT_SECONDS,
    LLM_TOKENIZE_TIMEOUT_SECONDS,
    RAG_DEBUG_DIR,
    RAG_DEBUG_ENABLED,
    load_ctx_size,
)
from core.file_io import atomic_write_text
from core.http_client import post_with_retries
from core.llm_server import SERVER_URL, get_current_model_path
from core.logger import log
from core.prompts import (
    RAG_COMPRESSION_PROMPT,
    RAG_HYDE_PROMPT,
    RAG_QUERY_MODIFICATION_PROMPT,
    RAG_QUERY_REPHRASE_PROMPT,
)

_TOKEN_CACHE_MAX_ITEMS = 4096
_token_count_cache: dict[str, int] = {}


def _dump_chunks_for_debug(query: str, chunks: list[dict]) -> None:
    if not RAG_DEBUG_ENABLED:
        return
    try:
        if RAG_DEBUG_DIR.exists():
            for old in RAG_DEBUG_DIR.iterdir():
                if old.is_file():
                    old.unlink()
        RAG_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        width = max(2, len(str(len(chunks))))
        for i, chunk in enumerate(chunks, start=1):
            path = RAG_DEBUG_DIR / f"chunk_{i:0{width}d}.md"
            atomic_write_text(
                path,
                f"# Query\n\n{query}\n\n"
                f"# Source\n\n{chunk['source']}\n\n"
                f"# Chunk\n\n{chunk['text']}\n"
            )
    except OSError as e:
        log("RAG", f"Failed to dump chunks: {e}")


def rag_token_budget(ctx_size: int) -> int:
    return int(ctx_size * 0.33)


def count_tokens(text: str, server_url: str = SERVER_URL, stop_event=None) -> int:
    if server_url == SERVER_URL and text in _token_count_cache:
        return _token_count_cache[text]

    r = post_with_retries(
        f"{server_url}/tokenize",
        json={"content": text},
        timeout=LLM_TOKENIZE_TIMEOUT_SECONDS,
        stop_event=stop_event,
    )
    r.raise_for_status()
    token_count = len(r.json().get("tokens", []))
    if server_url == SERVER_URL:
        if len(_token_count_cache) >= _TOKEN_CACHE_MAX_ITEMS:
            _token_count_cache.clear()
        _token_count_cache[text] = token_count
    return token_count


def _rag_char_budget_fallback(ctx_size: int) -> int:
    return int(ctx_size * 3 * 0.33)


def _format_chunk_for_context(chunk: dict) -> str:
    metadata = [
        f"Source: {chunk.get('source', '')}",
        f"Report date: {chunk.get('report_date', '')}",
        f"Report type: {chunk.get('report_type', '')}",
        f"Chunk index: {chunk.get('chunk_index', 0)}",
    ]
    return f"[{' | '.join(metadata)}]\n{chunk['text']}"


def _pack_chunks_by_tokens(
    chunks: list[dict],
    ctx_size: int,
    max_tokens: int | None = None,
    *,
    stop_event=None,
) -> tuple[list[str], int]:
    max_tokens = rag_token_budget(ctx_size) if max_tokens is None else max_tokens
    if max_tokens <= 0:
        return [], 0
    separator = "\n\n---\n\n"
    try:
        sep_tokens = count_tokens(separator, stop_event=stop_event)
    except Exception as e:
        log("RAG", f"Tokenizer unavailable ({e}); falling back to char budget")
        return _pack_chunks_by_chars(chunks, max_tokens * 3)

    entries: list[str] = []
    total = 0
    for chunk in chunks:
        entry = _format_chunk_for_context(chunk)
        try:
            entry_tokens = count_tokens(entry, stop_event=stop_event)
        except Exception as e:
            log("RAG", f"Tokenizer failed mid-pack ({e}); falling back to char budget")
            return _pack_chunks_by_chars(chunks, max_tokens * 3)
        if not entries and entry_tokens > max_tokens:
            break
        if entries and total + entry_tokens + sep_tokens > max_tokens:
            break
        entries.append(entry)
        total += entry_tokens + (sep_tokens if len(entries) > 1 else 0)
    return entries, total


def _pack_chunks_by_chars(chunks: list[dict], max_chars: int) -> tuple[list[str], int]:
    if max_chars <= 0:
        return [], 0
    entries: list[str] = []
    total = 0
    for chunk in chunks:
        entry = _format_chunk_for_context(chunk)
        if entries and total + len(entry) + 7 > max_chars:
            break
        entries.append(entry)
        total += len(entry) + (7 if len(entries) > 1 else 0)
    return entries, total


def _compress_rag_context(query: str, context: str, *, stop_event=None) -> str:
    if not context.strip():
        return context

    response = post_with_retries(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": RAG_COMPRESSION_PROMPT},
                {
                    "role": "user",
                    "content": f"Question:\n{query}\n\nRetrieved context:\n{context}",
                },
            ],
            "stream": False,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=LLM_RAG_COMPRESSION_TIMEOUT_SECONDS,
        stop_event=stop_event,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    compressed = (message.get("content") or "").strip()
    if not compressed:
        raise RuntimeError("RAG compression returned empty text")
    return compressed


def _modify_retrieval_query(query: str) -> str:
    if not query.strip():
        return query

    response = post_with_retries(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": RAG_QUERY_MODIFICATION_PROMPT},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=LLM_QUERY_MODIFICATION_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    modified = (message.get("content") or "").strip()
    if not modified:
        raise RuntimeError("RAG query modification returned empty text")
    return modified


def _rephrase_retrieval_query(query: str) -> str:
    if not query.strip():
        return query

    response = post_with_retries(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": RAG_QUERY_REPHRASE_PROMPT},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=LLM_QUERY_MODIFICATION_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    rephrased = (message.get("content") or "").strip()
    if not rephrased:
        raise RuntimeError("RAG query rephrase returned empty text")
    return rephrased


def _generate_hyde_query(query: str) -> str:
    if not query.strip():
        return query

    response = post_with_retries(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": RAG_HYDE_PROMPT},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=LLM_HYDE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    hyde_query = (message.get("content") or "").strip()
    if not hyde_query:
        raise RuntimeError("HyDE returned empty text")
    return hyde_query


def _build_hyde_retrieval_query(
    query: str, hyde_excerpt: str, expanded_query: str | None = None
) -> str:
    if expanded_query and expanded_query.strip() != query.strip():
        retrieval_query = expanded_query.strip()
    else:
        retrieval_query = query.strip()
    if retrieval_query and retrieval_query[-1] not in ".?!":
        retrieval_query += "."
    return f"{retrieval_query} {hyde_excerpt.strip()}"


def _retrieve_rag_context(
    history: list[dict],
    token_budget: int | None = None,
    *,
    stop_event=None,
) -> tuple[str, bool]:
    log("RAG", "Retrieving context from knowledge base...")
    try:
        from core.knowledge_base import list_indexed_documents, retrieve

        if not list_indexed_documents():
            log("RAG", "No documents indexed; skipping retrieval")
            return "", False

        query = ""
        for msg in reversed(history):
            if msg["role"] == "user":
                query = msg["content"]
                break
        if not query:
            log("RAG", "No user message found in history, skipping")
            return "", True

        log("RAG", f"Query: {query[:100]}...")
        results = retrieve(query)
        log("RAG", f"Retrieved {len(results)} chunks")
        if not results:
            return "", True

        _dump_chunks_for_debug(query, results)

        ctx_size = load_ctx_size(get_current_model_path()) or 8192
        sections, total = _pack_chunks_by_tokens(
            results,
            ctx_size,
            max_tokens=token_budget,
            stop_event=stop_event,
        )

        context = "\n\n---\n\n".join(sections)
        raw_length = len(context)
        try:
            compressed = _compress_rag_context(query, context, stop_event=stop_event)
            if compressed:
                context = compressed
                log("RAG", f"Compressed context: {raw_length} -> {len(context)} chars")
        except Exception as e:
            log("RAG", f"Compression failed, using raw context: {e}")

        used = f"{len(sections)}/{len(results)}"
        log("RAG", f"Context: {total} tokens, {len(context)} chars ({used} chunks used)")
        return context, True
    except Exception as e:
        log("RAG", f"Error during retrieval: {e}")
        return "", True
