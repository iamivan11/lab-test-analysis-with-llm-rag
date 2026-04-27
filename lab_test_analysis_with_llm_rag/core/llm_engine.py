import contextlib
import json
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from pathlib import Path

import httpx

from config import (
    DEFAULT_MMPROJ_LOCAL,
    DEFAULT_MODEL_FILE,
    LLM_HEALTH_TIMEOUT_SECONDS,
    LLM_HYDE_TIMEOUT_SECONDS,
    LLM_QUERY_MODIFICATION_TIMEOUT_SECONDS,
    LLM_RAG_COMPRESSION_TIMEOUT_SECONDS,
    LLM_SERVER_READY_TIMEOUT_SECONDS,
    LLM_TOKENIZE_TIMEOUT_SECONDS,
    MODELS_DIR,
    PROJECT_ROOT,
    RAG_DEBUG_DIR,
    SERVER_HOST,
    SERVER_PORT,
    load_ctx_size,
)
from core.logger import log

SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_BINARY = PROJECT_ROOT / "bin" / "llama-server"

SYSTEM_PROMPT = """\
You are a clinical lab test analyst assistant. You help patients understand their \
laboratory test results by comparing values, identifying trends, explaining findings and more.

You have access to the patient's historical lab data through a knowledge base. When \
historical data is provided below, use it directly — do not claim you lack access.

SCOPE: You ONLY answer questions related to health, medicine, lab tests, \
medical conditions, and biology. If the user asks about anything outside \
this scope, reply: "I can only help with health and medical \
related questions."

RESPONSE RULES (follow strictly):
- Be concise and direct. Answer the question asked — no filler, no preamble.
- Do NOT second-guess or re-analyze your previous responses. Treat your earlier \
answers as correct and build on them only if they are relevant. Focus only on the new question.
- Professional clinical tone. No emojis, no exclamation marks, no casual language.
- Never use LaTeX notation ($, \\text, \\times, etc.). Write math as plain text.
- Use markdown tables when comparing values across dates. Every row MUST start \
and end with |. Always leave a blank line before and after the table. Example:

| Date | Value | Reference | Status |
| --- | --- | --- | --- |
| 2025-12-03 | 27 | 15-200 | Normal |

- Do NOT use bullet points or numbered lists. Write short paragraphs instead.
- Separate sections with blank lines, not list markers.
- Do not prescribe treatments or medications.
"""
RAG_COMPRESSION_PROMPT = """\
You compress retrieved medical context before it is sent to the answering model.

Keep the essential facts exactly:
- report dates
- report types
- test names
- results
- flags
- units
- reference ranges
- clinically relevant findings and conclusions

Remove only unnecessary detail, repetition, and boilerplate.
Do not invent, rewrite, interpret, or normalize facts.
Return only the compressed medical context.
"""
RAG_QUERY_MODIFICATION_PROMPT = """\
You improve a medical retrieval query before vector search.

Keep the user's original question wording and structure.
Do not replace the question with a keyword list.
Add concise synonyms, abbreviations, and spelling variants only in parentheses immediately
after the original key words.
Example:
"What was my sperm concentration in 2023?"
→ "What was my sperm concentration (sperm count, sperm density) in 2023?"
Do not answer the question.
Do not add unrelated medical concepts.
Return one plain-text retrieval query only.
"""
RAG_QUERY_REPHRASE_PROMPT = """\
You rewrite a medical retrieval query only if it is badly formulated.

Fix grammar, missing words, awkward phrasing, and unclear wording.
Keep the same intent, medical meaning, dates, test names, and constraints.
Do not add synonyms, explanations, or new medical concepts.
Do not answer the question.
Return one concise plain-text question only.
"""
RAG_HYDE_PROMPT = """\
You create a short hypothetical direct answer for retrieval.

Write 1-2 short plain sentences that imitate a direct answer to the user's question.
Use natural language only.
Do not use Markdown, tables, bullets, separators, citations, labels, or special symbols.
Include important terms, synonyms, report types, dates, and units when relevant.
Focus only on likely report names, dates, test names, units, and comparison terms.
Avoid background explanation, definitions, filler, cautious phrasing, and generic medical advice.
Do not invent exact numeric results or conclusions.
Never refuse.
Never say that data, reports, or values are unavailable.
If exact values are unknown, write a generic answer shape that mentions the likely
report type, date, test name, units, and related synonyms without numeric values.
Return only the hypothetical answer text.
"""

# Log line substrings → human-readable progress messages
_PROGRESS_MAP = [
    ("loading model", "Reading model file..."),
    ("loaded meta data", "Parsing model metadata..."),
    ("load_tensors: loading model tensors", "Loading model tensors..."),
    ("offloading output layer to GPU", "Offloading layers to GPU..."),
    ("offloaded", "Offloaded layers to GPU"),
    ("constructing llama_context", "Initializing context..."),
    ("warming up the model", "Warming up (first run takes a moment)..."),
    ("model loaded", "Model loaded, starting server..."),
    ("server is listening", "Server ready"),
]

_STDERR_RING_SIZE = 500


def _kill_port() -> None:
    """Kill any process already holding our port (e.g. from a previous crash)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True,
            text=True,
        )
        for pid in result.stdout.strip().splitlines():
            subprocess.run(["kill", "-9", pid], capture_output=True)
    except Exception:
        pass


def _parse_progress(line: str) -> str | None:
    line_lower = line.lower()
    for keyword, message in _PROGRESS_MAP:
        if keyword in line_lower:
            return message
    return None


class LlamaServer:
    """Manages the llama-server subprocess lifecycle.

    All mutating operations are serialized through a single lock so overlapping
    start/stop calls cannot leak subprocesses. A background thread drains stderr
    into a ring buffer; on load failure the full buffer is logged for diagnosis.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._model_path: str | None = None
        self._stderr_buffer: deque[str] = deque(maxlen=_STDERR_RING_SIZE)
        self._stderr_thread: threading.Thread | None = None
        self._progress_cb: Callable[[str], None] | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def current_model_path(self) -> str | None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return None
            return self._model_path

    def start(
        self,
        model_path: str,
        n_ctx: int = 32768,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        with self._lock:
            log("SERVER", f"start called, model={model_path}, n_ctx={n_ctx}")

            from core.llama_setup import ensure_server

            ensure_server(on_progress=on_progress)

            self._stop_locked()
            _kill_port()

            cmd = [
                str(SERVER_BINARY),
                "--model",
                model_path,
                "--host",
                SERVER_HOST,
                "--port",
                str(SERVER_PORT),
                "--ctx-size",
                str(n_ctx),
                "-ngl",
                "99",
                "--no-webui",
                "--parallel",
                "2",
            ]
            if Path(model_path).name == DEFAULT_MODEL_FILE:
                mmproj_path = MODELS_DIR / DEFAULT_MMPROJ_LOCAL
                if mmproj_path.exists():
                    cmd.extend(["--mmproj", str(mmproj_path)])
            log("SERVER", f"Launching llama-server: {' '.join(cmd)}")

            self._stderr_buffer.clear()
            self._progress_cb = on_progress
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            log("SERVER", f"Process started, pid={self._process.pid}")

            self._stderr_thread = threading.Thread(
                target=self._pump_stderr, daemon=True, name="llama-stderr-pump"
            )
            self._stderr_thread.start()

            try:
                self._wait_for_ready()
            except Exception:
                self._dump_stderr_buffer()
                self._stop_locked()
                raise
            self._model_path = model_path
            log("SERVER", "Server is ready")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        """Must be called with self._lock held."""
        status = "alive" if self._process and self._process.poll() is None else "none"
        log("SERVER", f"stop called, process={status}")
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._stderr_thread:
            self._stderr_thread.join(timeout=2)
        self._process = None
        self._model_path = None
        self._stderr_thread = None
        self._progress_cb = None

    def _pump_stderr(self) -> None:
        """Background thread: push every stderr line into the ring buffer."""
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        with contextlib.suppress(Exception):
            for raw in proc.stderr:
                line = raw.rstrip()
                if not line:
                    continue
                self._stderr_buffer.append(line)
                cb = self._progress_cb
                if cb is not None:
                    msg = _parse_progress(line)
                    if msg:
                        with contextlib.suppress(Exception):
                            cb(msg)

    def _wait_for_ready(self, timeout: int = LLM_SERVER_READY_TIMEOUT_SECONDS) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self._process
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("llama-server process exited unexpectedly")
            try:
                r = httpx.get(f"{SERVER_URL}/health", timeout=1)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
                pass
            time.sleep(0.2)
        raise RuntimeError("llama-server did not become ready in time.")

    def _dump_stderr_buffer(self) -> None:
        """Write the full ring buffer to the log for post-mortem diagnosis."""
        if not self._stderr_buffer:
            return
        lines = list(self._stderr_buffer)
        log("LLM_STDERR", f"--- server stderr ({len(lines)} lines) ---")
        for line in lines:
            log("LLM_STDERR", line)
        log("LLM_STDERR", "--- end ---")


_server = LlamaServer()


def start_server(
    model_path: str,
    n_ctx: int = 32768,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    _server.start(model_path, n_ctx=n_ctx, on_progress=on_progress)


def stop_server() -> None:
    _server.stop()


def get_current_model_path() -> str | None:
    """Path of the model currently served by llama-server, or None if not running."""
    return _server.current_model_path()


def is_server_running() -> bool:
    if not _server.is_running():
        return False
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=LLM_HEALTH_TIMEOUT_SECONDS)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
        return False


def summarize_history(history: list[dict]) -> str:
    """Send a blocking request to compress conversation history into ~800 tokens."""
    log("LLM", f"summarize_history called, {len(history)} messages")
    conversation = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. Summarize the conversation below "
                "in 800 tokens or fewer, preserving all key medical data, lab values, "
                "findings, user preferences, and conclusions reached."
            ),
        },
        {"role": "user", "content": f"Summarize this conversation:\n\n{conversation}"},
    ]
    response = httpx.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={"model": "local", "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _dump_chunks_for_debug(query: str, chunks: list[dict]) -> None:
    """Write retrieved chunks as .md files to tmp/rug_chunks/ for inspection."""
    try:
        if RAG_DEBUG_DIR.exists():
            for old in RAG_DEBUG_DIR.iterdir():
                if old.is_file():
                    old.unlink()
        RAG_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        width = max(2, len(str(len(chunks))))
        for i, chunk in enumerate(chunks, start=1):
            path = RAG_DEBUG_DIR / f"chunk_{i:0{width}d}.md"
            path.write_text(
                f"# Query\n\n{query}\n\n"
                f"# Source\n\n{chunk['source']}\n\n"
                f"# Chunk\n\n{chunk['text']}\n"
            )
    except OSError as e:
        log("RAG", f"Failed to dump chunks: {e}")


def rag_token_budget(ctx_size: int) -> int:
    """Token budget for RAG chunks: 33% of the model's context window."""
    return int(ctx_size * 0.33)


def count_tokens(text: str, server_url: str = SERVER_URL) -> int:
    """Count tokens using the loaded model's own tokenizer (llama-server /tokenize)."""
    r = httpx.post(
        f"{server_url}/tokenize",
        json={"content": text},
        timeout=LLM_TOKENIZE_TIMEOUT_SECONDS,
    )
    r.raise_for_status()
    return len(r.json().get("tokens", []))


def _rag_char_budget_fallback(ctx_size: int) -> int:
    """Fallback heuristic when /tokenize is unavailable: ~33% of ctx at ~3 chars/token."""
    return int(ctx_size * 3 * 0.33)


def _format_chunk_for_context(chunk: dict) -> str:
    """Format chunk text with all metadata available to the answer/compression model."""
    metadata = [
        f"Source: {chunk.get('source', '')}",
        f"Report date: {chunk.get('report_date', '')}",
        f"Report type: {chunk.get('report_type', '')}",
        f"Chunk index: {chunk.get('chunk_index', 0)}",
    ]
    return f"[{' | '.join(metadata)}]\n{chunk['text']}"


def _pack_chunks_by_tokens(
    chunks: list[dict], ctx_size: int
) -> tuple[list[str], int]:
    """Pack chunks into entries using the model's tokenizer.

    Falls back to a char-based heuristic if /tokenize is unreachable.
    Returns (entries, total_tokens_or_chars).
    """
    max_tokens = rag_token_budget(ctx_size)
    separator = "\n\n---\n\n"
    try:
        sep_tokens = count_tokens(separator)
    except Exception as e:
        log("RAG", f"Tokenizer unavailable ({e}); falling back to char budget")
        return _pack_chunks_by_chars(chunks, _rag_char_budget_fallback(ctx_size))

    entries: list[str] = []
    total = 0
    for chunk in chunks:
        entry = _format_chunk_for_context(chunk)
        try:
            entry_tokens = count_tokens(entry)
        except Exception as e:
            log("RAG", f"Tokenizer failed mid-pack ({e}); falling back to char budget")
            return _pack_chunks_by_chars(chunks, _rag_char_budget_fallback(ctx_size))
        if entries and total + entry_tokens + sep_tokens > max_tokens:
            break
        entries.append(entry)
        total += entry_tokens + (sep_tokens if len(entries) > 1 else 0)
    return entries, total


def _pack_chunks_by_chars(chunks: list[dict], max_chars: int) -> tuple[list[str], int]:
    """Legacy char-budget packing, kept as fallback."""
    entries: list[str] = []
    total = 0
    for chunk in chunks:
        entry = _format_chunk_for_context(chunk)
        if entries and total + len(entry) + 7 > max_chars:
            break
        entries.append(entry)
        total += len(entry) + (7 if len(entries) > 1 else 0)
    return entries, total


def _compress_rag_context(query: str, context: str) -> str:
    """Compress retrieved context while preserving key medical facts."""
    if not context.strip():
        return context

    response = httpx.post(
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
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    compressed = (message.get("content") or "").strip()
    if not compressed:
        raise RuntimeError("RAG compression returned empty text")
    return compressed


def _modify_retrieval_query(query: str) -> str:
    """Expand the retrieval query with synonyms while preserving the user's intent."""
    if not query.strip():
        return query

    response = httpx.post(
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
    """Rephrase a poorly formulated query while preserving the user's intent."""
    if not query.strip():
        return query

    response = httpx.post(
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
    """Generate a hypothetical document excerpt for HyDE retrieval."""
    if not query.strip():
        return query

    response = httpx.post(
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
    """Combine the retrieval query and HyDE excerpt for vector retrieval."""
    if expanded_query and expanded_query.strip() != query.strip():
        retrieval_query = expanded_query.strip()
    else:
        retrieval_query = query.strip()
    if retrieval_query and retrieval_query[-1] not in ".?!":
        retrieval_query += "."
    return f"{retrieval_query} {hyde_excerpt.strip()}"


def _retrieve_rag_context(history: list[dict]) -> tuple[str, bool]:
    """Retrieve relevant chunks for the last user message.

    Returns (context_text, has_indexed_docs). When no documents are indexed,
    returns ("", False) so the caller can inform the model rather than silently
    running without RAG.
    """
    log("RAG", "Retrieving context from knowledge base...")
    try:
        from core.knowledge_base import list_indexed_documents, retrieve

        if not list_indexed_documents():
            log("RAG", "No documents indexed; skipping retrieval")
            return "", False

        # Use the last user message as the retrieval query
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

        ctx_size = load_ctx_size() or 8192
        sections, total = _pack_chunks_by_tokens(results, ctx_size)

        context = "\n\n---\n\n".join(sections)
        raw_length = len(context)
        try:
            compressed = _compress_rag_context(query, context)
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


def generate_stream(
    history: list[dict],
    context: str = "",
    stop_event=None,
    max_tokens: int | None = None,
) -> Generator[tuple[str, str], None, None]:
    """Yields (kind, token) tuples where kind is 'thinking' or 'response'.

    history: full conversation so far as [{role, content}, ...] — the last
             entry must be the current user message.
    context: raw file content injected once before the first user turn.
    """
    log(
        "LLM",
        f"generate_stream called, history={len(history)} msgs, context={len(context)} chars",
    )
    if not is_server_running():
        raise RuntimeError("LLM server is not running. Please load a model first.")

    # Build system prompt with RAG context baked in
    system_content = SYSTEM_PROMPT

    rag_context, has_docs = _retrieve_rag_context(history)
    if rag_context:
        system_content += (
            "\n\n--- PATIENT'S HISTORICAL LAB DATA ---\n"
            "Below is relevant data retrieved from the patient's previous lab test records. "
            "Use this data to answer the user's questions, compare values, and identify trends.\n\n"
            + rag_context
        )
    elif not has_docs:
        system_content += (
            "\n\n--- KNOWLEDGE BASE STATUS ---\n"
            "The user has not uploaded any lab test documents yet. If their question "
            "requires historical lab data, tell them to upload documents via the "
            "Documents window in the app, then retry the question."
        )

    if context:
        system_content += (
            "\n\n--- ATTACHED LAB REPORT ---\n"
            "The user has attached the following lab report for analysis.\n\n" + context
        )

    # Embed prior exchanges as reference context in the system prompt,
    # so thinking models don't re-analyze them as active conversation turns.
    current_msg = history[-1]  # always the current user message
    prior = history[:-1]
    if prior:
        lines = []
        for msg in prior:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        system_content += (
            "\n\n--- CONVERSATION SO FAR (settled context, do not re-analyze) ---\n"
            + "\n".join(lines)
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": current_msg["content"]},
    ]

    total_chars = sum(len(m["content"]) for m in messages)
    log("LLM", f"Sending {len(messages)} messages, {total_chars} total chars to server")
    for i, m in enumerate(messages):
        log("LLM", f"  msg[{i}] role={m['role']} len={len(m['content'])}")

    request_body: dict = {"model": "local", "messages": messages, "stream": True}
    if max_tokens is not None:
        request_body["max_tokens"] = max_tokens

    log("LLM", "Opening streaming connection to server...")
    with httpx.stream(
        "POST",
        f"{SERVER_URL}/v1/chat/completions",
        json=request_body,
        timeout=None,
    ) as response:
        response.raise_for_status()
        log("LLM", f"Stream opened, status={response.status_code}")
        for line in response.iter_lines():
            if stop_event and stop_event.is_set():
                return
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                return
            chunk = json.loads(data)
            if choices := chunk.get("choices"):
                delta = choices[0].get("delta", {})
                if token := delta.get("reasoning_content", ""):
                    yield ("thinking", token)
                if token := delta.get("content", ""):
                    yield ("response", token)
