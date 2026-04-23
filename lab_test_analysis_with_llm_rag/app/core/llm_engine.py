import json
import subprocess
import time
from collections.abc import Callable, Generator
from pathlib import Path

import httpx

from app.config import DEFAULT_MMPROJ_LOCAL, DEFAULT_MODEL_FILE, MODELS_DIR, load_ctx_size
from app.core.logger import log

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_BINARY = PROJECT_ROOT / "bin" / "llama-server"
RAG_DEBUG_DIR = PROJECT_ROOT / "tmp" / "rag-chunks"

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

_server_process: subprocess.Popen | None = None
_current_model_path: str | None = None


def get_current_model_path() -> str | None:
    """Path of the model currently served by llama-server, or None if not running."""
    if _server_process is None or _server_process.poll() is not None:
        return None
    return _current_model_path


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


def start_server(
    model_path: str,
    n_ctx: int = 32768,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    global _server_process, _current_model_path
    log("SERVER", f"start_server called, model={model_path}, n_ctx={n_ctx}")

    from app.core.llama_setup import ensure_server

    ensure_server(on_progress=on_progress)

    stop_server()
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
    # Enable vision only for the default model (mmproj is model-specific)
    if Path(model_path).name == DEFAULT_MODEL_FILE:
        mmproj_path = MODELS_DIR / DEFAULT_MMPROJ_LOCAL
        if mmproj_path.exists():
            cmd.extend(["--mmproj", str(mmproj_path)])
    log("SERVER", f"Launching llama-server: {' '.join(cmd)}")
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    log("SERVER", f"Process started, pid={_server_process.pid}")

    _wait_for_ready(on_progress=on_progress)
    _current_model_path = model_path
    log("SERVER", "Server is ready")


def _wait_for_ready(
    timeout: int = 120,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    deadline = time.time() + timeout

    while time.time() < deadline:
        if _server_process and _server_process.poll() is not None:
            stderr_output = ""
            if _server_process.stderr:
                stderr_output = _server_process.stderr.read()
            last_lines = "\n".join(stderr_output.strip().splitlines()[-5:])
            raise RuntimeError(f"llama-server process exited unexpectedly.\n{last_lines}")

        # Drain any available stderr lines for progress reporting
        if _server_process and _server_process.stderr:
            import select

            ready, _, _ = select.select([_server_process.stderr], [], [], 0.1)
            if ready:
                line = _server_process.stderr.readline().strip()
                if line and on_progress:
                    msg = _parse_progress(line)
                    if msg:
                        on_progress(msg)

        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=1)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return
        except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
            pass

    raise RuntimeError("llama-server did not become ready in time.")


def _parse_progress(line: str) -> str | None:
    line_lower = line.lower()
    for keyword, message in _PROGRESS_MAP:
        if keyword in line_lower:
            return message
    return None


def stop_server() -> None:
    global _server_process, _current_model_path
    status = "alive" if _server_process and _server_process.poll() is None else "none"
    log("SERVER", f"stop_server called, process={status}")
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None
    _current_model_path = None


def is_server_running() -> bool:
    if _server_process is None or _server_process.poll() is not None:
        return False
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=2)
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
    """Write retrieved chunks as .md files to tmp/rag-chunks/ for inspection."""
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


def rag_char_budget(ctx_size: int) -> int:
    """Character budget for RAG chunks: ~33% of context window at ~3 chars/token."""
    return int(ctx_size * 3 * 0.33)


def _retrieve_rag_context(history: list[dict]) -> tuple[str, bool]:
    """Retrieve relevant chunks for the last user message.

    Returns (context_text, has_indexed_docs). When no documents are indexed,
    returns ("", False) so the caller can inform the model rather than silently
    running without RAG.
    """
    log("RAG", "Retrieving context from knowledge base...")
    try:
        from app.core.knowledge_base import list_indexed_documents, retrieve

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
        results = retrieve(query, top_k=20)
        log("RAG", f"Retrieved {len(results)} chunks")
        if not results:
            return "", True

        _dump_chunks_for_debug(query, results)

        max_chars = rag_char_budget(load_ctx_size() or 8192)
        sections = []
        total_chars = 0
        for chunk in results:
            entry = f"[Source: {chunk['source']}]\n{chunk['text']}"
            if total_chars + len(entry) > max_chars and sections:
                break
            sections.append(entry)
            total_chars += len(entry) + 7  # account for "\n\n---\n\n" separator

        context = "\n\n---\n\n".join(sections)
        used = f"{len(sections)}/{len(results)}"
        log("RAG", f"Context length: {len(context)} chars ({used} chunks used)")
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
