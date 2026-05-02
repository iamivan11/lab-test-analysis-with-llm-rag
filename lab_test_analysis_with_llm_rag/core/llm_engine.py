import json
from collections.abc import Generator

import httpx

from core.llm_server import (
    SERVER_URL,
    get_current_model_path,
    is_server_running,
    start_server,
    stop_server,
)
from core.logger import log
from core.prompts import SYSTEM_PROMPT
from core.rag_context import (
    _build_hyde_retrieval_query,
    _compress_rag_context,
    _format_chunk_for_context,
    _generate_hyde_query,
    _modify_retrieval_query,
    _pack_chunks_by_chars,
    _pack_chunks_by_tokens,
    _rephrase_retrieval_query,
    _retrieve_rag_context,
    count_tokens,
    rag_token_budget,
)

__all__ = [
    "SERVER_URL",
    "_build_hyde_retrieval_query",
    "_compress_rag_context",
    "_format_chunk_for_context",
    "_generate_hyde_query",
    "_modify_retrieval_query",
    "_pack_chunks_by_chars",
    "_pack_chunks_by_tokens",
    "_rephrase_retrieval_query",
    "_retrieve_rag_context",
    "count_tokens",
    "generate_stream",
    "get_current_model_path",
    "is_server_running",
    "rag_token_budget",
    "start_server",
    "stop_server",
    "summarize_history",
]


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


def generate_stream(
    history: list[dict],
    context: str = "",
    stop_event=None,
    max_tokens: int | None = None,
    use_rag: bool = True,
    enable_thinking: bool | None = None,
    system_prompt_override: str | None = None,
) -> Generator[tuple[str, str], None, None]:
    """Yields (kind, token) tuples where kind is 'thinking' or 'response'.

    history: full conversation so far as [{role, content}, ...] — the last
             entry must be the current user message.
    context: raw file content injected once before the first user turn.
    use_rag: when False, skip the ChromaDB retrieval entirely. The caller
             (e.g. the health-report generator) is responsible for stuffing
             whatever document content it wants into `context` directly.
    """
    log(
        "LLM",
        f"generate_stream called, history={len(history)} msgs, "
        f"context={len(context)} chars, use_rag={use_rag}",
    )
    if not is_server_running():
        raise RuntimeError("LLM server is not running. Please load a model first.")

    # Build system prompt with RAG context baked in (unless disabled).
    system_content = system_prompt_override if system_prompt_override is not None else SYSTEM_PROMPT

    if use_rag:
        rag_context, has_docs = _retrieve_rag_context(history)
        if rag_context:
            system_content += (
                "\n\n--- PATIENT'S HISTORICAL LAB DATA ---\n"
                "Below is relevant data retrieved from the patient's previous lab test records. "
                "Use this data to answer the user's questions, compare values, "
                "and identify trends.\n\n"
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
    if enable_thinking is not None:
        # Qwen 3's chat template honors `enable_thinking` to gate its
        # internal reasoning. llama-server's OpenAI-compat endpoint forwards
        # `chat_template_kwargs` to the template at apply-time.
        request_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    log("LLM", "Opening streaming connection to server...")
    with httpx.stream(
        "POST",
        f"{SERVER_URL}/v1/chat/completions",
        json=request_body,
        timeout=None,
    ) as response:
        if response.status_code >= 400:
            # Read the body so the caller sees the real reason instead of a
            # generic "400 Bad Request". llama-server emits a JSON body with
            # "error: { message: ... }" — surface that text.
            body = b"".join(response.iter_bytes()).decode("utf-8", errors="replace")
            log("LLM", f"Server returned {response.status_code}: {body[:500]}")
            try:
                err = json.loads(body).get("error", {})
                msg = err.get("message") or err.get("type") or body
            except (json.JSONDecodeError, AttributeError):
                msg = body or f"HTTP {response.status_code}"
            # Heuristic: turn the common context-overflow message into
            # something actionable for the user.
            if any(k in msg.lower() for k in ("context", "n_ctx", "tokens", "exceeds")):
                msg = (
                    "The documents are too large for the current context "
                    "window. Increase Context Window in Settings, or remove "
                    "some documents.\n\n(server said: " + msg + ")"
                )
            raise RuntimeError(msg)
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
