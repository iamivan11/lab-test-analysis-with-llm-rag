import json
from collections.abc import Generator

import httpx

from config import load_ctx_size, load_max_tokens, load_model_meta, load_model_path
from core.http_client import post_with_retries
from core.llm_server import (
    SERVER_URL,
    get_current_model_path,
    is_server_running,
    recent_server_stderr,
    start_server,
    stop_server,
)
from core.logger import log
from core.prompts import ANSWER_DETAIL_INSTRUCTIONS, SYSTEM_PROMPT
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

_DEFAULT_CTX_SIZE = 8192
_DEFAULT_MAX_TOKENS = 4096
_CONTEXT_SAFETY_TOKENS = 512
_RECENT_HISTORY_MESSAGES = 4
_RAG_OPTIONAL_SHARE = 0.70
_MIN_OLDER_SUMMARY_TOKENS = 160
_CHARS_PER_TOKEN = 3

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


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _active_context_size() -> int:
    # Prefer the running server's model path: chat budgeting must match
    # the `--ctx-size` the server was actually started with, even if the
    # user has since selected a different default model in Settings.
    path = get_current_model_path() or load_model_path()
    n_ctx = load_ctx_size(path)
    if n_ctx:
        return n_ctx
    meta = load_model_meta(path) if path else None
    model_max = (meta or {}).get("context_length")
    return min(_DEFAULT_CTX_SIZE, model_max) if model_max else _DEFAULT_CTX_SIZE


def _input_token_budget(max_tokens: int | None) -> int:
    output_budget = max_tokens or load_max_tokens() or _DEFAULT_MAX_TOKENS
    return max(0, _active_context_size() - output_budget - _CONTEXT_SAFETY_TOKENS)


def _format_history_messages(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _trim_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    if _estimate_tokens(text) <= token_budget:
        return text
    max_chars = max(0, token_budget * _CHARS_PER_TOKEN)
    if max_chars <= 20:
        return ""
    return text[: max_chars - 15].rstrip() + "\n[truncated]"


def _fit_recent_history(prior: list[dict], token_budget: int) -> str:
    if token_budget <= 0:
        return ""

    candidates = prior[-_RECENT_HISTORY_MESSAGES:]
    selected: list[dict] = []
    for msg in reversed(candidates):
        candidate = [msg, *selected]
        text = _format_history_messages(candidate)
        if _estimate_tokens(text) <= token_budget:
            selected = candidate

    if selected:
        return _format_history_messages(selected)

    if not candidates:
        return ""
    last = candidates[-1]
    role = "User" if last.get("role") == "user" else "Assistant"
    return _trim_to_token_budget(f"{role}: {last.get('content', '')}", token_budget)


def _build_history_context(prior: list[dict], token_budget: int) -> str:
    if not prior or token_budget <= 0:
        return ""

    recent_text = _fit_recent_history(prior, token_budget)
    recent_tokens = _estimate_tokens(recent_text)
    remaining = max(0, token_budget - recent_tokens)

    older = prior[: max(0, len(prior) - _RECENT_HISTORY_MESSAGES)]
    sections = []
    if older and remaining >= _MIN_OLDER_SUMMARY_TOKENS:
        try:
            summary = summarize_history(older).strip()
        except Exception as e:
            log("LLM", f"History summary failed, omitting older turns: {e}")
            summary = ""
        if summary:
            summary_text = "Summary of older conversation:\n" + summary
            summary_text = _trim_to_token_budget(summary_text, remaining)
            if summary_text:
                sections.append(summary_text)

    if recent_text:
        sections.append("Recent conversation:\n" + recent_text)
    return "\n\n".join(sections)


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
    response = post_with_retries(
        f"{SERVER_URL}/v1/chat/completions",
        json={"model": "local", "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _build_system_content(
    history: list[dict],
    context: str,
    *,
    use_rag: bool,
    system_content: str,
) -> str:
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
    return system_content


def _build_budgeted_chat_system_content(
    history: list[dict],
    context: str,
    *,
    max_tokens: int | None,
    system_content: str,
) -> str:
    current_msg = history[-1]
    input_budget = _input_token_budget(max_tokens)

    if context:
        system_content += (
            "\n\n--- ATTACHED LAB REPORT ---\n"
            "The user has attached the following lab report for analysis.\n\n" + context
        )

    base_tokens = _estimate_tokens(system_content) + _estimate_tokens(
        current_msg.get("content", "")
    )
    optional_budget = max(0, input_budget - base_tokens)
    rag_budget = int(optional_budget * _RAG_OPTIONAL_SHARE)

    rag_context = ""
    has_docs = True
    if rag_budget > 0:
        rag_context, has_docs = _retrieve_rag_context(history, token_budget=rag_budget)
        rag_context = _trim_to_token_budget(rag_context, rag_budget)
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

    history_budget = max(
        0,
        input_budget
        - _estimate_tokens(system_content)
        - _estimate_tokens(current_msg.get("content", "")),
    )
    history_context = _build_history_context(history[:-1], history_budget)
    if history_context:
        system_content += (
            "\n\n--- CONVERSATION SO FAR (budgeted; documents take priority) ---\n"
            + history_context
        )

    log(
        "LLM",
        "Context budget: "
        f"input={input_budget}, optional={optional_budget}, "
        f"rag_budget={rag_budget}, history_budget={history_budget}, "
        f"rag_chars={len(rag_context)}, history_chars={len(history_context)}",
    )
    return system_content


def generate_stream(
    history: list[dict],
    context: str = "",
    stop_event=None,
    max_tokens: int | None = None,
    use_rag: bool = True,
    enable_thinking: bool | None = None,
    system_prompt_override: str | None = None,
    response_format: dict | None = None,
    answer_detail: str | None = None,
) -> Generator[tuple[str, str], None, None]:
    """Yields (kind, token) tuples where kind is 'thinking' or 'response'.

    history: full conversation so far as [{role, content}, ...] — the last
             entry must be the current user message.
    context: raw file content injected once before the first user turn.
    use_rag: when False, skip the ChromaDB retrieval entirely. The caller
             (e.g. the health-report generator) is responsible for stuffing
             whatever document content it wants into `context` directly.
    response_format: optional OpenAI-style response_format dict, e.g.
             {"type": "json_schema", "json_schema": {...}}. llama-server
             enforces the schema during generation, so the model cannot
             produce malformed JSON or surrounding chatter.
    """
    log(
        "LLM",
        f"generate_stream called, history={len(history)} msgs, "
        f"context={len(context)} chars, use_rag={use_rag}",
    )
    if not is_server_running():
        raise RuntimeError("LLM server is not running. Please load a model first.")

    system_content = system_prompt_override if system_prompt_override is not None else SYSTEM_PROMPT
    if system_prompt_override is None and answer_detail:
        instruction = ANSWER_DETAIL_INSTRUCTIONS.get(answer_detail)
        if instruction:
            system_content += "\n\nANSWER DETAIL:\n" + instruction
    current_msg = history[-1]  # always the current user message
    if use_rag and system_prompt_override is None:
        system_content = _build_budgeted_chat_system_content(
            history,
            context,
            max_tokens=max_tokens,
            system_content=system_content,
        )
    else:
        system_content = _build_system_content(
            history,
            context,
            use_rag=use_rag,
            system_content=system_content,
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
    if response_format is not None:
        request_body["response_format"] = response_format

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
            # llama-server's HTTP body for 5xx is generic ("Compute error.").
            # The actual cause (OOM, Metal kernel error, KV-cache failure,
            # etc.) is in stderr. Append the last few stderr lines so the
            # error surfaces something diagnostic.
            if response.status_code >= 500:
                tail = recent_server_stderr(n=15)
                if tail:
                    log("LLM", "Recent llama-server stderr (tail):")
                    for line in tail:
                        log("LLM_STDERR", line)
                    msg = msg + "\n\n(server stderr tail:\n" + "\n".join(tail) + ")"
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
