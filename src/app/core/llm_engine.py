import json
import subprocess
import time
from collections.abc import Callable, Generator
from pathlib import Path

import httpx

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_BINARY = Path(__file__).parent.parent.parent.parent / "bin" / "llama-server"

SYSTEM_PROMPT = """\
You are a medical laboratory test analyst assistant. Your role is to:
1. Analyze lab test results provided by the user
2. Identify values outside normal reference ranges
3. Explain what each indicator means in plain language
4. Suggest possible implications and recommend follow-up actions

Important disclaimers you must always include:
- You are NOT a doctor and this is NOT a medical diagnosis
- Always recommend consulting a healthcare professional
- Do not prescribe treatments or medications
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


def _kill_port() -> None:
    """Kill any process already holding our port (e.g. from a previous crash)."""
    import subprocess as sp
    try:
        result = sp.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True, text=True
        )
        for pid in result.stdout.strip().splitlines():
            sp.run(["kill", "-9", pid], capture_output=True)
    except Exception:
        pass


def start_server(
    model_path: str,
    n_ctx: int = 32768,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    global _server_process
    stop_server()
    _kill_port()

    cmd = [
        str(SERVER_BINARY),
        "--model", model_path,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--ctx-size", str(n_ctx),
        "-ngl", "99",
        "--no-webui",
    ]
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    _wait_for_ready(on_progress=on_progress)


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
            raise RuntimeError(
                f"llama-server process exited unexpectedly.\n{last_lines}"
            )

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
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None


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
    conversation = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in history
    )
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
    thinking: bool = True,
    stop_event=None,
) -> Generator[tuple[str, str], None, None]:
    """Yields (kind, token) tuples where kind is 'thinking' or 'response'.

    history: full conversation so far as [{role, content}, ...] — the last
             entry must be the current user message.
    context: raw file content injected once before the first user turn.
    """
    if not is_server_running():
        raise RuntimeError("LLM server is not running. Please load a model first.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "user",
            "content": f"Here is the lab report data:\n\n{context}",
        })
        messages.append({
            "role": "assistant",
            "content": "I've received the lab report data. I'll analyze it along with your question.",
        })

    messages.extend(history)

    request_body: dict = {"model": "local", "messages": messages, "stream": True}
    if not thinking:
        # Pass Qwen3 template variable directly to disable the reasoning phase
        request_body["chat_template_kwargs"] = {"enable_thinking": False}

    with httpx.stream(
        "POST",
        f"{SERVER_URL}/v1/chat/completions",
        json=request_body,
        timeout=None,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if stop_event and stop_event.is_set():
                return
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                return
            chunk = json.loads(data)
            if choices := chunk.get("choices"):
                delta = choices[0].get("delta", {})
                if token := delta.get("reasoning_content", ""):
                    yield ("thinking", token)
                if token := delta.get("content", ""):
                    yield ("response", token)
