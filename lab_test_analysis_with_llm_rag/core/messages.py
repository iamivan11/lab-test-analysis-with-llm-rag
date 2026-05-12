"""Centralised user-facing error and status messages.

Every non-trivial error message is defined here and imported by the
module that surfaces it. Single source of truth: rewording a message
(or adjusting the recommended action) is a one-file edit.

Convention: each message is two sentences — what went wrong, then
the recommended action. Format strings use named placeholders so the
intent at the call site is self-documenting.
"""


def classify_by_substring(
    text: str,
    patterns: "tuple[tuple[tuple[str, ...], str], ...]",
) -> str | None:
    """Return the first message whose needles appear in `text`.

    Each pattern is a `(needles, message)` pair: `needles` is a tuple
    of lower-case substrings; if any of them appears in `text` (also
    lower-cased), `message` is returned. Used by both the llama-server
    stderr classifier and the chat-generation friendly-error mapper.
    """
    haystack = text.lower()
    for needles, message in patterns:
        if any(needle in haystack for needle in needles):
            return message
    return None

# ── Model server (load failures classified from llama.cpp stderr) ──
MODEL_NOT_ENOUGH_MEMORY = (
    "Not enough memory to load this model. "
    "Try a smaller model or lower Context Window in Settings."
)
MODEL_ARCH_UNSUPPORTED = (
    "This model architecture isn't supported. "
    "Pick a different model from the Models tab."
)
MODEL_FILE_INVALID = (
    "Model file is corrupted or invalid. "
    "Delete it from the Models tab and download it again."
)
MODEL_CONTEXT_TOO_LARGE = (
    "Context size too large for available memory. "
    "Lower Context Window in Settings and reload the model."
)
MODEL_LOAD_TIMEOUT = (
    "Model is taking too long to load. "
    "Reload the model or restart the app."
)

# ── Chat generation (in-flight LLM errors) ──
CHAT_TOKEN_LIMIT_NO_RESPONSE = (
    "Model used all available tokens on reasoning and produced no response. "
    "Try increasing Answer Detail in Settings or simplifying your question."
)
CHAT_EMPTY_RESPONSE = (
    "Model produced no response. Try sending the message again."
)
CHAT_INTERRUPTED_ON_CLOSE = (
    "Generation was interrupted because the app was closing. "
    "Send the message again after reopening."
)
CHAT_CONTEXT_TOO_LARGE = (
    "This request does not fit in the current context window. "
    "Increase Context Window in Settings or reduce the attached/history content."
)
CHAT_LOCAL_MODEL_STOPPED = (
    "The local model stopped during generation. Reload the model and try again."
)
CHAT_LOCAL_MODEL_CONNECTION = (
    "Connection to the local model was interrupted during generation. "
    "Try again; if it repeats, reload the model."
)
CHAT_LOCAL_MODEL_MEMORY = (
    "The local model ran out of memory or failed during generation. "
    "Lower Context Window or Answer Detail, or load a smaller model."
)
CHAT_COMPRESSION_FAILED = (
    "Could not compress history. Please start a new conversation."
)

# ── Documents (indexing, deletion) ──
INDEXING_FAILED = (
    "Indexing failed: {error}. Verify the document is valid and try again."
)
REINDEX_FAILED_BATCH = (
    "Reindex failed for {count} document(s). "
    "Re-upload them from the Medical Documents tab."
)
REINDEX_FAILED_SINGLE = (
    "Reindex failed: {error}. "
    "Re-upload the document from the Medical Documents tab."
)
# Used for both the "cancelled cleanup" path and the "failed-files
# cleanup" path — same semantic ("file left behind on disk"), same
# recovery, one message.
DOCS_COULD_NOT_REMOVE_BATCH = (
    "Couldn't remove {count} file(s) from disk: {first}. "
    "Delete them manually from the Medical Documents tab."
)
DOCS_COULD_NOT_DELETE_SINGLE = (
    "Couldn't delete {name}: {err}. "
    "Check file permissions and try again."
)

# ── Onboarding ──
ONBOARDING_DOWNLOAD_FAILED = (
    "Download failed: {error}. "
    "Check your network and click Back then Continue to retry."
)

# ── Trends / biomarker extraction ──
EXTRACTION_FAILED = (
    "Extraction failed: {msg}. Verify documents are valid and try again."
)

# ── Settings / user data ──
CLEAR_USER_DATA_FAILED = (
    "Failed to clear user data: {error}. Restart the app and try again."
)

# ── Security (raised as SecurityError, displayed in Settings) ──
_SECURITY_REPAIR_ACTION = (
    "Disable password protection in Settings and re-enable it."
)
SECURITY_FORMAT_UNSUPPORTED = (
    f"Encrypted file format is unsupported. {_SECURITY_REPAIR_ACTION}"
)
SECURITY_METADATA_UNREADABLE = (
    f"Cannot read security metadata: {{error}}. {_SECURITY_REPAIR_ACTION}"
)
SECURITY_METADATA_CORRUPTED = (
    f"Security metadata is corrupted. {_SECURITY_REPAIR_ACTION}"
)
SECURITY_METADATA_INCOMPLETE = (
    f"Security metadata is incomplete. {_SECURITY_REPAIR_ACTION}"
)

# ── Platform compatibility ──
UNSUPPORTED_MACOS = (
    "This app requires macOS {minimum} or newer; you have {current}. "
    "Upgrade macOS to continue."
)
