import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import DATA_DIR
from app.core.logger import log

CHATS_DIR = DATA_DIR / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def new_chat() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "history": [],
    }


def title_from_first_message(message: str) -> str:
    words = message.split()[:5]
    return " ".join(words) if words else "New Chat"


def list_chats() -> list[dict]:
    """Returns chat metadata (no history) sorted by updated_at descending."""
    chats = []
    for path in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            chats.append({k: v for k, v in data.items() if k != "history"})
        except json.JSONDecodeError:
            log("CHAT", f"Corrupted chat file, skipping: {path.name}")
        except OSError as e:
            log("CHAT", f"Error reading {path.name}: {e}")
    return sorted(chats, key=lambda c: c.get("updated_at", ""), reverse=True)


def load_chat(chat_id: str) -> dict | None:
    path = _chat_path(chat_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log("CHAT", f"Corrupted chat file: {path.name}")
        except OSError as e:
            log("CHAT", f"Error reading {path.name}: {e}")
    return None


def save_chat(chat: dict) -> None:
    chat["updated_at"] = datetime.now(UTC).isoformat()
    _chat_path(chat["id"]).write_text(json.dumps(chat, ensure_ascii=False, indent=2))


def delete_chat(chat_id: str) -> None:
    path = _chat_path(chat_id)
    if path.exists():
        path.unlink()


def rename_chat(chat_id: str, title: str) -> None:
    chat = load_chat(chat_id)
    if chat:
        chat["title"] = title
        save_chat(chat)
