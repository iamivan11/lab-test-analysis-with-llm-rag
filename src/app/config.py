import json
from pathlib import Path

APP_NAME = "Lab Test Analyzer"
APP_VERSION = "0.1.0"

DATA_DIR = Path.home() / "Library" / "Application Support" / "Lab Test Analyzer"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"


def load_model_path() -> str | None:
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text())
        path = data.get("model_path", "")
        if path and Path(path).exists():
            return path
    return None


def save_model_path(path: str) -> None:
    data = {}
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text())
    data["model_path"] = path
    SETTINGS_FILE.write_text(json.dumps(data))


def load_profile() -> dict:
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text())
        return data.get("profile", {})
    return {}


def save_profile(profile: dict) -> None:
    data = {}
    if SETTINGS_FILE.exists():
        data = json.loads(SETTINGS_FILE.read_text())
    data["profile"] = profile
    SETTINGS_FILE.write_text(json.dumps(data))
