"""Filesystem paths the app reads / writes.

All app data lives under DATA_DIR (macOS Application Support). The
HF_HOME environment variable is bent toward CACHE_DIR here to keep the
HuggingFace cache inside the app's writable space — must happen before
any module that touches `huggingface_hub` or `sentence_transformers`
imports, which is why this lives under `config` (imported first by
main.py).
"""

import os
import shutil
import sys
from pathlib import Path

from core.macos_compat import application_support_dir


APP_NAME = "Lab Analyzer"
APP_VERSION = "0.1.0"
MIN_MACOS_VERSION = "13.0"


def _project_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def _assets_root(project_root: Path) -> Path:
    if getattr(sys, "frozen", False):
        resources_root = project_root.parent / "Resources"
        if resources_root.exists():
            return resources_root
    return project_root


PROJECT_ROOT = _project_root()
ASSETS_ROOT = _assets_root(PROJECT_ROOT)
PROJECT_TMP_DIR = PROJECT_ROOT / "tmp"
ICONS_DIR = ASSETS_ROOT / "assets" / "icons"

LEGACY_DATA_DIR = application_support_dir("Lab Test Analyzer")
DATA_DIR = application_support_dir(APP_NAME)


def _migrate_legacy_data_dir() -> None:
    if not LEGACY_DATA_DIR.exists() or DATA_DIR.exists():
        return
    try:
        shutil.move(str(LEGACY_DATA_DIR), str(DATA_DIR))
    except OSError as e:
        # Surface to stderr (logger may not be configured yet at import
        # time): a silent failure leaves the user with two data dirs and
        # no idea their old chats/profile didn't carry over.
        print(
            f"WARNING: Failed to migrate legacy data dir from {LEGACY_DATA_DIR} "
            f"to {DATA_DIR}: {e}. Existing data may be inaccessible until "
            f"resolved manually.",
            file=sys.stderr,
        )


_migrate_legacy_data_dir()

MODELS_DIR = DATA_DIR / "models"
DOCS_DIR = DATA_DIR / "documents"


def list_uploaded_doc_paths() -> list[Path]:
    """Return uploaded document paths, sorted by name (case-insensitive).

    Single source of truth for "what counts as an uploaded document":
    a regular file in DOCS_DIR whose name doesn't start with a dot.
    """
    if not DOCS_DIR.exists():
        return []
    return sorted(
        (f for f in DOCS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")),
        key=lambda f: f.name.lower(),
    )


SETTINGS_FILE = DATA_DIR / "settings.json"
PROFILE_FILE = DATA_DIR / "profile.json"
APP_LOG_FILE = DATA_DIR / "logs" / "app.log"
CACHE_DIR = DATA_DIR / "cache"
PARSING_STAGING_DIR = CACHE_DIR / "parsing_staging"
PARSING_OUTPUT_DIR = DATA_DIR / "parsing_output"
FILTERING_OUTPUT_DIR = DATA_DIR / "filtering_output"
REPORTS_DIR = DATA_DIR / "reports"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
RAG_DEBUG_ENABLED = False
RAG_DEBUG_DIR = DATA_DIR / "debug" / "rug_chunks"

# Redirect huggingface_hub / sentence_transformers to a writable cache
# inside our app-data dir. Default ~/.cache/huggingface can fail with
# PermissionError when the .app runs from /Applications under macOS
# Gatekeeper translocation, leaving embedding-model downloads broken.
# Must be set BEFORE huggingface_hub or sentence_transformers is imported,
# so this module — imported first by main.py — is the right place.
#
# In a frozen .app we override unconditionally: a stale HF_HOME exported
# in the user's shell rc would otherwise leak into the GUI process and
# point us back at the unwritable default. In dev mode we respect a
# pre-set value so contributors can share caches across projects.
if getattr(sys, "frozen", False):
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
else:
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
# We never authenticate with HF and never opt into telemetry; explicit
# values keep the lib from probing optional code paths that would touch
# unrelated cache directories.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _ensure_private_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    except OSError:
        pass


for _dir in (
    DATA_DIR,
    MODELS_DIR,
    DOCS_DIR,
    APP_LOG_FILE.parent,
    CACHE_DIR,
    PARSING_STAGING_DIR,
    PARSING_OUTPUT_DIR,
    FILTERING_OUTPUT_DIR,
    REPORTS_DIR,
    HF_CACHE_DIR,
):
    _ensure_private_dir(_dir)


def _migrate_legacy_runtime_dir(old_path: Path, new_path: Path) -> None:
    """Copy legacy project-tmp runtime files into app data without deleting originals."""
    try:
        if not old_path.exists() or (new_path.exists() and any(new_path.iterdir())):
            return
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
    except OSError as e:
        print(
            f"WARNING: Failed to migrate legacy runtime dir from {old_path} "
            f"to {new_path}: {e}",
            file=sys.stderr,
        )


_migrate_legacy_runtime_dir(PROJECT_TMP_DIR / "parsing_output", PARSING_OUTPUT_DIR)
_migrate_legacy_runtime_dir(PROJECT_TMP_DIR / "filtering_output", FILTERING_OUTPUT_DIR)
