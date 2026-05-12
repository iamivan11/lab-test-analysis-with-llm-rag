"""App configuration: paths, model catalog, persisted settings, runtime constants.

Re-exports every public name the rest of the codebase imports so the
existing `from config import X` call sites keep working after the
file was split into a package. Submodule breakdown:

- paths.py:    filesystem layout (DATA_DIR, MODELS_DIR, DOCS_DIR, …)
- models.py:   APPROVED_MODELS catalog + helpers + SYSTEM_MODEL aliases
- settings.py: AppSettings model + load/save/accessor helpers

Runtime tunables (LLM/parser timeouts, embedder configs, etc.) and the
small `format_size` utility live here at package level — they have no
dependencies and would not benefit from another submodule.
"""

# Re-exports from submodules ──────────────────────────────────────────
from config.models import (
    ALLOW_ALL_MODEL_DOWNLOADS,
    ALLOWED_DOWNLOAD_MODEL_IDS,
    APPROVED_MODELS,
    DEFAULT_MMPROJ_FILE,
    DEFAULT_MMPROJ_LOCAL,
    DEFAULT_MODEL_CHOICE_IDS,
    DEFAULT_MODEL_DISPLAY_NAME,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    MULTIMODAL_MODEL_IDS,
    SYSTEM_MODEL,
    SYSTEM_MODEL_ID,
    approved_main_model_paths,
    approved_model_dir,
    approved_model_file_path,
    approved_model_for_file,
)
from config.paths import (
    _ensure_private_dir,
    _migrate_legacy_data_dir,
    _migrate_legacy_runtime_dir,
    APP_LOG_FILE,
    APP_NAME,
    APP_VERSION,
    ASSETS_ROOT,
    CACHE_DIR,
    DATA_DIR,
    DOCS_DIR,
    FILTERING_OUTPUT_DIR,
    HF_CACHE_DIR,
    ICONS_DIR,
    LEGACY_DATA_DIR,
    MIN_MACOS_VERSION,
    MODELS_DIR,
    PARSING_OUTPUT_DIR,
    PARSING_STAGING_DIR,
    PROFILE_FILE,
    PROJECT_ROOT,
    PROJECT_TMP_DIR,
    RAG_DEBUG_DIR,
    RAG_DEBUG_ENABLED,
    REPORTS_DIR,
    SETTINGS_FILE,
    list_uploaded_doc_paths,
)
from config.settings import (
    ANSWER_DETAIL_BALANCED,
    ANSWER_DETAIL_DEFAULT,
    ANSWER_DETAIL_DETAILED,
    ANSWER_DETAIL_MAX_TOKENS,
    ANSWER_DETAIL_SHORT,
    CURRENT_SETTINGS_SCHEMA_VERSION,
    AppSettings,
    ModelMetaSettings,
    _load_app_settings,
    _load_settings,
    _migrate_settings,
    _save_settings,
    add_hidden_biomarker,
    answer_detail_max_tokens,
    clear_hidden_biomarkers,
    get_default_model,
    is_onboarding_complete,
    load_answer_detail,
    load_ctx_size,
    load_default_model_id,
    load_hidden_biomarkers,
    load_max_tokens,
    load_model_meta,
    load_model_path,
    load_profile,
    migrate_profile_to_protected_file,
    save_answer_detail,
    save_ctx_size,
    save_default_model_id,
    save_max_tokens,
    save_model_meta,
    save_model_path,
    save_profile,
    set_onboarding_complete,
)


# Runtime tunables ─────────────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8765
KB_COLLECTION_NAME = "lab_documents"
EMBEDDER_BGE_M3 = "BAAI/bge-m3"
EMBEDDER_SNOWFLAKE_ARCTIC_L_V2 = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDER_JINA_V5_TEXT_SMALL = "jinaai/jina-embeddings-v5-text-small"
EMBEDDER_JINA_V5_TEXT_NANO = "jinaai/jina-embeddings-v5-text-nano"
KB_EMBEDDING_MODEL = EMBEDDER_JINA_V5_TEXT_SMALL
EMBEDDER_CONFIGS = {
    EMBEDDER_BGE_M3: {
        "loader_kwargs": {},
        "document_encode_kwargs": {},
        "query_encode_kwargs": {},
    },
    EMBEDDER_SNOWFLAKE_ARCTIC_L_V2: {
        "loader_kwargs": {},
        "document_encode_kwargs": {},
        "query_encode_kwargs": {},
    },
    EMBEDDER_JINA_V5_TEXT_SMALL: {
        "loader_kwargs": {"trust_remote_code": True},
        "document_encode_kwargs": {"task": "retrieval", "prompt_name": "document"},
        "query_encode_kwargs": {"task": "retrieval", "prompt_name": "query"},
    },
    EMBEDDER_JINA_V5_TEXT_NANO: {
        "loader_kwargs": {"trust_remote_code": True},
        "document_encode_kwargs": {"task": "retrieval", "prompt_name": "document"},
        "query_encode_kwargs": {"task": "retrieval", "prompt_name": "query"},
    },
}
KB_CHUNK_SIZE = 500
KB_CHUNK_OVERLAP = 100
KB_TOP_K = 15
LOGGER_LEVEL = "INFO"
LOGGER_MAX_BYTES = 1_000_000
LOGGER_BACKUP_COUNT = 5
PARSER_MAX_OUTPUT_TOKENS = 8192
PARSER_PDF_DPI = 200
PARSER_MAX_PARALLEL_PAGES = 2
PARSER_VISION_TIMEOUT_SECONDS = 600
PARSER_SANITIZE_TIMEOUT_SECONDS = 300
PARSER_METADATA_TIMEOUT_SECONDS = 120
LLM_SERVER_READY_TIMEOUT_SECONDS = 120
LLM_HEALTH_TIMEOUT_SECONDS = 2
LLM_TOKENIZE_TIMEOUT_SECONDS = 10
LLM_QUERY_MODIFICATION_TIMEOUT_SECONDS = 60
LLM_HYDE_TIMEOUT_SECONDS = 60
LLM_RAG_COMPRESSION_TIMEOUT_SECONDS = 120


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"
