# Lab Analyzer

macOS desktop app for analyzing medical lab reports with a local LLM and RAG pipeline.

The app is built for private, on-device processing: documents, parsed results, vector index,
profile data, logs, and generated reports are stored locally under macOS Application Support.

## Features

- Profile: stores basic user context used in prompts.
- Model management: downloads and runs approved local GGUF models.
- Document parsing: imports PDF, PNG, and JPEG reports and parses them with a local VLM.
- Document filtering: removes non-essential document noise before indexing.
- RAG chat: retrieves relevant chunks from the local vector DB and answers questions with the local model.
- Biomarker trends: extracts numeric biomarkers and displays their dynamics over time.
- Health report: generates a concise medical summary report from parsed documents.
- Reindexing: rebuilds the vector DB from already parsed/filtered outputs without reparsing files.
- macOS packaging: builds `.app` and `.dmg` artifacts.

## Tech Stack

- Python 3.12
- PySide6 for the desktop UI
- llama.cpp `llama-server` for local LLM/VLM inference
- GGUF models
- ChromaDB for the local vector database
- sentence-transformers for embeddings
- Jina embeddings as the current embedding model
- pydantic for typed settings/config validation
- PyInstaller for macOS app packaging
- uv for dependency and environment management

## Project Structure

```text
lab_test_analysis_with_llm_rag/
  main.py                  # app entrypoint
  config/                  # central app config package
    paths.py               # filesystem layout, dir creation, migrations
    models.py              # APPROVED_MODELS catalog + helpers
    settings.py            # AppSettings + persistence (Pydantic + RLock)
    __init__.py            # re-exports + runtime tunables, format_size
  core/                    # non-UI logic
    biomarkers/            # biomarker extraction package
      store.py             # cache load/save, document helpers
      units.py             # canonical-unit conversion table
      aggregate.py         # per-biomarker time-series aggregation
      extract.py           # BiomarkerExtractionWorker + LLM pipeline
      __init__.py          # re-exports
    document_parser.py     # document parsing/filtering pipeline
    knowledge_base.py      # chunking, embeddings, ChromaDB indexing/retrieval
    llm_engine.py          # streaming LLM client (chat, RAG, summarize)
    llm_server.py          # llama-server lifecycle + port selection
    model_hub.py           # approved model downloads
    rag_context.py         # retrieved context preparation/compression
    health_report.py       # medical report generation
    messages.py            # centralised user-facing error/status strings
    qthread_utils.py       # StoppableQThread base for workers
    http_client.py         # cancellable POST + retry helper
    security.py            # password-protected file encryption
    file_io.py             # atomic JSON/text writes
    logger.py              # rotating-file logger + crash hooks
    user_data.py           # Clear User Data helper
    macos_compat.py        # macOS version check + app-support dir
    device_compat.py       # Metal/CPU capability detection
    llama_setup.py         # bundled llama-server install
    model_meta.py          # GGUF metadata reading
    prompts.py             # system prompts (chat, RAG, health report)
    chat_store.py          # chat persistence
  ui/                      # PySide6 UI; each section follows the
    components.py          # `view.py | form.py + controller.py + workers.py`
    screens.py             # pattern as needed.
    sections.py
    styles.py
    main_window.py
    chat/                  # document chat
    documents/             # upload, parse, reindex
    models/                # model download/selection
    profile/               # profile form
    settings/              # app settings
    health_report/         # report UI
    biomarkers/            # biomarker trends UI (view + charts)
    onboarding/            # first-run onboarding
    security/              # password unlock screen
assets/
  icons/                   # in-app UI icons
  app_icon/                # macOS bundle icon (logo.png + logo.icns)
bin/                       # local llama-server binary and dylibs
packaging/                 # macOS packaging scripts/spec
tests/                     # automated tests
report/                    # thesis/report sources
tmp/                       # local dev/debug outputs, ignored by git
```

## Current RAG Setup

- Embedding model: `jinaai/jina-embeddings-v5-text-small`
- Chunk size: `500`
- Chunk overlap: `100`
- Retrieval count: `top_k = 15`
- Stored metadata per chunk:
  - `source`
  - `report_date`
  - `report_type`
  - `chunk_index`
- Retrieved chunks are compressed before final answer generation.

## Local Data

Runtime data is stored outside the repository:

```text
~/Library/Application Support/Lab Analyzer/
  models/
  documents/
  parsing_output/
  filtering_output/
  reports/
  logs/
  settings.json
  profile.json
```

The packaged app should not include user medical documents, parsed outputs, eval files, logs,
settings, or profile data.

## Development Setup

Install dependencies:

```bash
uv sync
```

Run the app from source:

```bash
uv run lab-analyzer
```

Alternative direct run:

```bash
./.venv/bin/python lab_test_analysis_with_llm_rag/main.py
```

## Tests

```bash
./.venv/bin/python -m pytest
```

## Build macOS App

```bash
packaging/build_macos.sh
```

Output: `dist/Lab Analyzer.dmg`.

The DMG is not notarized, so macOS Gatekeeper may warn external users.

### macOS refuses to open the app ("damaged" / "cannot be opened")

Because the DMG isn't notarized, macOS quarantines the `.app` on download and Gatekeeper refuses to launch it. After dragging `Lab Analyzer.app` into `/Applications`, strip the quarantine attribute:

```bash
xattr -dr com.apple.quarantine "/Applications/Lab Analyzer.app"
```

Then open the app normally.

## Evaluation Dataset

The open synthetic dataset used for evaluation (30 reports across 6 panels and 5 timepoints, with ground-truth biomarker values and chat Q&A pairs) is available at:
https://drive.google.com/drive/folders/1mQY7hP6vl4UkyA5q8l4BlnCi0QMSXxhn?usp=sharing

