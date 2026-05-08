# Lab Analyzer

Local macOS desktop app for analyzing medical lab reports with a local LLM and RAG pipeline.

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

## Project Structure

```text
lab_test_analysis_with_llm_rag/
  main.py                  # app entrypoint
  config.py                # central app config, paths, model registry, RAG settings
  core/                    # non-UI logic
    document_parser.py     # document parsing/filtering pipeline
    knowledge_base.py      # chunking, embeddings, ChromaDB indexing/retrieval
    llm_server.py          # llama-server lifecycle
    model_hub.py           # approved model downloads
    rag_context.py         # retrieved context preparation/compression
    health_report.py       # medical report generation
    biomarkers.py          # biomarker extraction/trend data
  ui/                      # PySide6 UI
    chat/                  # document chat
    documents/             # upload, parse, reindex
    models/                # model download/selection
    profile/               # profile form
    settings/              # app settings
    health_report/         # report UI
    biomarkers/            # biomarker trends UI
    onboarding/            # first-run onboarding
assets/icons/              # app UI icons
bin/                       # local llama-server binary and dylibs
packaging/                 # macOS packaging scripts/spec
tests/                     # automated tests
report/                    # thesis/report sources
tmp/                       # local dev/debug outputs, ignored by git
```

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

Run all tests:

```bash
./.venv/bin/python -m pytest
```

Run focused smoke/config tests:

```bash
./.venv/bin/python -m pytest tests/test_config.py tests/test_ui_smoke.py -q
```

## Build macOS App

Build `.app` and `.dmg`:

```bash
packaging/build_macos.sh
```

Outputs:

```text
dist/Lab Analyzer.app
dist/Lab Analyzer.dmg
```

The build script:

- runs PyInstaller using `packaging/lab_analyzer.spec`;
- includes UI icons from `assets/`;
- includes `bin/llama-server` and required dylibs;
- verifies code signing;
- creates a compressed DMG with an Applications shortcut.

Current limitation: the DMG is not notarized, so macOS Gatekeeper may warn external users.

## Model Notes

Supported downloadable models are defined in `config.py` under `APPROVED_MODELS`.
The default system model is selected by `SYSTEM_MODEL_ID`.

Large GGUF model files are stored in Application Support and are not committed to the repo.

## Security Notes

- The app is designed for local processing.
- User data is stored locally in Application Support.
- Sensitive runtime directories are created with private permissions where possible.
- Final production distribution should add notarization and a stricter review of logs/debug outputs.

## Packaging Sanity Checks

After building, verify that no personal data was bundled:

```bash
find "dist/Lab Analyzer.app/Contents" -type d \
  \( -name parsing_output -o -name filtering_output -o -name tmp -o -name eval -o -name report \)

find "dist/Lab Analyzer.app/Contents" -type f \
  \( -name "profile.json" -o -name "settings.json" -o -name "app.log" -o -name "bt_*.md" -o -name "s_*.md" -o -name "ts_*.md" \)
```

Both commands should return no project/user medical files.
