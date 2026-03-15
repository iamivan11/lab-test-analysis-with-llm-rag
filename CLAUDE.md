# CLAUDE.md

## Project

Lab test analysis application using LLM and RAG (Retrieval-Augmented Generation).

- **Language**: Python
- **Repo root**: This directory
- **Package manager**: uv
- **GUI framework**: PySide6 (Qt)
- **LLM runtime**: Ollama (local, HTTP API on localhost:11434)
- **PDF parsing**: Docling
- **Vector store**: ChromaDB (planned)

## Code Style

1. Write minimalistic, easy-to-understand code, according to best practices.

## Workflow Rules

1. Before writing any code, describe your approach and wait for approval.
2. If the requirements are ambiguous, ask clarifying questions before writing any code.
3. After finishing any code, list the edge cases and suggest test cases to cover them.
4. If a task requires changes to more than 3 files, stop and break it into smaller tasks first.
5. When there's a bug, start by writing a test that reproduces it, then fix it until the test passes.
6. Every time I correct you, reflect on what you did wrong and come up with a plan to never make the same mistake again.
