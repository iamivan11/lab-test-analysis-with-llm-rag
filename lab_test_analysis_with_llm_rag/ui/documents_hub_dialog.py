"""Backward-compatible import shim for the moved documents UI package."""

from ui.documents import DocumentsHubDialog, EnsureVisionModelWorker, IndexWorker

__all__ = [
    "DocumentsHubDialog",
    "EnsureVisionModelWorker",
    "IndexWorker",
]
