from core.logger import log
from ui.documents.view import DocumentsHubDialog


class DocumentsController:
    def __init__(self, window):
        self.window = window

    def open_documents_hub(self):
        if self.window._docs_dialog is None:
            self.window._docs_dialog = DocumentsHubDialog(self.window)
            self.window._docs_dialog.model_swapped.connect(self.on_docs_model_swapped)
            self.window._docs_dialog.parsing_active_changed.connect(self.on_parsing_active_changed)
        self.window._docs_dialog.show()
        self.window._docs_dialog.raise_()
        self.window._docs_dialog.activateWindow()

    def on_docs_model_swapped(self, model_path: str, display_name: str):
        log("UI", f"Documents Hub swapped model to {display_name} ({model_path})")
        self.window._model_name = display_name
        self.window._status_label.setText(f"Model: {display_name}")

    def on_parsing_active_changed(self, active: bool):
        log("UI", f"Parsing active changed: {active}")
        self.window._parsing_active = active
        self.window._model_btn.setEnabled(not active)
