from core.logger import log


class DocumentsController:
    def __init__(self, window):
        self.window = window

    def on_docs_model_swapped(self, model_path: str, display_name: str):
        log("UI", f"Documents Hub swapped model to {display_name} ({model_path})")
        self.window._model_name = display_name
        self.window._status_label.setText(f"Model: {display_name}")

    def on_parsing_active_changed(self, active: bool):
        log("UI", f"Parsing active changed: {active}")
        self.window._parsing_active = active
        self.window._model_btn.setEnabled(not active)
