import threading

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import (
    DEFAULT_MMPROJ_FILE,
    DEFAULT_MMPROJ_LOCAL,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_REPO,
    MODELS_DIR,
    format_size,
    save_model_path,
)
from core.logger import log
from core.model_hub import download_model
from ui.components import header_label, profile_scroll_area
from ui.documents.view import DocumentsHubWidget
from ui.profile_dialog import ProfileForm


def _hero_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 56pt; font-weight: bold;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


_NAV_BTN_SIZE = (100, 38)


def _onboarding_nav_row(*, on_back, on_continue, continue_enabled: bool = True) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(12)
    row.addStretch()

    if on_back is None:
        spacer = QWidget()
        spacer.setFixedSize(*_NAV_BTN_SIZE)
        row.addWidget(spacer)
    else:
        back_btn = QPushButton("Back")
        back_btn.setObjectName("attachButton")
        back_btn.setFixedSize(*_NAV_BTN_SIZE)
        back_btn.clicked.connect(on_back)
        row.addWidget(back_btn)

    cont_btn = QPushButton("Continue")
    cont_btn.setFixedSize(*_NAV_BTN_SIZE)
    cont_btn.setEnabled(continue_enabled)
    cont_btn.clicked.connect(on_continue)
    row.addWidget(cont_btn)

    row.continue_btn = cont_btn  # type: ignore[attr-defined]
    return row


class WelcomeScreen(QWidget):
    start_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(32)

        layout.addStretch()
        layout.addWidget(_hero_label("Welcome!"))

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        start_btn = QPushButton("Start")
        start_btn.setFixedSize(200, 48)
        f = start_btn.font()
        f.setPointSize(14)
        f.setBold(True)
        start_btn.setFont(f)
        start_btn.clicked.connect(self.start_clicked.emit)
        btn_row.addWidget(start_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()


class ProfileSetupScreen(QWidget):
    continue_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 24, 40, 24)
        outer.setSpacing(12)

        outer.addWidget(header_label("To get started, please fill in your profile data:"))

        form_container = QWidget()
        form_wrap = QHBoxLayout(form_container)
        form_wrap.setContentsMargins(0, 0, 0, 0)
        form_wrap.addStretch()
        self._form = ProfileForm(self)
        self._form.setMinimumWidth(440)
        self._form.setMaximumWidth(560)
        self._form.submitted.connect(self._on_continue)
        form_wrap.addWidget(self._form)
        form_wrap.addStretch()

        outer.addWidget(profile_scroll_area(form_container), stretch=1)

        outer.addLayout(_onboarding_nav_row(on_back=None, on_continue=self._on_continue))

    def _on_continue(self):
        self._form.save()
        self.continue_clicked.emit()

    def reload(self):
        self._form.reload()


class DefaultModelDownloadWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            files = [
                (DEFAULT_MODEL_FILE, DEFAULT_MODEL_FILE),
                (DEFAULT_MMPROJ_FILE, DEFAULT_MMPROJ_LOCAL),
            ]
            model_path = MODELS_DIR / DEFAULT_MODEL_FILE

            done_bytes_prior = 0
            current_total_estimate = 0

            for hf_name, local_name in files:
                dest = MODELS_DIR / local_name
                if dest.exists():
                    size = dest.stat().st_size
                    done_bytes_prior += size
                    current_total_estimate += size
                    self.progress.emit(done_bytes_prior, current_total_estimate, local_name)
                    continue

                file_total_holder = {"total": 0}

                def on_progress(
                    downloaded: int,
                    total: int,
                    _name=local_name,
                    _done_bytes_prior=done_bytes_prior,
                    _file_total_holder=file_total_holder,
                ):
                    if total and _file_total_holder["total"] != total:
                        _file_total_holder["total"] = total
                    estimated_total = _done_bytes_prior + max(
                        _file_total_holder["total"], downloaded
                    )
                    self.progress.emit(
                        _done_bytes_prior + downloaded, estimated_total, _name
                    )

                log("HUB", f"Onboarding download: {hf_name} -> {local_name}")
                download_model(
                    DEFAULT_MODEL_REPO,
                    hf_name,
                    on_progress=on_progress,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    self.error_occurred.emit("Download cancelled")
                    return

                if hf_name != local_name:
                    src = MODELS_DIR / hf_name
                    if src.exists():
                        src.rename(dest)

                size = dest.stat().st_size if dest.exists() else file_total_holder["total"]
                done_bytes_prior += size

            self.finished.emit(str(model_path))
        except Exception as e:
            log("HUB", f"DefaultModelDownloadWorker: ERROR {e}")
            self.error_occurred.emit(str(e))


class ModelDownloadScreen(QWidget):
    proceed_clicked = Signal(str)
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: DefaultModelDownloadWorker | None = None
        self._model_path: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        layout.addStretch()

        layout.addWidget(header_label("Loading the default AI model to your device..."))

        info = QLabel(f"{DEFAULT_MODEL_REPO} — {DEFAULT_MODEL_FILE} (+ vision projector)")
        info.setObjectName("statusLabel")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        self._current_file_label = QLabel("Preparing download...")
        self._current_file_label.setObjectName("statusLabel")
        self._current_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._current_file_label)

        bar_row = QHBoxLayout()
        bar_row.addStretch()
        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(640)
        bar_row.addWidget(self._progress)
        bar_row.addStretch()
        layout.addLayout(bar_row)

        self._size_label = QLabel("0 B / —")
        self._size_label.setObjectName("statusLabel")
        self._size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._size_label)

        layout.addStretch()
        nav_row = _onboarding_nav_row(
            on_back=self.back_clicked.emit,
            on_continue=self._on_continue,
            continue_enabled=False,
        )
        self._continue_btn = nav_row.continue_btn
        layout.addLayout(nav_row)

    def start(self):
        if self._worker and self._worker.isRunning():
            return
        model_path = MODELS_DIR / DEFAULT_MODEL_FILE
        mmproj_path = MODELS_DIR / DEFAULT_MMPROJ_LOCAL
        if model_path.exists() and mmproj_path.exists():
            total = model_path.stat().st_size + mmproj_path.stat().st_size
            self._progress.setValue(1000)
            self._size_label.setText(f"{format_size(total)} / {format_size(total)}")
            self._on_done(str(model_path))
            return

        self._continue_btn.setEnabled(False)
        self._current_file_label.setText("Starting...")
        self._size_label.setText("0 B / —")
        self._worker = DefaultModelDownloadWorker()
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, downloaded: int, total: int, name: str):
        self._current_file_label.setText(f"Downloading {name}")
        if total > 0:
            self._progress.setValue(int(downloaded * 1000 / total))
            self._size_label.setText(f"{format_size(downloaded)} / {format_size(total)}")
        else:
            self._progress.setValue(0)
            self._size_label.setText(format_size(downloaded))

    def _on_done(self, model_path: str):
        self._model_path = model_path
        self._progress.setValue(1000)
        self._current_file_label.setText("Download complete")
        save_model_path(model_path)
        self._continue_btn.setEnabled(True)

    def _on_error(self, error: str):
        self._current_file_label.setText(f"Error: {error}")
        self._continue_btn.setEnabled(False)

    def _on_continue(self):
        if not self._model_path:
            return
        self.proceed_clicked.emit(self._model_path)


class DocumentsSetupScreen(QWidget):
    proceed_clicked = Signal()
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        layout.addWidget(
            header_label("Upload your medical documents now, or skip and add them later")
        )

        self.docs = DocumentsHubWidget(self)
        layout.addWidget(self.docs, stretch=1)

        nav_row = _onboarding_nav_row(
            on_back=self.back_clicked.emit,
            on_continue=self._on_continue,
        )
        self._continue_btn = nav_row.continue_btn
        layout.addLayout(nav_row)

    def _on_continue(self):
        if self.docs.is_busy():
            return
        self.proceed_clicked.emit()
