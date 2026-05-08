import sys

from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from config import (
    APP_LOG_FILE,
    MIN_MACOS_VERSION,
)
from core.logger import enable_file_logging, log
from core.macos_compat import check_macos_compatibility
from core.security import is_security_configured
from ui.main_window import MainWindow


def _create_main_window() -> MainWindow:
    window = MainWindow(
        start_locked=is_security_configured(),
    )
    window.show()
    log("APP", "Window shown")
    return window


def main():
    enable_file_logging(APP_LOG_FILE)
    log("APP", "Starting application")
    qt_app = QApplication(sys.argv)
    compatibility = check_macos_compatibility(MIN_MACOS_VERSION)
    if not compatibility.is_supported:
        log("APP", f"Unsupported macOS version: {compatibility.current_version}")
        QMessageBox.critical(
            None,
            "Unsupported macOS version",
            (
                f"This app requires macOS {compatibility.minimum_version} or newer.\n"
                f"Current version: {compatibility.current_version}"
            ),
        )
        sys.exit(1)

    qt_app.main_window = _create_main_window()

    log("APP", "Entering event loop")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
