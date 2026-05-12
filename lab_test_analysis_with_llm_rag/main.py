import multiprocessing
import sys

# PyInstaller-frozen apps re-invoke their own binary for any
# multiprocessing-spawned child. Without freeze_support(), the child
# falls through to main() and opens a new GUI window — observed in
# this app as the bundled .app spawning a fresh window per spawned
# child during onboarding (libraries like sentence-transformers and
# chromadb spin up worker processes for parallel work). The Python
# stdlib documents freeze_support() as a no-op outside Windows, but
# PyInstaller's bootloader installs a macOS hook that intercepts
# this call and routes spawned children back to the bootloader.
# This MUST run before any import that could trigger multiprocessing.
multiprocessing.freeze_support()

from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

from config import (
    APP_LOG_FILE,
    MIN_MACOS_VERSION,
)
from core.logger import enable_file_logging, install_global_excepthooks, log
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
    # Install crash hooks AFTER the file handler is wired so escaped
    # exceptions and native faults are captured on disk, not just stderr.
    install_global_excepthooks()
    log("APP", "Starting application")
    qt_app = QApplication(sys.argv)
    compatibility = check_macos_compatibility(MIN_MACOS_VERSION)
    if not compatibility.is_supported:
        from core.messages import UNSUPPORTED_MACOS

        log("APP", f"Unsupported macOS version: {compatibility.current_version}")
        QMessageBox.critical(
            None,
            "Unsupported macOS version",
            UNSUPPORTED_MACOS.format(
                minimum=compatibility.minimum_version,
                current=compatibility.current_version,
            ),
        )
        sys.exit(1)

    qt_app.main_window = _create_main_window()

    log("APP", "Entering event loop")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
