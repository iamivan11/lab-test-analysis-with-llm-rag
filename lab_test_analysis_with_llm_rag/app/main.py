import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.core.logger import enable_file_logging, log
from app.ui.main_window import MainWindow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_LOG_FILE = PROJECT_ROOT / "tmp" / "app.log"


def main():
    enable_file_logging(APP_LOG_FILE)
    log("APP", "Starting application")
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    log("APP", "Window shown, entering event loop")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
