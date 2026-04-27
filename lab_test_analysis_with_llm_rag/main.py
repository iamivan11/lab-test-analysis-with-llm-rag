import sys

from PySide6.QtWidgets import QApplication

from config import APP_LOG_FILE
from core.logger import enable_file_logging, log
from ui.main_window import MainWindow


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
