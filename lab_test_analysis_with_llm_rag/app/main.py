import sys

from PySide6.QtWidgets import QApplication

from app.core.logger import log
from app.ui.main_window import MainWindow


def main():
    log("APP", "Starting application")
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    log("APP", "Window shown, entering event loop")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
