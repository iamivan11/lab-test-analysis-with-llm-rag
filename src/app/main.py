import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main():
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
