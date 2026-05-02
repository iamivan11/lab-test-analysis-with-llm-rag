import sys

from PySide6.QtWidgets import QApplication

from config import APP_LOG_FILE, set_onboarding_complete
from core.logger import enable_file_logging, log
from ui.main_window import MainWindow

# Dev override: force the onboarding flow on every launch until removed.
# Remove the line below to restore "first launch onboards, then boots into Home".
FORCE_ONBOARDING_ON_EVERY_LAUNCH = True


def main():
    enable_file_logging(APP_LOG_FILE)
    log("APP", "Starting application")
    if FORCE_ONBOARDING_ON_EVERY_LAUNCH:
        set_onboarding_complete(False)
        log("APP", "Dev override: onboarding will replay on this launch")
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    log("APP", "Window shown, entering event loop")
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
