from config import load_profile
from ui.profile_dialog import ProfileDialog


class ProfileController:
    def __init__(self, window):
        self.window = window

    def open_profile(self):
        dlg = ProfileDialog(self.window)
        dlg.exec()

    def build_profile_context(self) -> str:
        profile = load_profile()
        labels = {
            "name": "Name",
            "age": "Age",
            "gender": "Gender",
            "weight": "Weight",
            "height": "Height",
            "other": "Other",
        }
        lines = [f"{labels[key]}: {value}" for key, value in profile.items() if value]
        if not lines:
            return ""
        return "## Patient Profile\n\n" + "\n".join(lines)
