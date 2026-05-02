from config import load_profile


class ProfileController:
    def __init__(self, window):
        self.window = window

    def build_profile_context(self) -> str:
        profile = load_profile()
        labels = {
            "name": "Name",
            "age": "Age",
            "gender": "Gender",
            "weight": "Weight",
            "height": "Height",
            "smoking": "Smoking",
            "alcohol": "Alcohol",
            "surgeries": "Surgeries",
            "allergies": "Allergies",
            "other": "Other",
        }
        lines = [
            f"{label}: {profile[key]}"
            for key, label in labels.items()
            if profile.get(key)
        ]
        if not lines:
            return ""
        return "## Patient Profile\n\n" + "\n".join(lines)
