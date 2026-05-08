"""Single source of truth for user-facing section names.

Sidebar tiles, Settings block headers, dialog titles, and any other place
in the UI that names an app section should import its label from here.
Renaming happens in one spot and propagates everywhere automatically.
"""


class SectionNames:
    PROFILE = "My Profile"
    MODELS = "Models"
    MEDICAL_DOCUMENTS = "Medical Documents"
    CHAT_WITH_DOCUMENTS = "Chat with Documents"
    HEALTH_REPORT = "Health Report"
    TRENDS = "Trends"
    SETTINGS = "Settings"


__all__ = ["SectionNames"]
