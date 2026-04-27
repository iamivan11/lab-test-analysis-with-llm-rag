from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QIntValidator, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import load_profile, save_profile
from ui.styles import STYLESHEET


class ProfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profile")
        self.setMinimumWidth(360)
        self.setStyleSheet(STYLESHEET)

        profile = load_profile()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # Name — letters, spaces, hyphens only
        self._name = QLineEdit(profile.get("name", ""))
        self._name.setPlaceholderText("e.g. John")
        self._name.setValidator(QRegularExpressionValidator(QRegularExpression(r"[A-Za-z\s\-']+")))
        form.addRow("Name", self._name)

        # Age -- integer 1-120
        self._age = QLineEdit(profile.get("age", ""))
        self._age.setPlaceholderText("e.g. 30")
        self._age.setValidator(QIntValidator(1, 120))
        form.addRow("Age", self._age)

        # Gender — toggle buttons
        gender_widget = QWidget()
        gender_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        gender_row = QHBoxLayout(gender_widget)
        gender_row.setContentsMargins(0, 0, 0, 0)
        gender_row.setSpacing(8)

        self._male_btn = QPushButton("Male")
        self._male_btn.setObjectName("genderButton")
        self._male_btn.setCheckable(True)
        self._male_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._female_btn = QPushButton("Female")
        self._female_btn.setObjectName("genderButton")
        self._female_btn.setCheckable(True)
        self._female_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._gender_group = QButtonGroup()
        self._gender_group.setExclusive(True)
        self._gender_group.addButton(self._male_btn)
        self._gender_group.addButton(self._female_btn)

        saved_gender = profile.get("gender", "")
        if saved_gender == "Male":
            self._male_btn.setChecked(True)
        elif saved_gender == "Female":
            self._female_btn.setChecked(True)

        gender_row.addWidget(self._male_btn, stretch=1)
        gender_row.addWidget(self._female_btn, stretch=1)
        form.addRow("Gender", gender_widget)

        # Weight — positive decimal number
        self._weight = QLineEdit(profile.get("weight", ""))
        self._weight.setPlaceholderText("e.g. 70")
        self._weight.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d{0,3}(\.\d{0,1})?"))
        )
        form.addRow("Weight (kg)", self._weight)

        # Height — positive integer
        self._height = QLineEdit(profile.get("height", ""))
        self._height.setPlaceholderText("e.g. 180")
        self._height.setValidator(QIntValidator(1, 300))
        form.addRow("Height (cm)", self._height)

        # Other — free text
        self._other = QTextEdit()
        self._other.setPlainText(profile.get("other", ""))
        self._other.setPlaceholderText("Any other relevant details...")
        self._other.setFixedHeight(80)
        form.addRow("Other", self._other)

        # Enter on any line edit acts as Save
        for field in (self._name, self._age, self._weight, self._height):
            field.returnPressed.connect(self._save)

        # Buttons — placed in the field column so width matches "Other" field
        btn_widget = QWidget()
        btn_row = QHBoxLayout(btn_widget)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("attachButton")
        cancel_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)

        form.addRow("", btn_widget)  # field column only → width matches Other
        layout.addLayout(form)

    def _save(self):
        profile = {
            "name": self._name.text().strip(),
            "age": self._age.text().strip(),
            "gender": (
                "Male"
                if self._male_btn.isChecked()
                else "Female"
                if self._female_btn.isChecked()
                else ""
            ),
            "weight": self._weight.text().strip(),
            "height": self._height.text().strip(),
            "other": self._other.toPlainText().strip(),
        }
        save_profile(profile)
        self.accept()
