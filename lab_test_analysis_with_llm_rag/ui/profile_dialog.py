from PySide6.QtCore import QRegularExpression, Qt, Signal
from PySide6.QtGui import QFont, QIntValidator, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import load_profile, save_profile


def _bold_label(text: str) -> QLabel:
    lbl = QLabel(text)
    f = QFont()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("profileSectionLabel")
    return lbl


def _vertical_spacer(height: int) -> QWidget:
    spacer = QWidget()
    spacer.setFixedHeight(height)
    spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return spacer


class ProfileForm(QWidget):
    """Reusable profile form (no buttons). Used in both the dialog and the
    onboarding/profile screen."""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        profile = load_profile()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.form = QFormLayout()
        form = self.form
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._add_section("Basic")

        self._name = QLineEdit(profile.get("name", ""))
        self._name.setPlaceholderText("e.g. John")
        self._name.setValidator(QRegularExpressionValidator(QRegularExpression(r"[A-Za-z\s\-']+")))
        form.addRow(_bold_label("Name"), self._name)

        self._age = QLineEdit(profile.get("age", ""))
        self._age.setPlaceholderText("e.g. 30")
        self._age.setValidator(QIntValidator(1, 120))
        form.addRow(_bold_label("Age"), self._age)

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

        self._gender_group = QButtonGroup(self)
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
        form.addRow(_bold_label("Gender"), gender_widget)

        self._weight = QLineEdit(profile.get("weight", ""))
        self._weight.setPlaceholderText("e.g. 70")
        self._weight.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d{0,3}(\.\d{0,1})?"))
        )
        form.addRow(_bold_label("Weight (kg)"), self._weight)

        self._height = QLineEdit(profile.get("height", ""))
        self._height.setPlaceholderText("e.g. 180")
        self._height.setValidator(QIntValidator(1, 300))
        form.addRow(_bold_label("Height (cm)"), self._height)

        self._add_section("Lifestyle")

        smoking_widget, self._smoking_yes_btn, self._smoking_no_btn, self._smoking_group = (
            self._build_yes_no_row(profile.get("smoking", ""))
        )
        form.addRow(_bold_label("Smoking"), smoking_widget)

        alcohol_widget, self._alcohol_yes_btn, self._alcohol_no_btn, self._alcohol_group = (
            self._build_yes_no_row(profile.get("alcohol", ""))
        )
        form.addRow(_bold_label("Alcohol"), alcohol_widget)

        TEXT_FIELD_HEIGHT = 88

        self._add_section("Medical")

        self._surgeries = QTextEdit()
        self._surgeries.setPlainText(profile.get("surgeries", ""))
        self._surgeries.setPlaceholderText("Past surgeries with year if known")
        self._surgeries.setMinimumHeight(TEXT_FIELD_HEIGHT)
        self._surgeries.setMaximumHeight(TEXT_FIELD_HEIGHT)
        form.addRow(_bold_label("Surgeries"), self._surgeries)

        self._allergies = QTextEdit()
        self._allergies.setPlainText(profile.get("allergies", ""))
        self._allergies.setPlaceholderText("Drug, food, or environmental allergies")
        self._allergies.setMinimumHeight(TEXT_FIELD_HEIGHT)
        self._allergies.setMaximumHeight(TEXT_FIELD_HEIGHT)
        form.addRow(_bold_label("Allergies"), self._allergies)

        self._other = QTextEdit()
        self._other.setPlainText(profile.get("other", ""))
        self._other.setPlaceholderText("Any other relevant details...")
        self._other.setMinimumHeight(TEXT_FIELD_HEIGHT)
        self._other.setMaximumHeight(TEXT_FIELD_HEIGHT)
        form.addRow(_bold_label("Other"), self._other)

        for field in (self._name, self._age, self._weight, self._height):
            field.returnPressed.connect(self.submitted.emit)

        outer.addLayout(form)

    def _add_section(self, title: str) -> None:
        self.form.addRow(_vertical_spacer(8))
        spacer = QWidget()
        spacer.setFixedHeight(0)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.form.addRow(_section_label(title), spacer)
        self.form.addRow(_vertical_spacer(4))

    def _build_yes_no_row(self, value: str):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        yes_btn = QPushButton("Yes")
        yes_btn.setObjectName("genderButton")
        yes_btn.setCheckable(True)
        yes_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        no_btn = QPushButton("No")
        no_btn.setObjectName("genderButton")
        no_btn.setCheckable(True)
        no_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(yes_btn)
        group.addButton(no_btn)

        if value == "Yes":
            yes_btn.setChecked(True)
        elif value == "No":
            no_btn.setChecked(True)

        row.addWidget(yes_btn, stretch=1)
        row.addWidget(no_btn, stretch=1)
        return widget, yes_btn, no_btn, group

    def reload(self):
        profile = load_profile()
        self._name.setText(profile.get("name", ""))
        self._age.setText(profile.get("age", ""))
        self._weight.setText(profile.get("weight", ""))
        self._height.setText(profile.get("height", ""))
        self._surgeries.setPlainText(profile.get("surgeries", ""))
        self._allergies.setPlainText(profile.get("allergies", ""))
        self._other.setPlainText(profile.get("other", ""))
        gender = profile.get("gender", "")
        self._male_btn.setChecked(gender == "Male")
        self._female_btn.setChecked(gender == "Female")
        smoking = profile.get("smoking", "")
        self._smoking_yes_btn.setChecked(smoking == "Yes")
        self._smoking_no_btn.setChecked(smoking == "No")
        alcohol = profile.get("alcohol", "")
        self._alcohol_yes_btn.setChecked(alcohol == "Yes")
        self._alcohol_no_btn.setChecked(alcohol == "No")

    def save(self):
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
            "smoking": (
                "Yes"
                if self._smoking_yes_btn.isChecked()
                else "No"
                if self._smoking_no_btn.isChecked()
                else ""
            ),
            "alcohol": (
                "Yes"
                if self._alcohol_yes_btn.isChecked()
                else "No"
                if self._alcohol_no_btn.isChecked()
                else ""
            ),
            "surgeries": self._surgeries.toPlainText().strip(),
            "allergies": self._allergies.toPlainText().strip(),
            "other": self._other.toPlainText().strip(),
        }
        save_profile(profile)


__all__ = ["ProfileForm"]
