import os
import platform
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MacOSCompatibility:
    is_supported: bool
    current_version: str
    minimum_version: str
    reason: str = ""


def application_support_dir(app_folder_name: str) -> Path:
    """Return the per-user app data directory on macOS."""
    if override := os.environ.get("LAB_ANALYZER_DATA_DIR"):
        return Path(override)
    return Path.home() / "Library" / "Application Support" / app_folder_name


def check_macos_compatibility(minimum_version: str) -> MacOSCompatibility:
    """Validate macOS version without blocking non-macOS dev/test environments."""
    if platform.system() != "Darwin":
        return MacOSCompatibility(True, platform.system(), minimum_version)

    current_version = platform.mac_ver()[0]
    if not current_version:
        return MacOSCompatibility(True, "unknown", minimum_version)

    if _version_tuple(current_version) < _version_tuple(minimum_version):
        return MacOSCompatibility(
            False,
            current_version,
            minimum_version,
            f"macOS {minimum_version} or newer is required.",
        )
    return MacOSCompatibility(True, current_version, minimum_version)


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)
