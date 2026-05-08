import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceCapabilities:
    system: str
    machine: str
    is_apple_silicon: bool
    is_intel_mac: bool
    metal_available: bool


def current_device_capabilities() -> DeviceCapabilities:
    system = platform.system()
    machine = platform.machine()
    is_mac = system == "Darwin"
    is_apple_silicon = is_mac and machine == "arm64"
    is_intel_mac = is_mac and machine == "x86_64"
    return DeviceCapabilities(
        system=system,
        machine=machine,
        is_apple_silicon=is_apple_silicon,
        is_intel_mac=is_intel_mac,
        metal_available=_metal_available(is_apple_silicon=is_apple_silicon),
    )


def llama_gpu_layer_args(capabilities: DeviceCapabilities) -> list[str]:
    """Use Metal when available; otherwise keep llama.cpp on CPU."""
    if capabilities.metal_available:
        return ["-ngl", "99"]
    return ["-ngl", "0"]


def _metal_available(*, is_apple_silicon: bool) -> bool:
    if platform.system() != "Darwin":
        return False
    if is_apple_silicon:
        return True
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = result.stdout.lower()
    return "metal: supported" in output or "metal family" in output
