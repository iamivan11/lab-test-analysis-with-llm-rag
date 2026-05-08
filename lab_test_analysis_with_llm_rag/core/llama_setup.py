"""Automatic llama-server build and installation.

Clones llama.cpp, builds llama-server, and installs the binary with all
required shared libraries into the project's bin/ directory.  The rpath is
fixed so the binary always finds its dylibs next to itself.
"""

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from config import PROJECT_ROOT
from core.logger import log

BIN_DIR = PROJECT_ROOT / "bin"
SERVER_BINARY = BIN_DIR / "llama-server"
REPO_URL = "https://github.com/ggerganov/llama.cpp.git"

REQUIRED_DYLIBS = [
    "libggml-base.0.dylib",
    "libggml-blas.0.dylib",
    "libggml-cpu.0.dylib",
    "libggml-metal.0.dylib",
    "libggml.0.dylib",
    "libllama.0.dylib",
    "libmtmd.0.dylib",
]

REQUIRED_BUILD_TOOLS = ["git", "cmake", "install_name_tool", "otool"]


def _check_build_tools() -> None:
    """Fail fast if required build tools are not on PATH."""
    missing = [tool for tool in REQUIRED_BUILD_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Cannot build llama-server — missing required tools: "
            f"{', '.join(missing)}. Install Xcode Command Line Tools "
            "(`xcode-select --install`) and CMake (`brew install cmake`), then retry."
        )


def is_server_ready() -> bool:
    """Check that the binary and all required dylibs are present."""
    if not SERVER_BINARY.exists():
        return False
    return all((BIN_DIR / lib).exists() for lib in REQUIRED_DYLIBS)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr[-500:]}")


def _cpu_count() -> str:
    import os

    return str(os.cpu_count() or 4)


def build_and_install(
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Clone llama.cpp, build llama-server, install to bin/."""

    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    _check_build_tools()

    build_dir = Path(tempfile.mkdtemp(prefix="llama-build-"))

    try:
        # Clone
        progress("Cloning llama.cpp (shallow)...")
        _run(["git", "clone", "--depth", "1", REPO_URL, str(build_dir / "llama.cpp")])

        src = build_dir / "llama.cpp"

        # Configure
        progress("Configuring build...")
        _run(
            [
                "cmake",
                "-B",
                "build",
                "-DGGML_METAL=ON",
                "-DLLAMA_CURL=ON",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            cwd=src,
        )

        # Build
        progress("Building llama-server (this may take a few minutes)...")
        _run(
            [
                "cmake",
                "--build",
                "build",
                "--target",
                "llama-server",
                "-j",
                _cpu_count(),
            ],
            cwd=src,
        )

        built_bin = src / "build" / "bin"

        # Install
        progress("Installing llama-server...")
        BIN_DIR.mkdir(parents=True, exist_ok=True)

        shutil.copy2(built_bin / "llama-server", SERVER_BINARY)

        for lib in REQUIRED_DYLIBS:
            lib_path = built_bin / lib
            if lib_path.exists():
                shutil.copy2(lib_path, BIN_DIR / lib)
            else:
                raise RuntimeError(f"Expected library not found: {lib}")

        # Fix rpath so the binary finds dylibs next to itself
        progress("Fixing library paths...")
        # Read current rpaths
        result = subprocess.run(
            ["otool", "-l", str(SERVER_BINARY)],
            capture_output=True,
            text=True,
        )
        # Remove any existing rpaths
        lines = result.stdout.splitlines()
        for i, line in enumerate(lines):
            if "LC_RPATH" in line:
                for j in range(i, min(i + 3, len(lines))):
                    if "path " in lines[j]:
                        old_path = lines[j].strip().split("path ")[1].split(" (")[0]
                        subprocess.run(
                            ["install_name_tool", "-delete_rpath", old_path, str(SERVER_BINARY)],
                            capture_output=True,
                        )

        # Add @executable_path as the sole rpath
        _run(["install_name_tool", "-add_rpath", "@executable_path", str(SERVER_BINARY)])

        progress("llama-server ready")

    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def ensure_server(
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Ensure llama-server is installed and ready. Build if necessary."""
    if is_server_ready():
        log("SETUP", "llama-server already installed, skipping build")
        return
    log("SETUP", "llama-server not found, starting build...")
    build_and_install(on_progress=on_progress)
    if not is_server_ready():
        raise RuntimeError(
            "llama-server build completed but binary check failed. "
            "Ensure cmake, git, and Xcode command-line tools are installed."
        )
