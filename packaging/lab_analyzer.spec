# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(SPECPATH).parent
SOURCE_ROOT = ROOT / "lab_test_analysis_with_llm_rag"

datas = []
if (ROOT / "assets").exists():
    datas.append((str(ROOT / "assets"), "assets"))

binaries = []
if (ROOT / "bin").exists():
    binaries.extend((str(path), "bin") for path in (ROOT / "bin").iterdir() if path.is_file())

# chromadb / huggingface_hub do dynamic imports (backend / telemetry /
# segment impls; HF utils submodules) that PyInstaller's static analysis
# doesn't see. Pulling every submodule into hiddenimports avoids
# "No module named 'chromadb.api.rust'" / similar at runtime in the
# bundled .app.
#
# Filter out chromadb's test/cli/server submodules: we never run them
# at runtime, and bundling them pulls in optional deps (pytest, fastapi,
# uvicorn, typer, kubernetes) that bloat the .app by tens of MB and emit
# noisy "module not found" warnings during the PyInstaller build.
def _keep_chromadb_submodule(name: str) -> bool:
    excluded_prefixes = (
        "chromadb.test",
        "chromadb.cli",
        "chromadb.server",
    )
    return not any(name.startswith(prefix) for prefix in excluded_prefixes)


chromadb_submodules = [
    name
    for name in collect_submodules("chromadb")
    if _keep_chromadb_submodule(name)
]
huggingface_hub_submodules = collect_submodules("huggingface_hub")

a = Analysis(
    [str(SOURCE_ROOT / "main.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        *chromadb_submodules,
        *huggingface_hub_submodules,
        "cryptography",
        "pypdfium2",
        "sentence_transformers",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Lab Analyzer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Lab Analyzer",
)
app = BUNDLE(
    coll,
    name="Lab Analyzer.app",
    icon=str(ROOT / "assets" / "app_icon" / "logo.icns"),
    bundle_identifier="local.lab-analyzer",
    info_plist={
        "CFBundleDisplayName": "Lab Analyzer",
        "CFBundleName": "Lab Analyzer",
        "NSHighResolutionCapable": True,
    },
)
