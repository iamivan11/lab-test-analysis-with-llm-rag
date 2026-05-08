# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH).parent
SOURCE_ROOT = ROOT / "lab_test_analysis_with_llm_rag"

datas = []
if (ROOT / "assets").exists():
    datas.append((str(ROOT / "assets"), "assets"))

binaries = []
if (ROOT / "bin").exists():
    binaries.extend((str(path), "bin") for path in (ROOT / "bin").iterdir() if path.is_file())

a = Analysis(
    [str(SOURCE_ROOT / "main.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "chromadb",
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
    icon=None,
    bundle_identifier="local.lab-analyzer",
    info_plist={
        "CFBundleDisplayName": "Lab Analyzer",
        "CFBundleName": "Lab Analyzer",
        "NSHighResolutionCapable": True,
    },
)
