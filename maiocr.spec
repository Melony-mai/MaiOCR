# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for MaiOCR (Windows).
#
# Build:  pyinstaller maiocr.spec --noconfirm
# Output: dist/MaiOCR/MaiOCR.exe
#
# For GPU support install onnxruntime-gpu in the build environment BEFORE
# building; the same spec picks it up automatically.
#
# Single-file alternative: change `onedir` -> `onefile` in EXE(...) below and
# remove the COLLECT() block. Note onefile startup is slower (self-unpacks).

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

app_name = "MaiOCR"

datas = [
    # rapidocr ships its ONNX models + YAML configs inside the package
    *collect_data_files("rapidocr"),
]
# The user-supplied icon lives at the project root; ship it verbatim so the
# runtime (tray + window icons) uses exactly this file.
if os.path.exists("MaiOCR.ico"):
    datas.append(("MaiOCR.ico", "resources"))
elif os.path.exists("resources/MaiOCR.ico"):
    datas.append(("resources/MaiOCR.ico", "resources"))
elif os.path.exists("icon.ico"):
    datas.append(("icon.ico", "resources"))
elif os.path.exists("resources/icon.ico"):
    datas.append(("resources/icon.ico", "resources"))
binaries = [
    # native libs of rapidocr's transitive deps (deduped against hooks)
    *collect_dynamic_libs("onnxruntime"),
    *collect_dynamic_libs("shapely"),
    *collect_dynamic_libs("pyclipper"),
]

hiddenimports = [
    "mss",
    "pyperclip",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "pytest",
        "setuptools",
        # rapidocr ships alternative inference backends we never use; their
        # modules import torch/paddle/openvino at top level. Excluded so the
        # analysis can neither fail on them nor bundle heavy frameworks.
        "rapidocr.inference_engine.pytorch",
        "rapidocr.inference_engine.paddle",
        "rapidocr.inference_engine.openvino",
        "rapidocr.inference_engine.tensorrt",
        "rapidocr.inference_engine.mnn",
        "torch",
        "torchvision",
        "paddle",
        "paddlepaddle",
        "openvino",
        "tensorrt",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                       # GUI app: no console window
    icon=(
        "MaiOCR.ico"
        if os.path.exists("MaiOCR.ico")
        else ("resources/MaiOCR.ico" if os.path.exists("resources/MaiOCR.ico") else None)
    ),
    version="version_info.txt" if os.path.exists("version_info.txt") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=app_name,
)
