# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path

mvs_runtime_dir = Path("C:/Program Files (x86)/Common Files/MVS/Runtime/Win64_x64")

datas = [
    ("models", "models"),
    ("scripts", "scripts"),
    ("C:\\Program Files (x86)\\MVS\\Development\\Samples\\Python\\MvImport", "MvImport"),
    ("C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64", "mvs_runtime"),
]
if mvs_runtime_dir.exists():
    datas += [(str(path), ".") for path in mvs_runtime_dir.iterdir() if path.is_file()]
binaries = []
hiddenimports = []
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("PIL")
hiddenimports += collect_submodules("cv2")
tmp_ret = collect_all("ultralytics")
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

a = Analysis(
    ["seven_segment_live_camera_server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="LedInspectionBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name="LedInspectionBackend",
)
