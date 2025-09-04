# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('advanced_analysis.py', '.'), ('NSE_MKT_ANALYSIS_V2.py', '.'), ('strategies.py', '.'), ('symbol.txt', '.'), ('requirements.txt', '.'), ('TA_Lib-0.6.4-cp312-cp312-win_amd64.whl', '.')],
    hiddenimports=['talib'],
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
    a.binaries,
    a.datas,
    [],
    name='run_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
