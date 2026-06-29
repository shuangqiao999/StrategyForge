# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for StrategyForge backend (onedir)."""
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('strategy_forge')
hiddenimports += [
    'kuzu',
    'lancedb',
    'jieba',
    'pyarrow',
    'starlette',
    'sse_starlette',
    'anyio',
    'h11',
    'httpcore',
    'fastapi',
    'pydantic',
    'httpx',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
]

a = Analysis(
    ['src/strategy_forge/main.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'pandas',
        'numpy.testing',
        'scipy',
        'PIL',
        'torch',
        'torchvision',
        'transformers',
        'sklearn',
        'scikit-learn',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='strategy-forge-backend',
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
    name='strategy-forge-backend',
)
