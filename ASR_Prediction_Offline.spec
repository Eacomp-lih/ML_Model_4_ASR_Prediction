"""PyInstaller build configuration for the Windows offline desktop app."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all


root = Path(SPECPATH)
datas = [
    (str(root / "streamlit_app.py"), "."),
    (str(root / "prediction_core.py"), "."),
    (str(root / "run_metadata.json"), "."),
    (str(root / ".streamlit" / "config.toml"), ".streamlit"),
    (str(root / "models"), "models"),
    (str(root / "data"), "data"),
    (str(root / "assets"), "assets"),
    (str(root / "training"), "training"),
]
binaries = []
hiddenimports = []
for package in ("streamlit", "sklearn", "plotly", "webview", "reportlab",
                "extra_streamlit_components", "openpyxl"):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

analysis = Analysis(
    [str(root / "desktop_launcher.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="ASR_Prediction_Offline",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
