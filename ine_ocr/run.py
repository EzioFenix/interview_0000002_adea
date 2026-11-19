#!/usr/bin/env python3
"""Arranca la interfaz:   python3 run.py      (abre http://localhost:8501)"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Si ya corriste install.sh, usa automáticamente el entorno virtual del proyecto
if VENV_PY.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), str(ROOT / "run.py"), *sys.argv[1:]])

try:
    import streamlit  # noqa: F401
except ImportError:
    sys.exit("Faltan dependencias. Ejecuta primero:  bash install.sh")

raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                                  "--browser.gatherUsageStats", "false", *sys.argv[1:]]))
