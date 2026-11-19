#!/usr/bin/env bash
# Instala todo en un entorno virtual local (.venv). Uso:  bash install.sh
set -euo pipefail
cd "$(dirname "$0")"

# zxing-cpp requiere Python >= 3.10: buscamos un intérprete compatible.
ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; }
PY=""
for c in "${PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3 \
         /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 /usr/local/bin/python3.12; do
  [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && ok "$c" && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> No hay Python >= 3.10; instalando python@3.12 con Homebrew"
    brew install python@3.12
    PY="$(brew --prefix python@3.12)/bin/python3.12"
  else
    echo "Se requiere Python 3.10 o superior (instálalo, p.ej. con Homebrew: brew install python@3.12)."; exit 1
  fi
fi
echo "==> Usando $($PY --version) ($PY)"

echo "==> Creando entorno virtual (.venv)"
rm -rf .venv
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Instalando dependencias (puede tardar unos minutos)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo "==> Verificando OCR y lectores"
python - <<'PYEOF'
import cv2, zxingcpp, streamlit
from ine_pipeline.ocr import get_engine
get_engine()
print("OK: OpenCV", cv2.__version__, "| Streamlit", streamlit.__version__, "| OCR cargado")
PYEOF

echo
echo "Listo. Para abrir la interfaz ejecuta:"
echo "    python3 run.py"
