#!/usr/bin/env python3
"""CLI del proceso por lotes.

  python cli.py procesar ./imagenes                       # guarda en SQLite local (local_store.db), 2 procesos
  python cli.py una foto.jpg                              # una sola imagen, imprime el JSON
  python cli.py ui                                        # abre la interfaz (modo test + lote)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Banco de imágenes → INE → datos estructurados → SQLite")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("procesar", help="procesa una carpeta (lote)")
    p.add_argument("carpeta")
    p.add_argument("--db", choices=["sqlite", "none"], default="sqlite",
                   help="sqlite = guarda en local_store.db; none = no persistir")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--sqlite", default="local_store.db")
    p.add_argument("--persistir-revision", action="store_true",
                   help="guardar en expedientes también los casos de baja confianza (por defecto solo audit log)")
    p.add_argument("--sin-recursion", action="store_true")
    p.add_argument("--salida", default="reporte_lote.csv", help="reporte .csv o .json")

    u = sub.add_parser("una", help="procesa una imagen e imprime el resultado JSON")
    u.add_argument("imagen")

    sub.add_parser("ui", help="abre la interfaz web (Streamlit)")

    a = ap.parse_args(argv)
    if a.cmd == "ui":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")])

    sys.path.insert(0, str(ROOT))
    if a.cmd == "una":
        from ine_pipeline.pipeline import process_image
        print(json.dumps(process_image(a.imagen).to_dict(), ensure_ascii=False, indent=1, default=str))
        return 0

    from ine_pipeline.batch import run_batch, write_report

    def progress(i, n, rec):
        print(f"[{i:>4}/{n}] {rec['decision']:<10} {rec.get('lado') or '-':<5} {rec['confianza']:.2f}  {Path(rec['archivo']).name}",
              flush=True)

    out = run_batch(a.carpeta, a.db, a.workers, a.sqlite, a.persistir_revision, not a.sin_recursion, progress)
    write_report(out, a.salida)
    print(f"\n{out['total']} imágenes en {out['segundos']} s → {out['resumen']}")
    if out["pares_frente_reverso"]:
        print(f"{len(out['pares_frente_reverso'])} pares anverso/reverso cruzados por identidad (MRZ ↔ CURP)")
    print(f"Reporte: {a.salida}" + (f" | BD: {a.sqlite}" if a.db == "sqlite" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
