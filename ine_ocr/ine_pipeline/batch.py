"""Procesamiento por lotes: carpeta de imágenes heterogéneas → separar INE → extraer → persistir."""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from .db import Persistor, build_persistor
from .pipeline import IMAGE_EXTS, pair_front_back, process_image


def _init_worker():
    from .ocr import get_engine
    get_engine()  # cargar modelos UNA vez por proceso


def _work(path: str) -> dict:
    try:
        return process_image(path).to_dict()
    except Exception as e:  # nunca tumbar el lote por una imagen
        return {"archivo": path, "sha256": "", "es_ine": False, "lado": None, "decision": "ERROR",
                "motivo": f"excepción: {type(e).__name__}: {e}", "confianza": 0.0, "clasificacion": None,
                "campos": {}, "codigos": [], "intentos": [], "tiempos_s": {}}


def list_images(folder: str | Path, recursive: bool = True) -> list[Path]:
    it = Path(folder).rglob("*") if recursive else Path(folder).glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def run_batch(folder: str | Path, db_mode: str = "sqlite", workers: int = 2, sqlite_path: str = "local_store.db",
              persist_review: bool = False, recursive: bool = True,
              progress: Callable[[int, int, dict], None] | None = None) -> dict:
    paths = list_images(folder, recursive)
    t0 = time.perf_counter()
    persistor: Persistor | None = build_persistor(db_mode, sqlite_path, persist_review)
    records: list[dict] = []
    acciones: list[dict] = []

    def handle(i: int, rec: dict):
        records.append(rec)
        acciones.append(persistor.persist(rec) if persistor else {"accion": "SIN_BD", "id": None})
        if progress:
            progress(i + 1, len(paths), rec)

    try:
        if workers > 1 and len(paths) > 1:
            with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
                for i, rec in enumerate(ex.map(_work, map(str, paths))):
                    handle(i, rec)
        else:
            for i, p in enumerate(paths):
                handle(i, _work(str(p)))
    finally:
        if persistor:
            persistor.close()

    resumen: dict[str, int] = {}
    for r in records:
        resumen[r["decision"]] = resumen.get(r["decision"], 0) + 1
    return {"total": len(paths), "resumen": resumen, "segundos": round(time.perf_counter() - t0, 1),
            "registros": records, "acciones": acciones, "pares_frente_reverso": pair_front_back(records)}


def write_report(out: dict, path: str | Path):
    """CSV (una fila por imagen, sin datos personales completos) o JSON completo según la extensión."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "decision", "es_ine", "lado", "confianza", "accion_bd", "motivo"])
        for r, a in zip(out["registros"], out["acciones"]):
            w.writerow([Path(r["archivo"]).name, r["decision"], r["es_ine"], r["lado"], r["confianza"],
                        a["accion"], r["motivo"]])
