"""Evalúa el pipeline contra el banco sintético (con etiquetas) y reporta métricas por campo.

Uso:  python synth/evaluate.py --bank synth/banco [--workers 4] [--out synth/reporte.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRONT_FIELDS = ["curp", "clave_elector", "apellido_paterno", "apellido_materno", "nombres", "calle",
                "numero_exterior", "colonia", "codigo_postal", "municipio", "estado", "estado_clave",
                "municipio_clave", "seccion", "localidad", "emision", "vigencia", "anio_registro"]


def _n(v) -> str:
    from ine_pipeline.validators import norm
    return " ".join(norm(str(v or "")).replace(".", "").replace(",", "").split())


def _expected(field: str, datos: dict) -> str:
    v = datos.get(field)
    if field == "colonia":
        v = _n(v)
        for pre in ("COL ", "COLONIA "):
            v = v[len(pre):] if v.startswith(pre) else v
    return _n(v)


def _work(args):
    from ine_pipeline.pipeline import process_image
    path, = args
    r = process_image(path)
    return r.to_dict()


def evaluate(bank: str, workers: int = 4) -> dict:
    bank_p = Path(bank)
    labels = json.loads((bank_p / "labels.json").read_text(encoding="utf-8"))
    paths = [(str(bank_p / l["archivo"]),) for l in labels]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_work, paths))
    else:
        results = [_work(p) for p in paths]

    cls_ok = side_ok = 0
    n_cls = 0
    by_sev = defaultdict(lambda: [0, 0])
    field_stats = defaultdict(lambda: [0, 0])
    field_nospace = defaultdict(lambda: [0, 0])
    mrz_ok = mrz_tot = code_ok = code_tot = 0
    decisions = defaultdict(int)
    times = []
    failures = []
    for lab, res in zip(labels, results):
        n_cls += 1
        times.append(res["tiempos_s"].get("total", 0))
        decisions[(("INE" if lab["es_ine"] else "NEG"), res["decision"])] += 1
        correct_cls = res["es_ine"] == lab["es_ine"]
        cls_ok += correct_cls
        if not correct_cls:
            failures.append((lab["archivo"], "clasificación", res["motivo"]))
        if lab["es_ine"]:
            side_ok += res["lado"] == lab["lado"]
            by_sev[lab["severidad"]][1] += 1
            by_sev[lab["severidad"]][0] += res["decision"] == "OK"
            if lab["lado"] == "front":
                for f in FRONT_FIELDS:
                    got = _n(res["campos"].get(f, {}).get("value"))
                    exp = _expected(f, lab["datos"])
                    ok = got == exp
                    field_stats[f][0] += ok
                    field_stats[f][1] += 1
                    field_nospace[f][0] += got.replace(" ", "") == exp.replace(" ", "")
                    field_nospace[f][1] += 1
                    if not ok and f in ("curp", "clave_elector"):
                        failures.append((lab["archivo"], f, f"esperado {exp} / leído {got or '∅'}"))
            else:
                for i, key in enumerate(("mrz_linea1", "mrz_linea2", "mrz_linea3")):
                    got = res["campos"].get(key, {}).get("value", "")
                    mrz_tot += 1
                    mrz_ok += got == lab["mrz"][i]
                    if got != lab["mrz"][i]:
                        failures.append((lab["archivo"], key, f"esperado {lab['mrz'][i]} / leído {got or '∅'}"))
                code_tot += 1
                code_ok += res["campos"].get("codigo_texto", {}).get("value") == lab["codigo"]["texto"]
    summary = {
        "imagenes": len(labels),
        "clasificacion_es_ine_acierto": round(cls_ok / n_cls, 4),
        "lado_correcto_en_ine": round(side_ok / max(1, sum(1 for l in labels if l["es_ine"])), 4),
        "decision_OK_por_severidad": {k: f"{v[0]}/{v[1]}" for k, v in by_sev.items()},
        "campos_anverso_exactos": {k: f"{v[0]}/{v[1]}" for k, v in field_stats.items()},
        "campos_anverso_ignorando_espacios": {k: f"{v[0]}/{v[1]}" for k, v in field_nospace.items()
                                              if field_nospace[k][0] != field_stats[k][0]},
        "mrz_lineas_exactas": f"{mrz_ok}/{mrz_tot}",
        "codigo_barras_exacto": f"{code_ok}/{code_tot}",
        "decisiones": {f"{a}->{b}": n for (a, b), n in sorted(decisions.items())},
        "tiempo_medio_s": round(sum(times) / len(times), 2),
        "fallos_ejemplo": failures[:25],
    }
    return {"resumen": summary, "resultados": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="synth/banco")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out", default="synth/reporte.json")
    a = ap.parse_args()
    out = evaluate(a.bank, a.workers)
    Path(a.out).write_text(json.dumps(out["resumen"], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["resumen"], ensure_ascii=False, indent=1))
