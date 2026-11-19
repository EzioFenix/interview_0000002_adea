"""Envoltura del OCR (RapidOCR / ONNX Runtime, modelos preentrenados incluidos; no se entrena nada)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from .validators import norm

_engine = None
_lock = threading.Lock()


@dataclass
class OcrLine:
    text: str
    score: float
    box: np.ndarray = field(repr=False)  # 4x2

    @property
    def x0(self): return float(self.box[:, 0].min())
    @property
    def x1(self): return float(self.box[:, 0].max())
    @property
    def y0(self): return float(self.box[:, 1].min())
    @property
    def y1(self): return float(self.box[:, 1].max())
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    @property
    def h(self):  # alto real del renglón (longitud del borde izquierdo), no el de la caja alineada a ejes
        return max(1.0, float(np.linalg.norm(self.box[3] - self.box[0])))
    @property
    def angle(self):
        d = self.box[1] - self.box[0]
        return float(np.degrees(np.arctan2(d[1], d[0])))
    @property
    def norm(self): return norm(self.text)


def get_engine():
    """Singleton: cargar los modelos una sola vez es la mayor ganancia de velocidad en lote."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                logging.getLogger("RapidOCR").setLevel(logging.ERROR)
                from rapidocr import RapidOCR
                _engine = RapidOCR()
    return _engine


def run_ocr(img_bgr: np.ndarray) -> list[OcrLine]:
    res = get_engine()(img_bgr)
    if res is None or getattr(res, "txts", None) is None or len(res.txts) == 0:
        return []
    out = []
    for box, txt, sc in zip(res.boxes, res.txts, res.scores):
        txt = (txt or "").strip()
        if txt:
            out.append(OcrLine(txt, float(sc), np.asarray(box, dtype=np.float32)))
    return out


def cluster_rows(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Agrupa cajas de texto en renglones visuales (misma altura) ordenados de izq. a der."""
    rows: list[list[OcrLine]] = []
    for ln in sorted(lines, key=lambda l: l.cy):
        for row in rows:
            ref = row[0]
            if abs(ln.cy - ref.cy) <= 0.55 * min(ln.h, ref.h) + 3:
                row.append(ln)
                break
        else:
            rows.append([ln])
    for r in rows:
        r.sort(key=lambda l: l.x0)
    rows.sort(key=lambda r: np.mean([l.cy for l in r]))
    return rows


def row_text(row: list[OcrLine]) -> str:
    return " ".join(l.text for l in row)


def median_text_angle(lines: list[OcrLine]) -> float:
    """Inclinación mediana del texto (solo cajas alargadas: ancho > 3× alto)."""
    angs = [l.angle for l in lines if (l.x1 - l.x0) > 3 * l.h and len(l.text) >= 4]
    return float(np.median(angs)) if len(angs) >= 3 else 0.0
