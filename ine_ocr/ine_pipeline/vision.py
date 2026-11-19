"""Visión clásica: detección de la tarjeta, corrección de perspectiva y limpieza para OCR."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CARD_RATIO = 85.6 / 54.0          # ISO/IEC 7810 ID-1 (INE/IFE)
WARP_W, WARP_H = 1280, int(round(1280 / CARD_RATIO))
DETECT_MAX_SIDE = 1000            # la detección corre en una versión reducida (rápido)


@dataclass
class CardDetection:
    quad: np.ndarray        # 4x2 en coordenadas de la imagen ORIGINAL, orden tl,tr,br,bl
    score: float
    method: str


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _long_short(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return max(w, h), max(1.0, min(w, h))


def _score(quad: np.ndarray, img_shape: tuple[int, int], convex_bonus: float) -> float:
    h, w = img_shape
    img_area = float(h * w)
    area = cv2.contourArea(quad.astype(np.float32))
    if area < 0.06 * img_area:
        return 0.0
    frac = area / img_area
    long_, short_ = _long_short(quad)
    ratio = long_ / short_
    aspect_fit = max(0.0, 1.0 - abs(ratio - CARD_RATIO) / 0.55)
    # tamaño ideal: la tarjeta ocupa entre ~15% y ~80% del cuadro; el cuadro completo es el fondo
    size_fit = min(1.0, frac / 0.35) if frac <= 0.8 else max(0.0, 1.0 - (frac - 0.8) / 0.15)
    # penalizar cuadriláteros pegados a los bordes de la imagen (suelen ser el fondo, no la tarjeta)
    mx, my = 0.015 * w, 0.015 * h
    touching = sum(p[0] <= mx or p[0] >= w - mx or p[1] <= my or p[1] >= h - my for p in quad)
    border_fit = 1.0 - 0.3 * touching
    return float((size_fit * 0.45 + aspect_fit * 0.55) * convex_bonus * max(0.0, border_fit) * (aspect_fit > 0.05))


def _candidates_from_mask(mask: np.ndarray, img_shape: tuple[int, int], tag: str):
    img_area = float(img_shape[0] * img_shape[1])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 0.06 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            q = order_quad(approx.reshape(4, 2))
            yield q, _score(q, img_shape, 1.0), f"{tag}:poly4"
        hull = cv2.convexHull(c)
        box = order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))
        yield box, _score(box, img_shape, 0.9), f"{tag}:minrect"


def detect_card(img_bgr: np.ndarray) -> CardDetection | None:
    """Busca el rectángulo de la tarjeta con varias estrategias y elige la mejor por área y proporción."""
    h0, w0 = img_bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h0, w0))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    h, w = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    masks = []
    edges = cv2.Canny(gray, 40, 120)
    masks.append(("canny", cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [("otsu", otsu), ("otsu_inv", cv2.bitwise_not(otsu))]
    _, sat = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(("sat", sat))
    # Relleno de las regiones cerradas por los bordes (la tarjeta completa, no solo su contorno)
    filled = cv2.morphologyEx(masks[0][1], cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    masks.append(("canny_filled", filled))

    best: tuple[np.ndarray, float, str] | None = None
    for tag, m in masks:
        for q, sc, meth in _candidates_from_mask(m, (h, w), tag):
            if best is None or sc > best[1]:
                best = (q, sc, meth)

    # La imagen ya viene recortada (proporción de tarjeta y casi todo el cuadro)
    if abs((max(h0, w0) / min(h0, w0)) - CARD_RATIO) < 0.12:
        full = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float32)
        full_small = full * scale
        sc_full = 0.6
        if best is None or best[1] < 0.55:
            best = (full_small, sc_full, "imagen_completa")

    if best is None or best[1] < 0.22:
        return None
    return CardDetection(best[0] / scale, best[1], best[2])


def warp_card(img_bgr: np.ndarray, quad: np.ndarray, out_w: int = WARP_W, out_h: int = WARP_H) -> np.ndarray:
    """Endereza la tarjeta (warp perspective) dejándola siempre apaisada."""
    tl, tr, br, bl = quad
    top, left = np.linalg.norm(tr - tl), np.linalg.norm(bl - tl)
    if left > top:  # el lado largo quedó vertical: rotar el orden de las esquinas
        quad = np.array([bl, tl, tr, br], np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def fallback_card(img_bgr: np.ndarray) -> np.ndarray:
    """Sin cuadrilátero fiable: usar la imagen entera reescalada a tamaño estándar."""
    h, w = img_bgr.shape[:2]
    if h > w:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(img_bgr, (WARP_W, WARP_H), interpolation=cv2.INTER_CUBIC)


def enhance(card_bgr: np.ndarray) -> np.ndarray:
    """Limpieza visual: reducción de ruido + CLAHE + corrección de iluminación. Devuelve BGR."""
    den = cv2.fastNlMeansDenoisingColored(card_bgr, None, 4, 4, 5, 15)
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Quitar gradientes de iluminación (divide por el fondo estimado)
    bg = cv2.medianBlur(cv2.dilate(l, np.ones((15, 15), np.uint8)), 31)
    l = cv2.normalize(cv2.divide(l, bg, scale=235), None, 0, 255, cv2.NORM_MINMAX)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def adaptive_binarize(card_bgr: np.ndarray) -> np.ndarray:
    """Binarización adaptativa (Gauss) → imagen BGR en blanco/negro para reintentos de OCR."""
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def sharpen_upscale(card_bgr: np.ndarray, factor: float = 1.5) -> np.ndarray:
    up = cv2.resize(card_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(up, (0, 0), 2.0)
    return cv2.addWeighted(up, 1.6, blur, -0.6, 0)


def blur_score(img_bgr: np.ndarray) -> float:
    """Varianza del Laplaciano: valores bajos = imagen borrosa."""
    return float(cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def draw_quad(img_bgr: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    out = img_bgr.copy()
    if quad is not None:
        th = max(2, int(max(out.shape[:2]) / 300))
        cv2.polylines(out, [quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 0), th)
        for p in quad.astype(int):
            cv2.circle(out, tuple(int(v) for v in p), th * 3, (0, 0, 255), -1)
    return out


def deskew(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota la imagen para enderezar texto inclinado `angle_deg` grados (horario positivo)."""
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
