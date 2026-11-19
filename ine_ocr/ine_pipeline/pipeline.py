"""Orquestador: Clasificación → Visión & ROI → Extracción → decisión de persistencia."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import vision
from .classify import Classification, classify
from .extract import Field, extract_front, extract_mrz, mrz_to_fields, needs_flip, read_barcodes
from .ocr import OcrLine, median_text_angle, run_ocr

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_INPUT_SIDE = 3000

# Umbrales de decisión autónoma (sin revisión humana)
MIN_CONF_OK = 0.75


@dataclass
class PipelineResult:
    path: str
    sha256: str
    is_ine: bool = False
    side: str | None = None
    classification: Classification | None = None
    fields: dict[str, Field] = field(default_factory=dict)
    barcodes: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    decision: str = "DESCARTADO"       # OK | REVISION | DESCARTADO | ERROR
    reason: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)   # imágenes / cajas para la interfaz (no se serializa)

    def fields_dict(self):
        return {k: v.to_dict() for k, v in self.fields.items()}

    def to_dict(self):
        return {"archivo": self.path, "sha256": self.sha256, "es_ine": self.is_ine, "lado": self.side,
                "decision": self.decision, "motivo": self.reason, "confianza": round(self.confidence, 3),
                "clasificacion": self.classification.to_dict() if self.classification else None,
                "campos": self.fields_dict(), "codigos": self.barcodes, "intentos": self.attempts,
                "tiempos_s": {k: round(v, 3) for k, v in self.timings.items()}}


def load_image(source) -> tuple[np.ndarray, str]:
    """Carga ruta / bytes / PIL / ndarray → BGR uint8 (respeta orientación EXIF) + sha256."""
    if isinstance(source, np.ndarray):
        img = source
        digest = hashlib.sha256(img.tobytes()).hexdigest()
    else:
        if isinstance(source, (str, Path)):
            raw = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        elif hasattr(source, "getvalue"):
            raw = source.getvalue()
        elif hasattr(source, "read"):
            raw = source.read()
        else:
            raise TypeError(f"Tipo de imagen no soportado: {type(source)}")
        digest = hashlib.sha256(raw).hexdigest()
        import io
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > MAX_INPUT_SIDE:
        s = MAX_INPUT_SIDE / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, digest


def _kw_score(lines: list[OcrLine]) -> float:
    t = " ".join(l.norm for l in lines)
    return sum(k in t for k in ("ELECTOR", "CURP", "INSTITUTO", "CREDENCIAL", "VOTAR", "DOMICILIO", "NOMBRE",
                                "IDMEX", "<<", "VIGENCIA")) + sum(l.score for l in lines) / max(1, len(lines)) * 0.5


def _quality(side: str | None, fields: dict[str, Field]) -> tuple[float, bool]:
    """Confianza global ponderada y si cumple los requisitos mínimos para persistir."""
    if side == "front":
        weights = {"curp": 0.35, "clave_elector": 0.25, "apellido_paterno": 0.10, "nombres": 0.10,
                   "domicilio_completo": 0.10, "seccion": 0.05, "vigencia": 0.05}
        tot = sum(w for k, w in weights.items())
        score = sum(w * fields[k].confidence * (1.0 if fields[k].valid is not False else 0.6)
                    for k, w in weights.items() if k in fields) / tot
        ok = (all(k in fields and fields[k].valid for k in ("curp", "clave_elector"))
              and "apellido_paterno" in fields and "nombres" in fields)
        return score, ok
    if side == "back":
        has_mrz = "mrz_checks" in fields and fields["mrz_checks"].valid
        has_bc = "codigo_texto" in fields
        mrz_c = fields["mrz_checks"].confidence if "mrz_checks" in fields else 0.0
        score = 0.95 if (has_mrz and has_bc) else 0.9 * max(mrz_c, 0.9) if has_mrz else 0.8 if has_bc else mrz_c * 0.5
        return score, bool(has_mrz or has_bc)
    return 0.0, False


def process_image(source, keep_debug: bool = False, name: str | None = None) -> PipelineResult:
    t_all = time.perf_counter()
    path = name or (str(source) if isinstance(source, (str, Path)) else "imagen")
    try:
        img, digest = load_image(source)
    except Exception as e:  # archivo corrupto / no es imagen
        return PipelineResult(path, "", decision="ERROR", reason=f"no se pudo leer la imagen: {e}")
    res = PipelineResult(path, digest)
    dbg = res.debug
    if keep_debug:
        dbg["original"] = img

    # ---------------------------------------------------------------- 2a. geometría de la tarjeta
    t = time.perf_counter()
    det = vision.detect_card(img)
    if det is not None:
        card = vision.warp_card(img, det.quad)
    else:
        card = vision.fallback_card(img)
    res.timings["geometria"] = time.perf_counter() - t
    if keep_debug:
        dbg["quad"] = None if det is None else det.quad
        dbg["deteccion"] = vision.draw_quad(img, None if det is None else det.quad)
        dbg["card_method"] = "ninguna (se usa la imagen completa)" if det is None else f"{det.method} (puntaje {det.score:.2f})"
        dbg["card"] = card

    # ---------------------------------------------------------------- 2b. limpieza visual
    t = time.perf_counter()
    enhanced = vision.enhance(card)
    binar = vision.adaptive_binarize(enhanced)
    res.timings["limpieza"] = time.perf_counter() - t
    if keep_debug:
        dbg["enhanced"], dbg["binarizada"] = enhanced, binar
        dbg["blur_score"] = vision.blur_score(card)

    # ---------------------------------------------------------------- 1/3. OCR + clasificación + extracción
    # Estrategia de reintentos: cada intento usa una variante de imagen distinta; se conserva el mejor.
    attempts_defs = [("mejorada", lambda: enhanced), ("original", lambda: card),
                     ("binarizada", lambda: binar), ("nitidez×1.5", lambda: vision.sharpen_upscale(enhanced))]
    best = None  # (quality, ok, side, cls, fields, lines, barcodes, variant_img, variant_name, mrz)
    t_ocr = time.perf_counter()
    rot = 0
    for idx, (vname, make) in enumerate(attempts_defs):
        var = make()
        vimg = var if rot == 0 else cv2.rotate(var, cv2.ROTATE_180)
        lines = run_ocr(vimg)
        if idx == 0 and needs_flip(lines, vimg.shape[0]):
            vimg, rot = cv2.rotate(var, cv2.ROTATE_180), 180
            lines = run_ocr(vimg)
        elif idx == 0 and _kw_score(lines) < 5:
            # ¿tarjeta al revés? probar a 180° y quedarse con la orientación con más palabras reconocibles
            alt = cv2.rotate(var, cv2.ROTATE_180)
            lines_alt = run_ocr(alt)
            if _kw_score(lines_alt) > _kw_score(lines):
                lines, vimg, rot = lines_alt, alt, 180
        ang = median_text_angle(lines) if idx == 0 else 0.0
        if abs(ang) > 1.2:
            # texto inclinado: enderezar TODAS las variantes y repetir el OCR una vez
            card, enhanced, binar, var = (vision.deskew(x, ang) for x in (card, enhanced, binar, var))
            # (las lambdas de attempts_defs leen estas variables al llamarse, así que ya usan las versiones enderezadas)
            vimg = var if rot == 0 else cv2.rotate(var, cv2.ROTATE_180)
            lines = run_ocr(vimg)
            if keep_debug:
                dbg["card"], dbg["enhanced"], dbg["binarizada"] = card, enhanced, binar
                dbg["deskew_deg"] = round(ang, 2)
        H, W = vimg.shape[:2]
        bcs = read_barcodes(vimg) if idx == 0 or not best or best[2] != "front" else best[6]
        cls = classify(lines, bcs, det is not None, det.score if det else 0.0)
        fields: dict[str, Field] = {}
        if cls.is_ine and cls.side == "front":
            fields = extract_front(lines, W, H)
        elif cls.is_ine and cls.side == "back":
            mrz, msc = extract_mrz(lines)
            if mrz:
                fields.update(mrz_to_fields(mrz, msc))
            if bcs:
                fields["codigo_formato"] = Field(bcs[0]["formato"], 1.0, True, "zxing")
                fields["codigo_texto"] = Field(bcs[0]["texto"], 1.0, bcs[0]["valido"], "zxing")
        q, ok = _quality(cls.side, fields)
        res.attempts.append({"variante": vname, "rotacion": rot, "lineas_ocr": len(lines),
                             "lado": cls.side, "calidad": round(q, 3), "cumple_minimos": ok})
        cand = (q + (0.5 if ok else 0.0) + (0.0 if cls.is_ine else -0.5) + 0.01 * (cls.front_score + cls.back_score),
                ok, cls.side, cls, fields, lines, bcs, vimg, vname, q)
        if best is None or cand[0] > best[0]:
            best = cand
        if not cls.is_ine:
            if idx >= 1:   # dos variantes coinciden en que no es INE → descartar sin gastar más OCR
                break
            continue
        complete = side_complete = (cls.side == "front") or ("mrz_checks" in fields and fields["mrz_checks"].valid) or idx >= 2
        if ok and q >= MIN_CONF_OK and complete:
            break          # ya hay un resultado fiable: no seguir reintentando
    res.timings["ocr_extraccion"] = time.perf_counter() - t_ocr

    _, ok, side, cls, fields, lines, bcs, vimg, vname, q = best
    res.classification, res.barcodes = cls, bcs
    res.is_ine, res.side, res.fields = cls.is_ine, side, fields
    res.confidence = q
    if keep_debug:
        dbg["ocr_lines"] = lines
        dbg["ocr_image"] = vimg
        dbg["variante"] = vname

    # ---------------------------------------------------------------- decisión autónoma
    if not cls.is_ine:
        res.decision, res.reason = "DESCARTADO", "no es una INE/IFE: " + (cls.reasons[-1] if cls.reasons else "")
    elif ok and q >= MIN_CONF_OK:
        res.decision, res.reason = "OK", f"campos clave válidos (confianza {q:.2f}, variante «{vname}»)"
    else:
        faltan = []
        if side == "front":
            for k in ("curp", "clave_elector", "apellido_paterno", "nombres"):
                if k not in fields or not fields[k].valid and k in ("curp", "clave_elector"):
                    faltan.append(k)
        else:
            faltan.append("MRZ válida / código de barras")
        res.decision = "REVISION"
        res.reason = "INE con baja confianza" + (f"; falta/inválido: {', '.join(faltan)}" if faltan else "") + f" (confianza {q:.2f})"
    res.timings["total"] = time.perf_counter() - t_all
    return res


def pair_front_back(records: list[dict]) -> list[dict]:
    """Cruce de identidad opcional entre registros (dicts de `to_dict()`): reverso (MRZ) ↔ anverso (CURP)
    por fecha de nacimiento, sexo y apellido."""
    fronts = [r for r in records if r.get("lado") == "front" and "curp" in r["campos"]]
    backs = [r for r in records if r.get("lado") == "back" and "mrz_nacimiento_yymmdd" in r["campos"]]
    pairs = []
    for b in backs:
        by = b["campos"]["mrz_nacimiento_yymmdd"]["value"]
        bs = b["campos"]["mrz_sexo"]["value"]
        ap = str(b["campos"].get("mrz_apellidos", {}).get("value") or "").split(" ")[0]
        for f in fronts:
            curp = f["campos"]["curp"]["value"]
            sex_ok = (bs in ("H", "M") and bs == curp[10]) or (bs == "F" and curp[10] == "M")
            if curp[4:10] == by and sex_ok:
                pat = str(f["campos"].get("apellido_paterno", {}).get("value") or "")
                pairs.append({"anverso": f["archivo"], "reverso": b["archivo"],
                              "apellido_coincide": bool(ap and pat and ap[:4] == pat.upper().replace("Ñ", "N")[:4])})
    return pairs
