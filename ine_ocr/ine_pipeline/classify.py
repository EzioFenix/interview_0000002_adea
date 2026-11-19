"""Clasificación por evidencias: ¿esta imagen es una INE/IFE (anverso o reverso) o es otra cosa?

Sin modelos entrenados: puntaje ponderado de (a) geometría de tarjeta, (b) palabras clave con
coincidencia difusa, (c) patrones que solo una INE cumple (CURP válido, clave de elector, MRZ,
PDF417/QR), y (d) palabras que delatan otros documentos (licencia, contrato, escritura...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from . import validators as V
from .extract import find_clave, find_curp, extract_mrz
from .ocr import OcrLine

FRONT_STRONG = {"INSTITUTO NACIONAL ELECTORAL": 2.0, "INSTITUTO FEDERAL ELECTORAL": 2.0,
                "CREDENCIAL PARA VOTAR": 2.5, "CLAVE DE ELECTOR": 1.5}
FRONT_MEDIUM = {"DOMICILIO": 0.5, "SECCION": 0.7, "VIGENCIA": 0.7, "EMISION": 0.5, "ANO DE REGISTRO": 0.6,
                "MUNICIPIO": 0.4, "LOCALIDAD": 0.4, "FECHA DE NACIMIENTO": 0.4, "CURP": 0.6}
BACK_KEYS = {"INSTITUTO NACIONAL ELECTORAL": 1.5, "SECRETARIO EJECUTIVO": 1.5, "IDENTIFICADOR": 0.8,
             "INE.MX": 1.0, "CREDENCIAL": 0.4, "FIRMA": 0.3, "HUELLA": 0.3}
NEGATIVE = ["LICENCIA", "CONDUCIR", "ESCRITURA", "NOTARIO", "NOTARIA", "CLAUSULA", "CONTRATO",
            "COMPRAVENTA", "PASAPORTE", "CEDULA PROFESIONAL", "TARJETA DE CIRCULACION", "COMPARECEN",
            "CONSTAR", "DECLARAN", "LICENCE", "DRIVER"]

FRONT_THRESHOLD = 3.5
BACK_THRESHOLD = 3.0


@dataclass
class Classification:
    is_ine: bool
    side: str | None             # "front" | "back" | None
    front_score: float
    back_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self):
        return {"is_ine": self.is_ine, "side": self.side, "front_score": round(self.front_score, 2),
                "back_score": round(self.back_score, 2), "reasons": self.reasons}


def _has(text: str, kw: str, thr: int = 82) -> bool:
    if kw in text:
        return True
    return fuzz.partial_ratio(kw, text) >= thr if len(text) >= len(kw) - 2 else False


def classify(lines: list[OcrLine], barcodes: list[dict], card_found: bool, card_score: float) -> Classification:
    text = " ".join(V.norm(l.text) for l in lines)
    reasons: list[str] = []
    f = b = 0.0

    for kw, w in {**FRONT_STRONG, **FRONT_MEDIUM}.items():
        if _has(text, kw, 88 if len(kw) > 6 else 95):
            f += w
            reasons.append(f"+{w} anverso: «{kw}»")
    clave = find_clave(lines)
    curp = find_curp(lines, clave.value if clave else None)
    if curp:
        f += 2.0
        reasons.append("+2.0 anverso: CURP válido (regex + dígito verificador)")
    if clave:
        f += 1.5
        reasons.append("+1.5 anverso: clave de elector con formato válido")

    mrz, _ = extract_mrz(lines)
    if mrz:
        b += 3.0
        reasons.append("+3.0 reverso: MRZ de 3 líneas detectada")
    elif re.search(r"IDMEX|1DMEX|IDMEX", re.sub(r"[^A-Z0-9]", "", text)):
        b += 2.0
        reasons.append("+2.0 reverso: cadena «IDMEX» (MRZ parcial)")
    for kw, w in BACK_KEYS.items():
        if _has(text, kw, 88):
            b += w
            reasons.append(f"+{w} reverso: «{kw}»")
    for bc in barcodes:
        w = 2.5 if "PDF417" in bc["formato"] else 1.2
        b += w
        reasons.append(f"+{w} reverso: código {bc['formato']} decodificado")

    if card_found and card_score >= 0.5:
        f += 0.5
        b += 0.5
        reasons.append("+0.5 geometría de tarjeta ID-1 (proporción 1.586)")

    neg = [kw for kw in NEGATIVE if kw in text]
    if neg:
        pen = 2.0 * len(neg)
        f -= pen
        b -= pen
        reasons.append(f"-{pen} palabras de otro documento: {', '.join(neg)}")

    if f >= FRONT_THRESHOLD and f >= b:
        return Classification(True, "front", f, b, reasons)
    if b >= BACK_THRESHOLD:
        return Classification(True, "back", f, b, reasons)
    reasons.append(f"puntaje insuficiente (anverso {f:.1f}/{FRONT_THRESHOLD}, reverso {b:.1f}/{BACK_THRESHOLD})")
    return Classification(False, None, f, b, reasons)
