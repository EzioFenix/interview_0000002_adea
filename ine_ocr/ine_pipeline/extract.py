"""Extracción de campos de INE/IFE (anverso y reverso) a partir de las cajas de texto del OCR.

Estrategia anclada en etiquetas ("NOMBRE", "DOMICILIO", "CURP"...) en lugar de coordenadas
fijas, porque cada modelo de credencial (C, D, E, F, G, H) acomoda los datos distinto.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from rapidfuzz import fuzz

from . import validators as V
from .ocr import OcrLine, cluster_rows, row_text

LABEL_WORDS = ("NOMBRE", "DOMICILIO", "CLAVE", "ELECTOR", "CURP", "ESTADO", "MUNICIPIO", "SECCION",
               "LOCALIDAD", "EMISION", "VIGENCIA", "REGISTRO", "SEXO", "FECHA", "NACIMIENTO", "ANO DE",
               "CREDENCIAL", "VOTAR", "INSTITUTO", "NACIONAL", "ELECTORAL", "MEXICO", "FEDERAL")


@dataclass
class Field:
    value: object
    confidence: float = 0.0
    valid: bool | None = None
    source: str = "ocr"
    note: str = ""

    def to_dict(self):
        v = self.value.isoformat() if isinstance(self.value, date) else self.value
        return {"value": v, "confidence": round(float(self.confidence), 3), "valid": self.valid,
                "source": self.source, "note": self.note}


def _mean(xs, default=0.0):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else default


def _is_label(l: OcrLine) -> bool:
    t = l.norm
    return any(w in t for w in LABEL_WORDS) and len(t) < 40


def _find_label(lines: list[OcrLine], word: str, thr: int = 84) -> OcrLine | None:
    best, best_sc = None, 0
    for l in lines:
        t = re.sub(r"[^A-Z ]", "", l.norm).strip()
        if not t or len(t) > len(word) + 12:
            continue
        sc = fuzz.partial_ratio(word, t) if len(t) >= len(word) - 2 else 0
        if t.startswith(word[:4]) and len(t) <= len(word) + 2:
            sc = max(sc, 90)
        if sc >= thr and sc > best_sc:
            best, best_sc = l, sc
    return best


# --------------------------------------------------------------------------- CURP / Clave
def _windows(text: str, n: int = 18):
    for i in range(0, max(0, len(text) - n + 1)):
        yield i, text[i:i + n]


def _candidate_strings(lines: list[OcrLine]):
    """Texto limpio por caja y por renglón, con la confianza OCR de donde viene."""
    seen = set()
    for l in lines:
        a = V.alnum(l.text)
        if len(a) >= 14 and a not in seen:
            seen.add(a)
            yield a, l.score
    for row in cluster_rows(lines):
        a = V.alnum(row_text(row))
        if len(a) >= 14 and a not in seen:
            seen.add(a)
            yield a, _mean([l.score for l in row])


def find_curp(lines: list[OcrLine], clave: str | None = None) -> Field | None:
    best: tuple[float, str, float, str | None] | None = None
    for text, sc in _candidate_strings(lines):
        for _, w in _windows(text):
            rep = V.repair_curp(w, clave)
            if not rep:
                continue
            curp, how = rep
            q = 1.0 if how is None else (0.85 if how == "sustitución" else 0.7)
            # preferir ventanas a continuación de la etiqueta CURP
            label_bonus = 0.05 if text[max(0, text.find(w) - 4):text.find(w)].endswith("CURP") else 0
            total = q + label_bonus + (0.1 if V.consistent_curp_clave(curp, clave or "") else 0)
            if best is None or total > best[0]:
                best = (total, curp, sc, how)
    if not best:
        return None
    _, curp, sc, how = best
    note = {None: "", "sustitución": "1 carácter corregido por confusión OCR (validado con dígito verificador)",
            "dígito recalculado": "dígito verificador recalculado: revisar último carácter"}[how]
    conf = sc * (1.0 if how is None else 0.88 if how == "sustitución" else 0.7)
    return Field(curp, conf, True, "ocr", note)


def find_clave(lines: list[OcrLine]) -> Field | None:
    best = None
    for text, sc in _candidate_strings(lines):
        for _, w in _windows(text):
            cand = V.apply_classes(w, V.CLAVE_CLASSES)
            val = V.validate_clave(cand)
            if val["valid"]:
                s = val["score"] + (0.05 if "ELECTOR" in text[:text.find(w) + 1][-12:] else 0)
                if best is None or s > best[0]:
                    best = (s, cand, sc)
    if not best:
        # intento tolerante: una sustitución de un carácter
        for text, sc in _candidate_strings(lines):
            for _, w in _windows(text):
                base = V.apply_classes(w, V.CLAVE_CLASSES)
                for cand in V.repair_candidates(base, V.CLAVE_CLASSES, 1):
                    if cand != base and V.validate_clave(cand)["valid"]:
                        return Field(cand, sc * 0.8, True, "ocr", "1 carácter corregido por confusión OCR")
        return None
    return Field(best[1], best[2], True, "ocr")


# ------------------------------------------------------------------------------ Nombre
def _text_ok(t: str) -> bool:
    letters = sum(c.isalpha() or c in " .'-" for c in t)
    return len(t) >= 2 and letters / len(t) >= 0.8


def _block_below(lines: list[OcrLine], label: OcrLine, stop: OcrLine | None, card_h: float,
                 max_rows: int, max_dy: float, x_tol: float) -> list[list[OcrLine]]:
    y_stop = (stop.y0 - 2) if stop is not None and stop.y0 > label.y1 else label.y1 + max_dy
    y_stop = min(y_stop, label.y1 + max_dy)
    cands = [l for l in lines if l is not label and l.cy > label.y1 - 2 and l.y0 < y_stop
             and l.x1 > label.x0 - x_tol and l.cx > label.x0 - x_tol]
    rows = cluster_rows(cands)
    return rows[:max_rows + 3]


def extract_name(lines: list[OcrLine], curp: str | None, W: float, H: float) -> dict[str, Field]:
    lab = _find_label(lines, "NOMBRE", 88)
    dom = _find_label(lines, "DOMICILIO", 84)
    if lab is None:
        return {}
    rows = _block_below(lines, lab, dom, H, 6, 0.32 * H, 0.04 * W)
    names = []
    for r in rows:
        r = [l for l in r if not _is_label(l) and _text_ok(l.text) and l.x0 > lab.x0 - 0.04 * W]
        if not r:
            continue
        txt = " ".join(l.text for l in r).strip()
        txt = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ .'-]", "", txt).strip()
        if len(txt) >= 2 and not re.search(r"\b(SEXO|FECHA|NACIMIENTO)\b", V.norm(txt)):
            names.append((txt.upper(), _mean([l.score for l in r])))
    names = names[:4]
    out: dict[str, Field] = {}
    if len(names) >= 3:
        pat, mat = names[0], names[1]
        nom = (" ".join(n[0] for n in names[2:]), _mean([n[1] for n in names[2:]]))
        note = ""
    elif len(names) == 2:
        pat, mat, nom = names[0], None, names[1]
        note = "solo 2 renglones: se asume sin apellido materno"
    elif len(names) == 1:
        pat, mat, nom, note = names[0], None, None, "solo 1 renglón de nombre"
    else:
        return {}
    for key, item in (("apellido_paterno", pat), ("apellido_materno", mat), ("nombres", nom)):
        if item:
            conf = item[1] * (0.85 if note else 1.0)
            out[key] = Field(item[0], conf, None, "ocr", note)
    if curp and len(curp) == 18:  # verificación cruzada con las iniciales del CURP
        if "apellido_paterno" in out:
            ok = V.norm(out["apellido_paterno"].value)[:1] == curp[0]
            out["apellido_paterno"].valid = ok
            if ok:
                out["apellido_paterno"].confidence = min(1.0, out["apellido_paterno"].confidence * 1.05)
            else:
                out["apellido_paterno"].note = (out["apellido_paterno"].note + " inicial no coincide con CURP").strip()
        if "apellido_materno" in out:
            ok = V.norm(out["apellido_materno"].value)[:1] == curp[2]
            out["apellido_materno"].valid = ok
    full = " ".join(out[k].value for k in ("nombres", "apellido_paterno", "apellido_materno") if k in out)
    out["nombre_completo"] = Field(full, _mean([f.confidence for f in out.values()]), None, "derivado")
    return out


# --------------------------------------------------------------------------- Domicilio
_STREET_PREFIX = re.compile(r"^(AV|CDA|BLVD|PRIV|CALLE|AVENIDA|CALZ|CAMINO|CARR|PROL|RET)(?=[A-Z]{4,})")


def _unglue(s: str) -> str:
    """El OCR a veces omite espacios: separa letra↔dígito y prefijos comunes de calle."""
    s = re.sub(r"(?<=[A-Z])(?=\d)|(?<=\d)(?=[A-Z])", " ", s)
    s = _STREET_PREFIX.sub(r"\1 ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_domicilio(lines: list[OcrLine], W: float, H: float) -> dict[str, Field]:
    dom = _find_label(lines, "DOMICILIO", 84)
    if dom is None:
        return {}
    stop = _find_label(lines, "CLAVE DE ELECTOR", 80) or _find_label(lines, "CURP", 88)
    rows = _block_below(lines, dom, stop, H, 3, 0.30 * H, 0.03 * W)
    texts = []
    for r in rows:
        r = [l for l in r if not _is_label(l) and l.x0 > dom.x0 - 0.03 * W]
        if not r:
            continue
        texts.append((_unglue(V.norm(" ".join(l.text for l in r)).strip()), _mean([l.score for l in r])))
        if len(texts) == 3:
            break
    if not texts:
        return {}
    out: dict[str, Field] = {}
    conf_all = _mean([t[1] for t in texts]) * 0.85
    raw = ", ".join(t[0] for t in texts)
    out["domicilio_completo"] = Field(raw, conf_all, None, "ocr", "parseo por heurística (formato variable)")

    all_txt = " ".join(t[0] for t in texts)
    cp = re.search(r"(?<!\d)(\d{5})(?!\d)", all_txt)
    if cp:
        out["codigo_postal"] = Field(cp.group(1), conf_all, True)
    l1 = texts[0][0]
    interior = re.search(r"\b(?:INT\.?|INTERIOR)\s*([A-Z0-9-]+)", l1)
    l1_clean = re.sub(r"\b(?:INT\.?|INTERIOR)\s*[A-Z0-9-]+", "", l1).strip()
    m = re.match(r"^(.*?)[\s,]*(?:NO\.?\s*|#\s*)?(\d+[A-Z]?(?:-\d+)?)\s*$", l1_clean)
    if m:
        out["calle"] = Field(m.group(1).strip(" ,"), texts[0][1], None)
        out["numero_exterior"] = Field(m.group(2), texts[0][1], None)
    else:
        out["calle"] = Field(l1_clean, texts[0][1], None)
    if interior:
        out["numero_interior"] = Field(interior.group(1), texts[0][1], None)
    if len(texts) >= 2:
        l2 = re.sub(r"(?<!\d)\d{5}(?!\d)", "", texts[1][0]).strip(" ,")
        l2 = re.sub(r"^(COL\.?|COLONIA|FRACC\.?|FRACCIONAMIENTO)(?:\s+|(?<=\.))", "", l2).strip()
        # el OCR suele pegar «COL» al nombre: se quita salvo nombres reales que empiezan con COL
        if re.match(r"^COL[A-Z]{3,}", l2) and not re.match(r"^(COLIN|COLIM|COLOR|COLON|COLOM|COLEG|COLUM|COLCH)", l2):
            l2 = l2[3:]
        if l2:
            out["colonia"] = Field(l2, texts[1][1], None)
    if len(texts) >= 3:
        l3 = re.sub(r"(?<!\d)\d{5}(?!\d)", "", texts[2][0]).strip(" ,")
        parts = [p.strip() for p in re.split(r",|\s{2,}", l3) if p.strip()]
        if len(parts) >= 2:
            out["municipio"] = Field(parts[0], texts[2][1], None)
            out["estado"] = Field(parts[-1].rstrip("."), texts[2][1], None)
        elif parts:
            out["municipio"] = Field(parts[0], texts[2][1], None)
    return out


# ------------------------------------------------------------------------- Electorales
_D = r"([0-9OQDILSBZ]{%d})"


def _digits(s: str) -> str:
    return V.apply_classes(s, "D" * len(s))


def _near_label(lines: list[OcrLine], label_word: str, n: int, thr: int = 84) -> str | None:
    """Busca n dígitos en cajas a la derecha o justo debajo de la caja de la etiqueta."""
    lab = _find_label(lines, label_word, thr)
    if lab is None:
        return None
    best = None
    for l in lines:
        if l is lab:
            continue
        right = abs(l.cy - lab.cy) < 0.7 * lab.h and l.x0 >= lab.x0
        below = 0 < l.cy - lab.cy < 2.6 * lab.h and l.x1 > lab.x0 and l.x0 < lab.x1 + 0.12 * (lab.x1 - lab.x0 + 200)
        if not (right or below):
            continue
        m = re.search(_D % n, V.norm(l.text))
        if m:
            dist = abs(l.x0 - lab.x0) + abs(l.cy - lab.cy) * 2
            if best is None or dist < best[0]:
                best = (dist, _digits(m.group(1)))
    return best[1] if best else None


def extract_electoral(lines: list[OcrLine]) -> dict[str, Field]:
    rows = cluster_rows(lines)
    txt = " | ".join(V.norm(row_text(r)) for r in rows)
    sc = _mean([l.score for l in lines])
    out: dict[str, Field] = {}

    specs = [("estado_clave", 2, r"\bESTADO", "ESTADO"), ("municipio_clave", 3, r"MUNICIPIO", "MUNICIPIO"),
             ("seccion", 4, r"SECC\w{0,4}", "SECCION"), ("localidad", 4, r"LOCALIDAD", "LOCALIDAD"),
             ("emision", 4, r"EMIS\w{0,4}", "EMISION"), ("vigencia", 4, r"VIGENCIA", "VIGENCIA"),
             ("anio_registro", 4, r"REGISTRO", "ANO DE REGISTRO")]
    for key, n, label_pat, label_word in specs:
        m = re.search(label_pat + r"\s*[:\-]?\s*" + (_D % n), txt)
        if m:
            out[key] = Field(_digits(m.group(1)), sc, True)
            continue
        v = _near_label(lines, label_word, n)
        if v:
            out[key] = Field(v, sc * 0.9, True, "ocr", "valor ubicado por posición respecto a la etiqueta")
    if "emision" in out and "vigencia" not in out:
        # etiqueta ilegible: buscar un año suelto plausible (emisión + 1..15), preferentemente a 10 años
        e = int(out["emision"].value)
        years = [int(m.group(0)) for l in lines if (m := re.fullmatch(r"\d{4}", re.sub(r"\s", "", l.text)))]
        cands = [y for y in years if e < y <= e + 15]
        if cands:
            y = min(cands, key=lambda v: abs(v - (e + 10)))
            out["vigencia"] = Field(str(y), sc * 0.7, True, "ocr", "etiqueta ilegible: año inferido por rango (emisión + 1..15)")
    if "emision" in out and "vigencia" in out:
        e, v = int(out["emision"].value), int(out["vigencia"].value)
        out["vigencia"].valid = 0 < v - e <= 15 and 2000 <= e <= date.today().year + 1
        out["vigencia"].note = "vigente" if v >= date.today().year else "vencida"
    return out


# -------------------------------------------------------------------------------- Front
def extract_front(lines: list[OcrLine], W: float, H: float) -> dict[str, Field]:
    fields: dict[str, Field] = {}
    clave = find_clave(lines)
    curp = find_curp(lines, clave.value if clave else None)
    if curp is not None:
        fields["curp"] = curp
        d = V.curp_birth_date(curp.value)
        if d:
            fields["fecha_nacimiento"] = Field(d, curp.confidence, True, "derivado de CURP")
        fields["sexo"] = Field(V.curp_sex(curp.value), curp.confidence, True, "derivado de CURP")
        fields["entidad_nacimiento"] = Field(curp.value[11:13], curp.confidence, curp.value[11:13] in V.ENTIDADES_CURP,
                                             "derivado de CURP")
    if clave is not None:
        fields["clave_elector"] = clave
        cons = V.consistent_curp_clave(curp.value, clave.value) if curp else None
        if cons is not None:
            clave.valid = bool(cons)
            clave.note = "coincide con CURP (fecha y sexo)" if cons else "NO coincide con CURP (fecha/sexo)"
            if not cons:
                clave.confidence *= 0.6
    fields.update(extract_name(lines, curp.value if curp else None, W, H))
    fields.update(extract_domicilio(lines, W, H))
    fields.update(extract_electoral(lines))
    return fields


# --------------------------------------------------------------------------------- Back
_DIG = "0-9OQDILSBZ"
_L1_RE = re.compile(rf"^[I1L]D[A-Z0]{{3}}[{_DIG}]{{8,}}")
_L2_RE = re.compile(rf"^[{_DIG}]{{7}}[MFHNK<][{_DIG}]{{6,7}}[A-Z0]{{3}}")
_L3_RE = re.compile(r"^[A-Z0<]*[A-Z0]<<[A-Z0]")


def _mrz_candidates(lines: list[OcrLine]) -> list[tuple[str, float]]:
    out = []
    for l in lines:
        c = V.clean_mrz_line(l.text)
        if len(c) >= 22:
            out.append((c, l.score))
    return out


def extract_mrz(lines: list[OcrLine]) -> tuple[dict | None, float]:
    cands = _mrz_candidates(lines)
    l1 = next(((c, s) for c, s in cands if _L1_RE.match(c)), None)
    l2 = next(((c, s) for c, s in cands if _L2_RE.match(c) and not _L1_RE.match(c)), None)
    l3 = next(((c, s) for c, s in cands if _L3_RE.match(c) and not _L1_RE.match(c) and not _L2_RE.match(c)
               and "IDENTIFICADOR" not in c), None)
    if not (l1 and l2 and l3):
        return None, 0.0
    mrz = V.parse_mrz_td1([l1[0], l2[0], l3[0]])
    return mrz, _mean([l1[1], l2[1], l3[1]])


def mrz_to_fields(mrz: dict, sc: float) -> dict[str, Field]:
    out = {"mrz_linea1": Field(mrz["line1"], sc, mrz["doc_number_ok"]),
           "mrz_linea2": Field(mrz["line2"], sc, mrz["birth_ok"] and mrz["expiry_ok"]),
           "mrz_linea3": Field(mrz["line3"], sc, None),
           "mrz_checks": Field(f"{mrz['checks_passed']}/4", sc, mrz["valid"], "derivado",
                               "dígitos verificadores ICAO 9303")}
    out["mrz_apellidos"] = Field(mrz["surname"], sc, None, "derivado de MRZ")
    out["mrz_nombres"] = Field(mrz["given_names"], sc, None, "derivado de MRZ")
    out["mrz_sexo"] = Field(mrz["sex"], sc, mrz["sex"] in ("M", "F", "H"), "derivado de MRZ")
    out["mrz_nacimiento_yymmdd"] = Field(mrz["birth_yymmdd"], sc, mrz["birth_ok"], "derivado de MRZ")
    out["mrz_vence_yymmdd"] = Field(mrz["expiry_yymmdd"], sc, mrz["expiry_ok"], "derivado de MRZ")
    out["mrz_cic"] = Field(mrz["doc_number"], sc, mrz["doc_number_ok"], "derivado de MRZ")
    return out


def read_barcodes(img_bgr: np.ndarray) -> list[dict]:
    """Decodifica PDF417 / QR con zxing-cpp probando varias versiones de la imagen."""
    import cv2
    import zxingcpp

    fmts = [zxingcpp.BarcodeFormat.PDF417, zxingcpp.BarcodeFormat.QRCode, zxingcpp.BarcodeFormat.MicroQRCode]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants = [img_bgr, gray, cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10),
                cv2.createCLAHE(2.0, (8, 8)).apply(gray)]
    found: dict[tuple, dict] = {}
    for v in variants:
        try:
            for r in zxingcpp.read_barcodes(v, formats=fmts, try_rotate=True, try_downscale=True):
                key = (str(r.format), r.text)
                found.setdefault(key, {"formato": str(r.format).split(".")[-1], "texto": r.text,
                                       "valido": bool(r.valid)})
        except Exception:
            continue
    return list(found.values())


_TOP_KEYS = ("INSTITUTO NACIONAL", "INSTITUTO FEDERAL", "CREDENCIAL PARA", "ELECTORAL", "MEXICO", "SECRETARIO EJECUTIVO")
_BOTTOM_KEYS = ("VIGENCIA", "EMISION", "SECCION", "LOCALIDAD", "ANO DE REGISTRO")


def needs_flip(lines: list[OcrLine], H: float) -> bool:
    """¿La tarjeta quedó de cabeza? El OCR corrige cada renglón por separado, así que se decide por
    GEOMETRÍA: el encabezado (INSTITUTO NACIONAL ELECTORAL...) va arriba; vigencia/sección/MRZ abajo."""
    top = [l.cy / H for l in lines if any(k in l.norm for k in _TOP_KEYS)]
    bottom = [l.cy / H for l in lines if any(k in l.norm for k in _BOTTOM_KEYS)
              or (len(V.clean_mrz_line(l.text)) >= 22 and V.clean_mrz_line(l.text).count("<") >= 2)]
    if not top or not bottom:
        return False
    return float(np.mean(top)) > float(np.mean(bottom)) + 0.05
