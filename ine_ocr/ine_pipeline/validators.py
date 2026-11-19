"""Validadores y correctores para CURP, Clave de Elector y MRZ (ICAO 9303 TD1).

Todo es determinista: no hay ningún modelo entrenado. Las reglas de validación
(regex del examen, dígito verificador, fechas, entidades) sirven para:
  1. elegir entre varias lecturas candidatas del OCR,
  2. corregir confusiones típicas (O/0, I/1, S/5, B/8...) según la posición,
  3. asignar una confianza a cada campo.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from itertools import product

# Regex EXACTAS del enunciado del examen
CURP_RE = re.compile(r"^[A-Z]{4}\d{6}[HM][A-Z]{5}[0-9A-Z]\d$")
CLAVE_RE = re.compile(r"^[A-Z]{6}\d{8}[HM]\d{3}$")

CURP_CHARS = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

# Entidades federativas (CURP) y su clave numérica (Clave de Elector 01-32, 33 = extranjero)
ENTIDADES_CURP = {
    "AS", "BC", "BS", "CC", "CL", "CM", "CS", "CH", "DF", "DG", "GT", "GR", "HG", "JC",
    "MC", "MN", "MS", "NT", "NL", "OC", "PL", "QT", "QR", "SP", "SL", "SR", "TC", "TS",
    "TL", "VZ", "YN", "ZS", "NE",
}

# Confusiones del OCR según el tipo de carácter que se espera en esa posición
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1", "Z": "2",
             "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}
_TO_SEX = {"N": "M", "W": "M", "K": "H", "4": "H"}

# Alternativas plausibles para reparar UN carácter mal leído (símétrico aproximado)
CONFUSABLE = {
    "O": "0QD", "0": "OQD", "Q": "O0", "D": "O0", "I": "1L", "1": "IL", "L": "I1T",
    "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6C", "6": "G",
    "C": "GO", "U": "V", "V": "UY", "M": "NH", "N": "MH", "H": "MN", "A": "4", "4": "A",
    "E": "F", "F": "E", "P": "R", "R": "P", "T": "7I", "7": "T", "Y": "V", "X": "K", "K": "X",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    """Mayúsculas sin acentos (la Ñ se conserva como N para comparar)."""
    return strip_accents(s).upper()


def alnum(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", norm(s))


def apply_classes(s: str, classes: str) -> str:
    """Corrige confusiones por posición. classes: L=letra, D=dígito, S=sexo H/M, A=alfanumérico."""
    out = []
    for ch, cl in zip(s, classes):
        if cl == "D" and not ch.isdigit():
            ch = _TO_DIGIT.get(ch, ch)
        elif cl == "L" and not ch.isalpha():
            ch = _TO_LETTER.get(ch, ch)
        elif cl == "S" and ch not in "HM":
            ch = _TO_SEX.get(ch, ch)
        out.append(ch)
    return "".join(out)


CURP_CLASSES = "LLLL" + "DDDDDD" + "S" + "LLLLL" + "A" + "D"
CLAVE_CLASSES = "LLLLLL" + "DDDDDDDD" + "S" + "DDD"


# --------------------------------------------------------------------------- CURP
def curp_check_digit(curp17: str) -> str | None:
    try:
        total = sum(CURP_CHARS.index(c) * (18 - i) for i, c in enumerate(curp17))
    except ValueError:
        return None
    return str((10 - total % 10) % 10)


def _valid_date(yy: int, mm: int, dd: int, century: int) -> bool:
    try:
        date(century + yy, mm, dd)
        return True
    except ValueError:
        return False


def curp_birth_date(curp: str) -> date | None:
    """Fecha de nacimiento. El carácter 17 distingue siglo: dígito = 1900s, letra = 2000s."""
    if len(curp) < 17 or not curp[4:10].isdigit():
        return None
    century = 1900 if curp[16].isdigit() else 2000
    yy, mm, dd = int(curp[4:6]), int(curp[6:8]), int(curp[8:10])
    return date(century + yy, mm, dd) if _valid_date(yy, mm, dd, century) else None


def validate_curp(curp: str) -> dict:
    """Devuelve {regex, checksum, date, entity, valid, score}. score 0..1."""
    res = {"regex": False, "checksum": False, "date": False, "entity": False}
    if len(curp) == 18 and CURP_RE.match(curp):
        res["regex"] = True
        res["checksum"] = curp_check_digit(curp[:17]) == curp[17]
        res["date"] = curp_birth_date(curp) is not None
        res["entity"] = curp[11:13] in ENTIDADES_CURP
    # Un CURP "válido" cumple regex + dígito verificador + fecha real
    res["valid"] = res["regex"] and res["checksum"] and res["date"]
    res["score"] = (0.4 * res["regex"] + 0.35 * res["checksum"] + 0.15 * res["date"]
                    + 0.10 * res["entity"])
    return res


def curp_sex(curp: str) -> str | None:
    return curp[10] if len(curp) > 10 and curp[10] in "HM" else None


# ----------------------------------------------------------------- Clave de elector
def validate_clave(clave: str) -> dict:
    res = {"regex": False, "date": False, "state": False, "valid": False, "score": 0.0}
    if len(clave) == 18 and CLAVE_RE.match(clave):
        res["regex"] = True
        yy, mm, dd = int(clave[6:8]), int(clave[8:10]), int(clave[10:12])
        res["date"] = _valid_date(yy, mm, dd, 1900) or _valid_date(yy, mm, dd, 2000)
        res["state"] = 1 <= int(clave[12:14]) <= 33
    res["valid"] = res["regex"] and res["date"] and res["state"]
    res["score"] = 0.5 * res["regex"] + 0.3 * res["date"] + 0.2 * res["state"]
    return res


def clave_birth_yymmdd(clave: str) -> str:
    return clave[6:12]


def consistent_curp_clave(curp: str, clave: str) -> bool | None:
    """Fecha de nacimiento y sexo deben coincidir entre CURP y Clave de Elector."""
    if not (curp and clave and len(curp) == 18 and len(clave) == 18):
        return None
    return curp[4:10] == clave[6:12] and curp[10] == clave[14]


# --------------------------------------------------------------- Reparación 1 carácter
def repair_candidates(s: str, classes: str, max_subs: int = 1):
    """Genera variantes con hasta `max_subs` sustituciones por confusiones plausibles."""
    yield s
    if max_subs < 1:
        return
    for i, ch in enumerate(s):
        for alt in CONFUSABLE.get(ch, ""):
            cand = s[:i] + alt + s[i + 1:]
            fixed = apply_classes(cand, classes)
            if fixed == cand:  # la sustitución respeta la clase de la posición
                yield cand


def repair_curp(s: str, clave: str | None = None) -> tuple[str, str | None] | None:
    """Intenta que `s` pase regex+checksum+fecha.

    Devuelve (curp, método) con método None (sin cambios), "sustitución" (1 carácter
    cambiado por confusión plausible) o "dígito recalculado" (los 17 primeros caracteres
    son plausibles y se recalcula el verificador). None si no hay forma.
    """
    base = apply_classes(s, CURP_CLASSES)
    if validate_curp(base)["valid"]:
        return base, None
    found = []
    for cand in repair_candidates(base, CURP_CLASSES, 1):
        if cand != base and validate_curp(cand)["valid"]:
            found.append(cand)
    if clave:
        agree = [c for c in found if consistent_curp_clave(c, clave)]
        found = agree or found
    if found:
        return found[0], "sustitución"
    # Último recurso: el dígito verificador es función de los 17 primeros caracteres
    if len(base) == 18 and CURP_RE.match(base):
        d = curp_check_digit(base[:17])
        cand = base[:17] + (d or "")
        if d and curp_birth_date(cand) is not None and (not clave or consistent_curp_clave(cand, clave) is not False):
            return cand, "dígito recalculado"
    return None


# ---------------------------------------------------------------------------- MRZ
def _mrz_val(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    return ord(c) - 55 if "A" <= c <= "Z" else 0


def mrz_check(data: str) -> str:
    w = (7, 3, 1)
    return str(sum(_mrz_val(c) * w[i % 3] for i, c in enumerate(data)) % 10)


_MRZ_FIX = str.maketrans({"«": "<", "‹": "<", "(": "<", "[": "<", "{": "<", " ": "<", "﹤": "<", "＜": "<"})


def clean_mrz_line(s: str) -> str:
    s = strip_accents(s).upper().translate(_MRZ_FIX)
    return re.sub(r"[^A-Z0-9<]", "", s)


def parse_mrz_td1(lines: list[str]) -> dict | None:
    """Parsea MRZ de 3 líneas x 30 (TD1). Devuelve dict con campos y verificaciones, o None."""
    if len(lines) != 3:
        return None
    l1, l2, l3 = [(clean_mrz_line(x) + "<" * 30)[:30] for x in lines]
    # corrección por posición (tipo de carácter esperado en cada zona de la MRZ TD1)
    l1 = apply_classes(l1, "LLLLL" + "D" * 10 + "AA" + "D" * 13)
    if l1[15:17] != "<<" and re.fullmatch(r"[<KCLEIS]{2}", l1[15:17]) and l1[17:].replace("<", "").isdigit():
        l1 = l1[:15] + "<<" + l1[17:]
    l2 = apply_classes(l2, "D" * 7 + "S" + "D" * 7 + "LLL" + "A" * 11 + "D")
    l3 = apply_classes(l3, "L" * 30)
    out = {"line1": l1, "line2": l2, "line3": l3}
    out["doc_type"] = l1[:2]
    out["country"] = l1[2:5]
    out["doc_number"] = l1[5:14].replace("<", "")
    out["doc_number_ok"] = mrz_check(l1[5:14]) == l1[14]
    out["optional1"] = l1[15:30].strip("<")
    # línea 2: nacimiento(6) chk sexo expira(6) chk nacionalidad(3) opcional(11) chk-compuesto
    b = apply_classes(l2[0:6], "DDDDDD")
    out["birth_yymmdd"] = b
    out["birth_ok"] = mrz_check(b) == apply_classes(l2[6], "D")
    out["sex"] = l2[7]
    e = apply_classes(l2[8:14], "DDDDDD")
    out["expiry_yymmdd"] = e
    out["expiry_ok"] = mrz_check(e) == apply_classes(l2[14], "D")
    out["nationality"] = l2[15:18]
    composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    out["composite_ok"] = mrz_check(composite) == l2[29]
    # línea 3: APELLIDOS<<NOMBRES
    parts = l3.split("<<")
    surname = parts[0].replace("<", " ").strip()
    given = " ".join(p.replace("<", " ").strip() for p in parts[1:] if p.strip("<"))
    out["surname"], out["given_names"] = surname, given
    checks = [out["doc_number_ok"], out["birth_ok"], out["expiry_ok"], out["composite_ok"]]
    out["checks_passed"] = sum(checks)
    out["valid"] = sum(checks) >= 3 and out["doc_type"].startswith("I")
    return out
