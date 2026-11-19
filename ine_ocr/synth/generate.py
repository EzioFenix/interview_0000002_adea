"""Generador de banco de imágenes SINTÉTICO (datos 100% ficticios) para probar el pipeline.

Dibuja credenciales tipo INE (anverso y reverso) con datos válidos (CURP con dígito verificador,
clave de elector coherente, MRZ con dígitos ICAO, PDF417/QR) y las fotografía "virtualmente":
perspectiva, rotación, sombras, ruido, desenfoque, reflejos y compresión JPEG. También genera
negativos: licencia, contrato, escritura notarial, tarjeta bancaria y fotos sin texto.

IMPORTANTE: no sustituye a imágenes reales; sirve para verificar la lógica y medir regresiones.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ine_pipeline import validators as V  # noqa: E402

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
CARD_W, CARD_H = 1011, 638

PATERNOS = ["GARCIA", "HERNANDEZ", "MARTINEZ", "LOPEZ", "GONZALEZ", "RODRIGUEZ", "PEREZ", "SANCHEZ", "RAMIREZ",
            "TORRES", "FLORES", "RIVERA", "GOMEZ", "DIAZ", "REYES", "MORALES", "CRUZ", "ORTIZ", "GUTIERREZ",
            "CHAVEZ", "RAMOS", "MENDOZA", "RUIZ", "ALVAREZ", "CASTILLO", "JIMENEZ", "VAZQUEZ", "ROMERO", "HERRERA",
            "MEDINA", "AGUILAR", "VARGAS", "CASTRO", "SOTO", "DELGADO", "SALAZAR", "BARRERA", "CONTRERAS"]
NOMBRES_H = ["JOSE LUIS", "JUAN CARLOS", "MIGUEL ANGEL", "VICTOR MANUEL", "FRANCISCO", "ALEJANDRO", "ROBERTO",
             "ERNESTO", "DANIEL", "ARTURO", "RICARDO", "SERGIO", "JORGE", "EDUARDO", "PABLO", "RAUL", "ADRIAN"]
NOMBRES_M = ["MARIA GUADALUPE", "ANA LAURA", "LAURA PATRICIA", "CLAUDIA", "ADRIANA", "VERONICA", "GABRIELA",
             "ROSA ELENA", "ALEJANDRA", "DIANA", "FERNANDA", "PAOLA", "LUCIA", "MONICA", "SILVIA", "TERESA"]
CALLES = ["AV INSURGENTES SUR", "C REFORMA", "C 5 DE MAYO", "AV JUAREZ", "C HIDALGO", "CDA DE LAS FLORES",
          "AV UNIVERSIDAD", "C MORELOS", "PRIV LOS PINOS", "AV TOLUCA", "C ALLENDE", "BLVD LAS TORRES"]
COLONIAS = ["COL CENTRO", "COL DEL VALLE", "COL ROMA NORTE", "COL SAN MATEO", "COL LAS AGUILAS", "COL NAPOLES",
            "COL JARDINES", "COL SANTA FE", "COL LOMAS VERDES", "COL AMERICAS"]
MUNICIPIOS = [("TOLUCA", "MEX.", "MC", 15), ("BENITO JUAREZ", "CDMX", "DF", 9), ("GUADALAJARA", "JAL.", "JC", 14),
              ("MONTERREY", "N.L.", "NL", 19), ("PUEBLA", "PUE.", "PL", 21), ("QUERETARO", "QRO.", "QT", 22),
              ("MERIDA", "YUC.", "YN", 31), ("LEON", "GTO.", "GT", 11), ("XALAPA", "VER.", "VZ", 30)]


def _font(bold=False, size=28, mono=False, cond=False):
    name = "DejaVuSansMono-Bold.ttf" if mono and bold else "DejaVuSansMono.ttf" if mono else \
        ("DejaVuSansCondensed-Bold.ttf" if bold else "DejaVuSansCondensed.ttf") if cond else \
        ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default(size)


# ------------------------------------------------------------------------------ datos ficticios
def _first_internal(s: str, vowel: bool) -> str:
    for c in s[1:]:
        if c.isalpha() and ((c in "AEIOU") == vowel):
            return c
    return "X"


def _consonants(s: str) -> str:
    return "".join(c for c in s if c.isalpha() and c not in "AEIOU")


def fake_person(rng: random.Random) -> dict:
    sex = rng.choice(["H", "M"])
    pat, mat = rng.sample(PATERNOS, 2)
    nom = rng.choice(NOMBRES_H if sex == "H" else NOMBRES_M)
    born = date(1950, 1, 1) + timedelta(days=rng.randint(0, 365 * 55))
    mun, est_abbr, ent_curp, ent_num = rng.choice(MUNICIPIOS)
    nac_ent = rng.choice(sorted(V.ENTIDADES_CURP - {"NE"}))
    first_name = nom.split()[0] if nom.split()[0] not in ("JOSE", "MARIA") or len(nom.split()) == 1 else nom.split()[1]
    yymmdd = born.strftime("%y%m%d")
    homo = str(rng.randint(0, 9)) if born.year < 2000 else rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    c17 = (pat[0] + _first_internal(pat, True) + mat[0] + first_name[0] + yymmdd + sex + nac_ent
           + _first_internal(pat, False) + _first_internal(mat, False) + _first_internal(first_name, False) + homo)
    curp = c17 + V.curp_check_digit(c17)
    cons = (_consonants(pat) + _consonants(mat) + _consonants(first_name) + "XXXXXX")[:6]
    clave = f"{cons}{yymmdd}{rng.randint(1, 32):02d}{sex}{rng.randint(0, 999):03d}"
    emision = rng.choice([2014, 2016, 2018, 2019, 2020, 2022, 2023])
    num = rng.randint(1, 4999)
    return {
        "apellido_paterno": pat, "apellido_materno": mat, "nombres": nom, "sexo": sex,
        "fecha_nacimiento": born.isoformat(), "curp": curp, "clave_elector": clave,
        "calle": rng.choice(CALLES), "numero_exterior": str(num),
        "colonia": rng.choice(COLONIAS), "codigo_postal": f"{rng.randint(1000, 99999):05d}",
        "municipio": mun, "estado": est_abbr,
        "estado_clave": f"{ent_num:02d}", "municipio_clave": f"{rng.randint(1, 120):03d}",
        "seccion": f"{rng.randint(1, 4999):04d}", "localidad": f"{rng.randint(1, 60):04d}",
        "emision": str(emision), "vigencia": str(emision + 10), "anio_registro": str(emision - rng.randint(0, 6)),
        "cic": f"{rng.randint(10**8, 10**9 - 1)}", "ocr_id": f"{rng.randint(10**12, 10**13 - 1)}",
    }


def build_mrz(p: dict) -> list[str]:
    l1 = "IDMEX" + p["cic"]
    l1 += V.mrz_check(p["cic"]) + "<<" + p["ocr_id"]
    born = p["fecha_nacimiento"][2:4] + p["fecha_nacimiento"][5:7] + p["fecha_nacimiento"][8:10]
    exp = p["vigencia"][2:] + "1231"
    l2 = born + V.mrz_check(born) + p["sexo"] + exp + V.mrz_check(exp) + "MEX" + "<" * 11
    composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    l2 += V.mrz_check(composite)
    sur = (p["apellido_paterno"] + "<" + p["apellido_materno"]).replace(" ", "<")
    l3 = (sur + "<<" + p["nombres"].replace(" ", "<"))[:30].ljust(30, "<")
    assert len(l1) == len(l2) == 30, (l1, l2)
    return [l1, l2, l3]


# ------------------------------------------------------------------------------ render tarjetas
def _gradient(w, h, c1, c2, rng) -> Image.Image:
    a = np.linspace(0, 1, w)[None, :, None] * np.ones((h, 1, 1))
    b = np.linspace(0, 1, h)[:, None, None] * np.ones((1, w, 1))
    t = 0.6 * a + 0.4 * b
    arr = (np.array(c1) * (1 - t) + np.array(c2) * t)
    arr += rng.normalvariate(0, 1) * 0 + np.random.default_rng(rng.randint(0, 10**6)).normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_front(p: dict, rng: random.Random) -> Image.Image:
    img = _gradient(CARD_W, CARD_H, (236, 224, 236), (222, 236, 226), rng)
    d = ImageDraw.Draw(img)
    for i in range(0, CARD_W, 36):  # patrón de seguridad tenue
        d.line([(i, 0), (i - 200, CARD_H)], fill=(214, 206, 222), width=1)
    d.rectangle([0, 0, CARD_W, 105], fill=(150, 40, 90))
    d.text((30, 20), "MÉXICO", font=_font(True, 34), fill="white")
    d.text((250, 18), "INSTITUTO NACIONAL ELECTORAL", font=_font(True, 30, cond=True), fill="white")
    d.text((560, 60), "CREDENCIAL PARA VOTAR", font=_font(True, 28, cond=True), fill="white")
    # foto
    d.rectangle([40, 140, 300, 470], fill=(160, 160, 165), outline=(90, 90, 90), width=3)
    d.ellipse([105, 190, 235, 340], fill=(205, 175, 150))
    d.ellipse([70, 340, 270, 540], fill=(70, 80, 110))
    lab, val = _font(False, 19, cond=True), _font(True, 30, cond=True)
    X = 335
    d.text((X, 125), "NOMBRE", font=lab, fill=(60, 60, 60))
    d.text((X, 150), p["apellido_paterno"], font=val, fill=(20, 20, 20))
    d.text((X, 186), p["apellido_materno"], font=val, fill=(20, 20, 20))
    d.text((X, 222), p["nombres"], font=val, fill=(20, 20, 20))
    d.text((820, 125), "SEXO", font=lab, fill=(60, 60, 60))
    d.text((890, 120), p["sexo"], font=val, fill=(20, 20, 20))
    d.text((X, 275), "DOMICILIO", font=lab, fill=(60, 60, 60))
    d.text((X, 300), f"{p['calle']} {p['numero_exterior']}", font=_font(True, 26, cond=True), fill=(20, 20, 20))
    d.text((X, 332), f"{p['colonia']} {p['codigo_postal']}", font=_font(True, 26, cond=True), fill=(20, 20, 20))
    d.text((X, 364), f"{p['municipio']}, {p['estado']}", font=_font(True, 26, cond=True), fill=(20, 20, 20))
    d.text((X, 412), "CLAVE DE ELECTOR", font=lab, fill=(60, 60, 60))
    d.text((X + 215, 405), p["clave_elector"], font=_font(True, 29, mono=True), fill=(20, 20, 20))
    d.text((X, 452), "CURP", font=lab, fill=(60, 60, 60))
    d.text((X + 215, 445), p["curp"], font=_font(True, 29, mono=True), fill=(20, 20, 20))
    d.text((X, 492), "ESTADO", font=lab, fill=(60, 60, 60))
    d.text((X + 95, 487), p["estado_clave"], font=val, fill=(20, 20, 20))
    d.text((X + 170, 492), "MUNICIPIO", font=lab, fill=(60, 60, 60))
    d.text((X + 285, 487), p["municipio_clave"], font=val, fill=(20, 20, 20))
    d.text((X + 400, 492), "SECCIÓN", font=lab, fill=(60, 60, 60))
    d.text((X + 495, 487), p["seccion"], font=val, fill=(20, 20, 20))
    d.text((X, 532), "LOCALIDAD", font=lab, fill=(60, 60, 60))
    d.text((X + 120, 527), p["localidad"], font=val, fill=(20, 20, 20))
    d.text((X + 215, 532), "EMISIÓN", font=lab, fill=(60, 60, 60))
    d.text((X + 310, 527), p["emision"], font=val, fill=(20, 20, 20))
    d.text((X + 400, 532), "VIGENCIA", font=lab, fill=(60, 60, 60))
    d.text((X + 510, 527), p["vigencia"], font=val, fill=(20, 20, 20))
    d.text((40, 560), "AÑO DE REGISTRO", font=lab, fill=(60, 60, 60))
    d.text((40, 585), f"{p['anio_registro']} 00", font=_font(True, 26, cond=True), fill=(20, 20, 20))
    return img


def render_back(p: dict, rng: random.Random, use_qr: bool = False) -> tuple[Image.Image, list[str], dict]:
    import zxingcpp
    img = _gradient(CARD_W, CARD_H, (232, 232, 240), (236, 226, 226), rng)
    d = ImageDraw.Draw(img)
    d.text((30, 25), "INSTITUTO NACIONAL ELECTORAL", font=_font(True, 28, cond=True), fill=(110, 30, 70))
    d.text((30, 70), "CREDENCIAL PARA VOTAR", font=_font(False, 22, cond=True), fill=(70, 70, 70))
    d.text((620, 25), "SECRETARIO EJECUTIVO", font=_font(False, 20, cond=True), fill=(60, 60, 60))
    d.text((620, 55), "www.ine.mx", font=_font(False, 22, cond=True), fill=(60, 60, 60))
    d.text((30, 130), "IDENTIFICADOR DE CIUDADANO", font=_font(False, 18, cond=True), fill=(70, 70, 70))
    payload = f"INE|{p['cic']}|{p['ocr_id']}|{p['curp']}|{p['vigencia']}"
    if use_qr:
        bc = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.QRCode)
        code = Image.fromarray(np.asarray(bc.to_image(scale=7))).convert("RGB")
    else:
        bc = zxingcpp.create_barcode(payload, zxingcpp.BarcodeFormat.PDF417)
        code = Image.fromarray(np.asarray(bc.to_image(scale=4))).convert("RGB")
        code = code.resize((min(code.width, 600), 150), Image.NEAREST)
    img.paste(code, (CARD_W - code.width - 40 if use_qr else 30, 165))
    mrz = build_mrz(p)
    f = _font(False, 40, mono=True)
    for i, line in enumerate(mrz):
        d.text((50, 400 + i * 62), line, font=f, fill=(15, 15, 15))
    return img, mrz, {"formato": "QRCode" if use_qr else "PDF417", "texto": payload}


def render_license(p: dict, rng) -> Image.Image:
    img = _gradient(CARD_W, CARD_H, (214, 232, 245), (245, 240, 214), rng)
    d = ImageDraw.Draw(img)
    d.text((40, 30), "ESTADOS UNIDOS MEXICANOS", font=_font(True, 30), fill=(20, 40, 100))
    d.text((40, 75), "LICENCIA PARA CONDUCIR", font=_font(True, 38), fill=(20, 40, 100))
    d.rectangle([40, 140, 280, 430], fill=(170, 170, 170))
    for i, t in enumerate([f"NOMBRE: {p['nombres']} {p['apellido_paterno']}", f"LICENCIA NO. {rng.randint(10**7, 10**8)}",
                           f"TIPO: AUTOMOVILISTA", f"VIGENCIA: {p['vigencia']}", f"DOMICILIO: {p['calle']} {p['numero_exterior']}",
                           f"{p['colonia']}", f"FECHA DE NACIMIENTO {p['fecha_nacimiento']}"]):
        d.text((320, 150 + i * 48), t, font=_font(False, 28), fill=(10, 10, 10))
    return img


def render_bank_card(p: dict, rng) -> Image.Image:
    img = _gradient(CARD_W, CARD_H, (20, 40, 90), (90, 20, 120), rng)
    d = ImageDraw.Draw(img)
    d.text((60, 60), "BANCO EJEMPLO", font=_font(True, 40), fill="white")
    num = " ".join(f"{rng.randint(0, 9999):04d}" for _ in range(4))
    d.text((60, 330), num, font=_font(True, 56, mono=True), fill="white")
    d.text((60, 480), f"{p['nombres']} {p['apellido_paterno']}", font=_font(False, 32), fill="white")
    d.text((700, 480), "VENCE 08/29", font=_font(False, 28), fill="white")
    return img


def render_document(kind: str, p: dict, rng) -> Image.Image:
    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), (250, 250, 247))
    d = ImageDraw.Draw(img)
    if kind == "contrato":
        title = "CONTRATO DE PRESTACION DE SERVICIOS"
        paras = ["CLAUSULA PRIMERA. El prestador se obliga a proporcionar los servicios descritos en el anexo uno.",
                 "CLAUSULA SEGUNDA. El cliente pagara la cantidad pactada dentro de los treinta dias naturales.",
                 "CLAUSULA TERCERA. Las partes declaran que no existe dolo ni mala fe en la celebracion del contrato.",
                 "CLAUSULA CUARTA. Para la interpretacion del presente contrato se someten a los tribunales competentes."]
    else:
        title = "ESCRITURA PUBLICA NUMERO " + str(rng.randint(1000, 9999))
        paras = ["En la ciudad de Toluca, ante mi, el Notario Publico, comparecen las partes para hacer constar",
                 "la compraventa del inmueble ubicado en " + p["calle"] + " numero " + p["numero_exterior"] + ".",
                 "Los comparecientes declaran bajo protesta de decir verdad que el inmueble se encuentra libre de gravamen.",
                 "Doy fe. NOTARIA PUBLICA NUMERO 12. Firman al calce los comparecientes."]
    d.text((120, 120), title, font=_font(True, 34), fill=(10, 10, 10))
    y = 220
    for para in paras * 3:
        words, line = para.split(), ""
        for w in words:
            if d.textlength(line + " " + w, font=_font(False, 26)) > W - 240:
                d.text((120, y), line, font=_font(False, 26), fill=(20, 20, 20))
                y += 40
                line = w
            else:
                line = (line + " " + w).strip()
        d.text((120, y), line, font=_font(False, 26), fill=(20, 20, 20))
        y += 70
    return img


def render_noise_photo(rng) -> Image.Image:
    arr = np.random.default_rng(rng.randint(0, 10**6))
    base = arr.normal(128, 40, (900, 1200, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(6))
    d = ImageDraw.Draw(img)
    for _ in range(8):
        x, y = rng.randint(0, 1000), rng.randint(0, 700)
        d.ellipse([x, y, x + rng.randint(40, 300), y + rng.randint(40, 300)],
                  fill=tuple(rng.randint(30, 220) for _ in range(3)))
    return img.filter(ImageFilter.GaussianBlur(3))


# ------------------------------------------------------------------------------ "fotografiar"
def photograph(card: Image.Image, rng: random.Random, severity: str = "medium", canvas=(1600, 1200),
               rot_choices=(0,), scale_range=(0.55, 0.8)) -> Image.Image:
    """Coloca la imagen sobre un fondo con textura, con perspectiva, ruido, desenfoque, reflejo y JPEG."""
    np_rng = np.random.default_rng(rng.randint(0, 10**9))
    sev = {"easy": 0.35, "medium": 0.8, "hard": 1.4}[severity]
    W, H = canvas
    bg_col = np.array([rng.randint(50, 200) for _ in range(3)], np.float32)
    bg = np.ones((H, W, 3), np.float32) * bg_col
    bg += np_rng.normal(0, 10, (H, W, 3))
    for _ in range(25):  # vetas/manchas del fondo
        cv2.line(bg, (rng.randint(0, W), rng.randint(0, H)), (rng.randint(0, W), rng.randint(0, H)),
                 tuple(float(c) for c in (bg_col + np_rng.normal(0, 18, 3))), rng.randint(1, 9))
    bg = cv2.GaussianBlur(bg, (0, 0), 2)

    card = card.rotate(rng.choice(rot_choices), expand=True)
    cw, ch = card.size
    s = rng.uniform(*scale_range) * min(W / cw, H / ch) * (1 / (1 + 0.1 * sev))
    card_arr = np.asarray(card.resize((int(cw * s), int(ch * s)), Image.LANCZOS))
    h, w = card_arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = W / 2 + rng.uniform(-0.05, 0.05) * W, H / 2 + rng.uniform(-0.05, 0.05) * H
    ang = np.deg2rad(rng.uniform(-25, 25) * sev)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = (src - [w / 2, h / 2]) @ R.T + [cx, cy]
    pts += np_rng.uniform(-0.04, 0.04, (4, 2)) * [w, h] * sev      # perspectiva
    M = cv2.getPerspectiveTransform(src, pts.astype(np.float32))
    warped = cv2.warpPerspective(card_arr, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    shadow = cv2.GaussianBlur(cv2.warpPerspective(np.full((h, w), 255, np.uint8),
                                                  M + np.array([[0, 0, 12], [0, 0, 12], [0, 0, 0]]) * 0, (W, H)), (0, 0), 14)
    out = bg * (1 - 0.35 * (shadow[..., None] / 255.0)) * 1.0
    m3 = (mask[..., None] / 255.0)
    out = out * (1 - m3) + warped.astype(np.float32) * m3
    # iluminación no uniforme + reflejo
    gx = np.linspace(1 - 0.18 * sev, 1 + 0.12 * sev, W)[None, :, None]
    out *= gx
    if rng.random() < 0.5 * sev:
        glare = np.zeros((H, W), np.float32)
        cv2.ellipse(glare, (int(cx + rng.uniform(-w / 4, w / 4)), int(cy + rng.uniform(-h / 4, h / 4))),
                    (int(w * 0.12), int(h * 0.08)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 35)
        out = out + glare[..., None] * 110 * sev
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.5) * sev)
    out += np_rng.normal(0, 4 * sev, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    pil = Image.fromarray(out)
    return pil


def save_jpeg(img: Image.Image, path: Path, rng: random.Random, severity: str):
    q = {"easy": rng.randint(80, 92), "medium": rng.randint(60, 85), "hard": rng.randint(40, 70)}[severity]
    img.save(path, "JPEG", quality=q)


# ------------------------------------------------------------------------------ dataset
def make_dataset(out_dir: str | Path, n: int = 12, seed: int = 7, severities=("easy", "medium", "hard")) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    labels = []

    def put(img, name, label, sev, **kw):
        path = out_dir / name
        save_jpeg(photograph(img, rng, sev, **kw), path, rng, sev)
        labels.append({"archivo": name, "severidad": sev, **label})

    for i in range(n):
        sev = severities[i % len(severities)]
        p = fake_person(rng)
        put(render_front(p, rng), f"ine_frente_{i:03d}_{sev}.jpg",
            {"es_ine": True, "lado": "front", "datos": p}, sev,
            rot_choices=(0, 0, 0, 180) if sev != "easy" else (0,))
        back, mrz, bc = render_back(p, rng, use_qr=(i % 3 == 2))
        put(back, f"ine_reverso_{i:03d}_{sev}.jpg",
            {"es_ine": True, "lado": "back", "datos": p, "mrz": mrz, "codigo": bc}, sev,
            rot_choices=(0, 0, 90, 270))
    for i in range(max(4, n // 2)):
        sev = severities[i % len(severities)]
        p = fake_person(rng)
        put(render_license(p, rng), f"neg_licencia_{i:03d}.jpg", {"es_ine": False, "tipo": "licencia"}, sev)
        put(render_bank_card(p, rng), f"neg_tarjeta_{i:03d}.jpg", {"es_ine": False, "tipo": "tarjeta"}, sev)
        put(render_document("contrato" if i % 2 else "escritura", p, rng),
            f"neg_documento_{i:03d}.jpg", {"es_ine": False, "tipo": "documento"}, "easy",
            canvas=(1700, 2200), scale_range=(0.7, 0.9))
        put(render_noise_photo(rng), f"neg_foto_{i:03d}.jpg", {"es_ine": False, "tipo": "foto"}, "easy",
            scale_range=(0.8, 0.95))
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Genera banco sintético de INE y negativos")
    ap.add_argument("--out", default="synth/banco")
    ap.add_argument("-n", type=int, default=12, help="número de personas (cada una → anverso+reverso)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    labs = make_dataset(a.out, a.n, a.seed)
    print(f"{len(labs)} imágenes en {a.out}")
