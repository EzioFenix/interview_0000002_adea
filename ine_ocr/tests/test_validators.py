import random

from ine_pipeline import validators as V
from synth.generate import build_mrz, fake_person


def test_curp_check_digit_known_sample():
    # CURP de ejemplo usada en la documentación pública de RENAPO
    assert V.validate_curp("HEGG560427MVZRRL04")["valid"]
    assert V.curp_check_digit("HEGG560427MVZRRL0") == "4"


def test_exam_regex_exact():
    assert V.CURP_RE.match("HEGG560427MVZRRL04")
    assert not V.CURP_RE.match("HEGG560427XVZRRL04")      # sexo inválido
    assert V.CLAVE_RE.match("GRMNJS80010209H100")
    assert not V.CLAVE_RE.match("GRMNJS8001020H100")      # falta un dígito


def test_repair_single_ocr_confusion():
    assert V.repair_curp("HEGG56O427MVZRRL04") == ("HEGG560427MVZRRL04", None)      # O por 0 (corrección por posición)
    assert V.repair_curp("HEGG560427MV2RRL04") == ("HEGG560427MVZRRL04", None)      # 2 por Z en zona de letras
    fixed = V.repair_curp("HEGG560427MVZRRL0A")                                      # verificador mal leído
    assert fixed and fixed[0] == "HEGG560427MVZRRL04"
    assert V.repair_curp("HEGG560427MVZRRL84")[1] == "dígito recalculado"


def test_clave_curp_consistency():
    rng = random.Random(1)
    for _ in range(50):
        p = fake_person(rng)
        assert V.validate_curp(p["curp"])["valid"]
        assert V.validate_clave(p["clave_elector"])["valid"]
        assert V.consistent_curp_clave(p["curp"], p["clave_elector"])


def test_mrz_td1_roundtrip():
    rng = random.Random(2)
    for _ in range(30):
        p = fake_person(rng)
        mrz = V.parse_mrz_td1(build_mrz(p))
        assert mrz["checks_passed"] == 4 and mrz["valid"]
        assert mrz["surname"].startswith(p["apellido_paterno"])


def test_mrz_ocr_confusions_corrected_by_position():
    rng = random.Random(3)
    p = fake_person(rng)
    l1, l2, l3 = build_mrz(p)
    l3 = l3.replace("O", "0")                       # el OCR lee la O de los nombres como cero
    mrz = V.parse_mrz_td1([l1, l2, l3])
    assert "0" not in mrz["line3"] and mrz["valid"]
