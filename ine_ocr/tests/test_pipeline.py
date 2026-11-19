"""Prueba de integración sobre un banco sintético pequeño (corre OCR real; ~1 min)."""
import json

import pytest

from ine_pipeline.pipeline import process_image
from synth.generate import make_dataset


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    d = tmp_path_factory.mktemp("banco")
    labels = make_dataset(d, n=3, seed=11, severities=("easy", "medium", "hard"))
    return d, labels


def test_end_to_end(bank):
    d, labels = bank
    hits_cls = hits_fields = n_front = 0
    for lab in labels:
        r = process_image(d / lab["archivo"])
        hits_cls += r.is_ine == lab["es_ine"]
        if lab["es_ine"] and lab["lado"] == "front":
            n_front += 1
            f = r.fields
            assert r.decision == "OK", (lab["archivo"], r.reason)
            hits_fields += f["curp"].value == lab["datos"]["curp"] and f["clave_elector"].value == lab["datos"]["clave_elector"]
        if not lab["es_ine"]:
            assert r.decision == "DESCARTADO", (lab["archivo"], r.reason)
        if lab["es_ine"] and lab["lado"] == "back":
            assert r.side == "back" and r.decision == "OK"
            assert r.fields["codigo_texto"].value == lab["codigo"]["texto"]
    assert hits_cls == len(labels)
    assert hits_fields == n_front


def test_corrupt_file_does_not_crash(tmp_path):
    bad = tmp_path / "roto.jpg"
    bad.write_bytes(b"esto no es una imagen")
    r = process_image(bad)
    assert r.decision == "ERROR"
