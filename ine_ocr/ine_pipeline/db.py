"""Persistencia local: todo se guarda en un archivo SQLite (local_store.db), sin Oracle ni SQL Server.

* Tabla `expedientes`: datos del anverso del INE (OK, o REVISION con --persistir-revision).
* Tabla `audit_log`  : decisión, motivos, confianza y qué campos se extrajeron/validaron.
  NO guarda valores de campos (datos personales).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path

def _val(rec: dict, key: str):
    f = rec.get("campos", {}).get(key)
    return f["value"] if f else None


def build_sp_params(rec: dict) -> dict:
    """Mapea un registro del pipeline a los valores (nombre → valor) que se guardan en `expedientes`."""
    fn = _val(rec, "fecha_nacimiento")
    return {
        "P_NOMBRE_ARCHIVO": os.path.basename(rec["archivo"]),
        "P_SHA256": rec["sha256"],
        "P_CURP": _val(rec, "curp"),
        "P_CLAVE_ELECTOR": _val(rec, "clave_elector"),
        "P_APELLIDO_PATERNO": _val(rec, "apellido_paterno"),
        "P_APELLIDO_MATERNO": _val(rec, "apellido_materno"),
        "P_NOMBRES": _val(rec, "nombres"),
        "P_FECHA_NACIMIENTO": datetime.fromisoformat(fn) if isinstance(fn, str) else fn,
        "P_SEXO": _val(rec, "sexo"),
        "P_CALLE": _val(rec, "calle"),
        "P_NUM_EXTERIOR": _val(rec, "numero_exterior"),
        "P_NUM_INTERIOR": _val(rec, "numero_interior"),
        "P_COLONIA": _val(rec, "colonia"),
        "P_CODIGO_POSTAL": _val(rec, "codigo_postal"),
        "P_MUNICIPIO": _val(rec, "municipio"),
        "P_ESTADO": _val(rec, "estado"),
        "P_SECCION": _val(rec, "seccion"),
        "P_VIGENCIA": _val(rec, "vigencia"),
        "P_CONFIANZA": rec["confianza"],
    }


def audit_detail(rec: dict) -> str:
    """JSON para el log de auditoría SIN valores personales."""
    campos = {k: {"conf": v["confidence"], "valido": v["valid"]} for k, v in rec.get("campos", {}).items()
              if not k.startswith("mrz_linea") and k != "codigo_texto"}
    return json.dumps({"clasificacion": rec.get("clasificacion"), "campos": campos,
                       "tiempos_s": rec.get("tiempos_s")}, ensure_ascii=False)


# --------------------------------------------------------------------------------------- SQLite
class SqliteStore:
    """Guarda expedientes y auditoría en un archivo SQLite local."""

    def __init__(self, path: str | Path = "local_store.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS expedientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, archivo TEXT, sha256 TEXT UNIQUE,
            curp TEXT, clave_elector TEXT, apellido_paterno TEXT, apellido_materno TEXT, nombres TEXT,
            fecha_nacimiento TEXT, sexo TEXT, calle TEXT, num_exterior TEXT, num_interior TEXT, colonia TEXT,
            codigo_postal TEXT, municipio TEXT, estado TEXT, seccion TEXT, vigencia TEXT, confianza REAL);
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, archivo TEXT, sha256 TEXT, decision TEXT,
            es_ine INTEGER, lado TEXT, confianza REAL, motivo TEXT, detalle TEXT);
        """)

    def already_ok(self, sha: str) -> bool:
        return self.conn.execute("SELECT 1 FROM audit_log WHERE sha256=? AND decision='OK'", (sha,)).fetchone() is not None

    def registrar_expediente(self, rec: dict):
        p = build_sp_params(rec)
        fn = p["P_FECHA_NACIMIENTO"]
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO expedientes (fecha,archivo,sha256,curp,clave_elector,apellido_paterno,"
            "apellido_materno,nombres,fecha_nacimiento,sexo,calle,num_exterior,num_interior,colonia,codigo_postal,"
            "municipio,estado,seccion,vigencia,confianza) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), p["P_NOMBRE_ARCHIVO"], p["P_SHA256"], p["P_CURP"],
             p["P_CLAVE_ELECTOR"], p["P_APELLIDO_PATERNO"], p["P_APELLIDO_MATERNO"], p["P_NOMBRES"],
             fn.date().isoformat() if isinstance(fn, datetime) else fn, p["P_SEXO"], p["P_CALLE"],
             p["P_NUM_EXTERIOR"], p["P_NUM_INTERIOR"], p["P_COLONIA"], p["P_CODIGO_POSTAL"], p["P_MUNICIPIO"],
             p["P_ESTADO"], p["P_SECCION"], p["P_VIGENCIA"], p["P_CONFIANZA"]))
        self.conn.commit()
        return cur.lastrowid

    def log(self, rec: dict):
        self.conn.execute(
            "INSERT INTO audit_log (fecha,archivo,sha256,decision,es_ine,lado,confianza,motivo,detalle) VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), os.path.basename(rec["archivo"]), rec["sha256"],
             rec["decision"], int(bool(rec.get("es_ine"))), rec.get("lado"), rec.get("confianza"),
             rec.get("motivo"), audit_detail(rec)))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------------- Orquestación
class Persistor:
    """Aplica la política de persistencia del diagrama del examen.

    OK anverso  → expedientes + audit log   (OK reverso → solo audit log)
    REVISION    → solo audit log (cuarentena automática; --persistir-revision para guardarlo también en expedientes)
    DESCARTADO  → solo audit log (no es INE)
    ERROR       → solo audit log
    """

    def __init__(self, expedientes, audit, persist_review: bool = False):
        self.expedientes, self.audit, self.persist_review = expedientes, audit, persist_review

    def persist(self, rec: dict) -> dict:
        sha = rec.get("sha256")
        if sha and rec["decision"] == "OK" and self.audit.already_ok(sha):
            return {"accion": "OMITIDO_DUPLICADO", "id": None}
        action, new_id = "AUDIT", None
        # El expediente se arma con el ANVERSO (CURP, nombre, domicilio). El reverso (MRZ/QR) queda en
        # el log de auditoría y se cruza por identidad en el reporte del lote.
        has_front_data = rec.get("lado") == "front" and "curp" in rec.get("campos", {})
        if has_front_data and (rec["decision"] == "OK" or (rec["decision"] == "REVISION" and self.persist_review)):
            new_id = self.expedientes.registrar_expediente(rec)
            action = "EXPEDIENTE+AUDIT"
        self.audit.log(rec)
        return {"accion": action, "id": new_id}

    def close(self):
        for s in {id(self.expedientes): self.expedientes, id(self.audit): self.audit}.values():
            s.close()


def build_persistor(mode: str = "sqlite", sqlite_path: str = "local_store.db", persist_review: bool = False) -> Persistor | None:
    if mode == "none":
        return None
    if mode == "sqlite":
        st = SqliteStore(sqlite_path)
        return Persistor(st, st, persist_review)
    raise ValueError(f"modo de BD desconocido: {mode}")
