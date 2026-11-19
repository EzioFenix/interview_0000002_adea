import sqlite3
from unittest.mock import MagicMock, patch

from ine_pipeline import db

REC_OK = {"archivo": "/x/a.jpg", "sha256": "a" * 64, "es_ine": True, "lado": "front", "decision": "OK",
          "motivo": "ok", "confianza": 0.97, "clasificacion": {"is_ine": True},
          "tiempos_s": {"total": 1.0}, "codigos": [],
          "campos": {"curp": {"value": "HEGG560427MVZRRL04", "confidence": 1, "valid": True},
                     "clave_elector": {"value": "GRMNJS80010209H100", "confidence": 1, "valid": True},
                     "apellido_paterno": {"value": "HERNANDEZ", "confidence": 1, "valid": True},
                     "nombres": {"value": "ANA", "confidence": 1, "valid": None},
                     "fecha_nacimiento": {"value": "1956-04-27", "confidence": 1, "valid": True}}}


def test_policy_sqlite(tmp_path):
    p = db.build_persistor("sqlite", str(tmp_path / "t.db"))
    assert p.persist(REC_OK)["accion"] == "ORACLE+AUDIT"
    assert p.persist(REC_OK)["accion"] == "OMITIDO_DUPLICADO"          # idempotencia por sha256
    p.persist({**REC_OK, "sha256": "b" * 64, "decision": "REVISION"})
    p.persist({**REC_OK, "sha256": "c" * 64, "decision": "DESCARTADO", "es_ine": False, "lado": None, "campos": {}})
    p.persist({**REC_OK, "sha256": "d" * 64, "lado": "back"})          # reverso: solo auditoría
    con = sqlite3.connect(str(tmp_path / "t.db"))
    assert con.execute("select count(*) from expedientes").fetchone()[0] == 1
    assert con.execute("select count(*) from audit_log").fetchone()[0] == 4
    # el log de auditoría no debe contener datos personales
    detalle = " ".join(r[0] for r in con.execute("select detalle from audit_log"))
    assert "HEGG560427" not in detalle and "HERNANDEZ" not in detalle
    p.close()


def test_oracle_calls_sp_with_keyword_params():
    fake_ora = MagicMock()
    with patch.dict("sys.modules", {"oracledb": fake_ora}):
        st = db.OracleStore("u", "p", "dsn")
        st.registrar_expediente(REC_OK)
    cur = fake_ora.connect.return_value.cursor.return_value
    name = cur.callproc.call_args.args[0]
    kw = cur.callproc.call_args.kwargs["keyword_parameters"]
    assert name == "SP_REGISTRAR_EXPEDIENTE"
    assert kw["P_CURP"] == "HEGG560427MVZRRL04" and "P_ID_EXPEDIENTE" in kw
    fake_ora.connect.return_value.commit.assert_called()


def test_sqlserver_insert_audit():
    fake_odbc = MagicMock()
    with patch.dict("sys.modules", {"pyodbc": fake_odbc}):
        st = db.SqlServerStore("DRIVER=x", create_table=False)
        st.log(REC_OK)
    cur = fake_odbc.connect.return_value.cursor.return_value
    sql = cur.execute.call_args.args[0]
    assert "INSERT INTO dbo.INE_AUDIT_LOG" in sql and cur.execute.call_args.args[3] == "OK"
