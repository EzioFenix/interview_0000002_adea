"""Interfaz web (Streamlit): modo TEST con drag & drop y modo LOTE.

  streamlit run app.py        (o:  python cli.py ui)

Funciona en el celular: abre la URL de red que imprime Streamlit y usa la pestaña «Cámara».
La cámara del navegador exige HTTPS (túnel de Cloudflare/ngrok) salvo en localhost.
"""
from __future__ import annotations

import io
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ine_pipeline.batch import run_batch, write_report  # noqa: E402
from ine_pipeline.db import build_persistor, build_sp_params  # noqa: E402
from ine_pipeline.pipeline import IMAGE_EXTS, process_image  # noqa: E402

st.set_page_config(page_title="Lector de INE", page_icon="🪪", layout="wide")

SENSITIVE = {"curp", "clave_elector", "apellido_paterno", "apellido_materno", "nombres", "nombre_completo",
             "domicilio_completo", "calle", "numero_exterior", "numero_interior", "colonia", "codigo_postal",
             "fecha_nacimiento", "mrz_linea1", "mrz_linea2", "mrz_linea3", "mrz_apellidos", "mrz_nombres",
             "mrz_cic", "codigo_texto"}
BADGE = {"OK": "🟢", "REVISION": "🟠", "DESCARTADO": "⚪", "ERROR": "🔴"}
LABEL_ES = {"curp": "CURP", "clave_elector": "Clave de elector", "apellido_paterno": "Apellido paterno",
            "apellido_materno": "Apellido materno", "nombres": "Nombre(s)", "nombre_completo": "Nombre completo",
            "domicilio_completo": "Domicilio (texto)", "calle": "Calle", "numero_exterior": "Núm. exterior",
            "numero_interior": "Núm. interior", "colonia": "Colonia", "codigo_postal": "Código postal",
            "municipio": "Alcaldía / municipio", "estado": "Estado", "estado_clave": "Estado (clave)",
            "municipio_clave": "Municipio (clave)", "seccion": "Sección", "localidad": "Localidad",
            "emision": "Emisión", "vigencia": "Vigencia", "anio_registro": "Año de registro",
            "fecha_nacimiento": "Fecha de nacimiento (de CURP)", "sexo": "Sexo (de CURP)",
            "entidad_nacimiento": "Entidad de nacimiento (de CURP)", "mrz_linea1": "MRZ línea 1",
            "mrz_linea2": "MRZ línea 2", "mrz_linea3": "MRZ línea 3", "mrz_checks": "MRZ dígitos verificadores",
            "mrz_apellidos": "MRZ apellidos", "mrz_nombres": "MRZ nombres", "mrz_sexo": "MRZ sexo",
            "mrz_nacimiento_yymmdd": "MRZ nacimiento", "mrz_vence_yymmdd": "MRZ vencimiento", "mrz_cic": "CIC (MRZ)",
            "codigo_formato": "Código: formato", "codigo_texto": "Código: contenido decodificado"}


# --------------------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False, max_entries=24)
def run_pipeline(name: str, raw: bytes):
    return process_image(raw, keep_debug=True, name=name)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_value(key: str, v, hide: bool):
    if not hide or key not in SENSITIVE or v is None:
        return v
    s = str(v)
    return s[:2] + "•" * max(0, len(s) - 2)


def draw_ocr(img_bgr, lines):
    out = img_bgr.copy()
    for l in lines:
        col = (60, 170, 60) if l.score >= 0.9 else (0, 150, 240) if l.score >= 0.7 else (40, 40, 220)
        cv2.polylines(out, [l.box.astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
    return out


def fields_table(res, hide: bool, expected: dict | None = None):
    rows = []
    for k, f in res.fields.items():
        row = {"Campo": LABEL_ES.get(k, k),
               "Valor": mask_value(k, f.value.isoformat() if hasattr(f.value, "isoformat") else f.value, hide),
               "Confianza": round(f.confidence, 3),
               "Validación": {True: "✅ válido", False: "⚠️ no válido", None: "—"}[f.valid],
               "Origen / nota": (f.source + (" · " + f.note if f.note else ""))}
        if expected is not None:
            exp = expected.get(k)
            row["Esperado (ficticio)"] = mask_value(k, exp, hide) if exp is not None else ""
            if exp is not None and k not in ("colonia", "estado", "municipio"):
                row["¿Coincide?"] = "✅" if str(exp).upper().replace(".", "") == str(f.value).upper().replace(".", "") else "❌"
        rows.append(row)
    return pd.DataFrame(rows)


def render_result(res, hide: bool, expected: dict | None = None, key: str = "r"):
    c = res.classification
    head = f"{BADGE.get(res.decision, '')} **{res.decision}** · "
    head += {"front": "anverso INE", "back": "reverso INE", None: "no es INE"}.get(res.side, "—")
    head += f" · confianza **{res.confidence:.2f}** · {res.timings.get('total', 0):.1f} s"
    st.markdown(head)
    st.caption(res.reason)
    t1, t2, t3, t4 = st.tabs(["1 · Clasificación", "2 · Visión y ROI", "3 · Extracción", "4 · Persistencia"])

    with t1:
        if c:
            a, b, d = st.columns(3)
            a.metric("Puntaje anverso", f"{c.front_score:.1f}", help="umbral 3.5")
            b.metric("Puntaje reverso", f"{c.back_score:.1f}", help="umbral 3.0")
            d.metric("¿Es INE?", "SÍ" if c.is_ine else "NO")
            st.markdown("**Evidencias que sumaron o restaron puntos:**")
            st.write("\n".join(f"- {r}" for r in c.reasons))
        else:
            st.error(res.reason)

    with t2:
        dbg = res.debug
        if "original" in dbg:
            cols = st.columns(4)
            cols[0].image(rgb(dbg["deteccion"]), caption=f"Detección de tarjeta: {dbg['card_method']}", width="stretch")
            cols[1].image(rgb(dbg["card"]), caption="Warp perspective (tarjeta enderezada)", width="stretch")
            cols[2].image(rgb(dbg["enhanced"]), caption="Limpieza: denoise + CLAHE + iluminación", width="stretch")
            cols[3].image(rgb(dbg["binarizada"]), caption="Binarización adaptativa (Gauss)", width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("Nitidez (var. Laplaciano)", f"{dbg.get('blur_score', 0):.0f}", help="< ~60 suele ser borrosa")
            m2.metric("Enderezado de texto", f"{dbg.get('deskew_deg', 0)}°")
            m3.metric("Variante usada para OCR", dbg.get("variante", "—"))
            if res.attempts:
                with st.expander("Reintentos automáticos (variantes de imagen probadas)"):
                    st.dataframe(pd.DataFrame(res.attempts), width="stretch", hide_index=True)

    with t3:
        dbg = res.debug
        if "ocr_lines" in dbg and dbg["ocr_lines"]:
            left, right = st.columns([2, 3])
            left.image(rgb(draw_ocr(dbg["ocr_image"], dbg["ocr_lines"])),
                       caption="Cajas de OCR (verde ≥0.9 · naranja ≥0.7 · rojo <0.7)", width="stretch")
            with right:
                if res.fields:
                    st.dataframe(fields_table(res, hide, expected), width="stretch", hide_index=True)
                else:
                    st.info("No se extrajeron campos (no es INE o no se encontraron datos válidos).")
        elif not res.is_ine:
            st.info("Imagen descartada: no se ejecuta extracción.")
        if res.barcodes:
            st.markdown("**Códigos decodificados (PDF417 / QR)**")
            st.dataframe(pd.DataFrame([{**b, "texto": mask_value("codigo_texto", b["texto"], hide)} for b in res.barcodes]),
                         width="stretch", hide_index=True)
        with st.expander("JSON completo"):
            d = res.to_dict()
            if hide:
                for k, f in d["campos"].items():
                    f["value"] = mask_value(k, f["value"], True)
            st.json(d, expanded=False)

    with t4:
        st.markdown("**Política de persistencia (según el diagrama del examen)**")
        policy = {"OK": "Anverso → tabla `expedientes` + `audit_log` (SQLite local). Reverso → solo auditoría.",
                  "REVISION": "Solo log de auditoría (cuarentena automática, sin intervención humana).",
                  "DESCARTADO": "Solo log de auditoría (no es INE).", "ERROR": "Solo log de auditoría."}
        st.info(policy.get(res.decision, ""))
        if res.side == "front" and "curp" in res.fields:
            payload = build_sp_params(res.to_dict())
            payload = {k: mask_value({"P_CURP": "curp"}.get(k, "x"), v, hide) if k == "P_CURP" else v for k, v in payload.items()}
            st.markdown("Valores que se guardarían en `expedientes`:")
            st.json({k: (str(v) if v is not None else None) for k, v in payload.items()}, expanded=False)
        if st.button("Guardar en SQLite (local_store.db)", key=f"save_{key}"):
            p = build_persistor("sqlite", str(ROOT / "local_store.db"))
            out = p.persist(res.to_dict())
            p.close()
            st.success(f"Acción: {out['accion']} (id {out['id']}) → local_store.db")


# --------------------------------------------------------------------------------------- UI
st.title("🪪 Lector de INE — pipeline sin modelos entrenados")
st.caption("Clasificación → Visión & ROI → Extracción → Persistencia · OCR preentrenado + visión clásica + validadores")

with st.sidebar:
    mode = st.radio("Modo", ["🧪 Test (arrastrar imagen)", "📦 Lote (carpeta / ZIP)"])
    hide = st.checkbox("Ocultar datos personales en pantalla", value=False,
                       help="Enmascara CURP, nombre, domicilio, etc. Útil para capturas o demos con datos reales.")
    st.markdown("---")
    st.caption("Todo el procesamiento ocurre en este equipo; no se envían imágenes a servicios externos.")

if mode.startswith("🧪"):
    tab_up, tab_cam, tab_syn = st.tabs(["📂 Arrastrar / subir", "📷 Cámara (celular)", "🧬 Ejemplo ficticio"])
    items: list[tuple[str, bytes, dict | None]] = []

    with tab_up:
        files = st.file_uploader("Arrastra una o varias imágenes aquí (frente o reverso de INE, u otros documentos)",
                                 type=[e.lstrip(".") for e in sorted(IMAGE_EXTS)], accept_multiple_files=True, key="uploader")
        for f in files or []:
            items.append((f.name, f.getvalue(), None))

    with tab_cam:
        shot = st.camera_input("Toma una foto de la credencial (con luz pareja, sin reflejos)")
        if shot is not None:
            items.append(("captura_camara.jpg", shot.getvalue(), None))

    with tab_syn:
        st.write("Genera una credencial **ficticia** (datos inventados, CURP con dígito verificador válido) "
                 "fotografiada virtualmente, y compara lo extraído con lo esperado.")
        c1, c2, c3 = st.columns([1, 1, 2])
        sev = c3.select_slider("Dificultad de la foto", ["easy", "medium", "hard"], value="medium", key="sev")
        if c1.button("Generar frente", key="gen_front"):
            st.session_state["syn"] = ("front", random.randint(0, 10**6), sev)
        if c2.button("Generar reverso", key="gen_back"):
            st.session_state["syn"] = ("back", random.randint(0, 10**6), sev)
        if "syn" in st.session_state:
            from synth import generate as G
            side, seed, sv = st.session_state["syn"]
            rng = random.Random(seed)
            person = G.fake_person(rng)
            if side == "front":
                img = G.photograph(G.render_front(person, rng), rng, sv, rot_choices=(0, 180) if sv != "easy" else (0,))
                expected = {k: v for k, v in person.items()}
                expected["fecha_nacimiento"] = person["fecha_nacimiento"]
            else:
                back, mrz, _ = G.render_back(person, rng, use_qr=bool(seed % 2))
                img = G.photograph(back, rng, sv, rot_choices=(0, 90, 270))
                expected = {"mrz_linea1": mrz[0], "mrz_linea2": mrz[1], "mrz_linea3": mrz[2]}
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            items.append((f"ficticia_{side}_{seed}.jpg", buf.getvalue(), expected))

    if not items:
        st.info("Arrastra una imagen en la primera pestaña para ver cómo se extraen los datos paso a paso.")
    for idx, (name, raw, expected) in enumerate(items):
        st.divider()
        st.subheader(name)
        left, right = st.columns([1, 2])
        left.image(raw, caption="Imagen original", width="stretch")
        with right:
            with st.spinner("Procesando…"):
                res = run_pipeline(name, raw)
            render_result(res, hide, expected, key=f"{idx}_{res.sha256[:8]}")

else:
    st.subheader("Procesamiento por lotes")
    src = st.radio("Origen", ["Carpeta en este equipo", "Subir ZIP de imágenes"], horizontal=True)
    folder = None
    if src.startswith("Carpeta"):
        folder = st.text_input("Ruta de la carpeta", value=str(ROOT / "synth" / "banco"))
    else:
        z = st.file_uploader("ZIP con imágenes", type=["zip"])
        if z is not None:
            tmp = Path(tempfile.mkdtemp(prefix="lote_"))
            with zipfile.ZipFile(io.BytesIO(z.getvalue())) as zf:
                zf.extractall(tmp)
            folder = str(tmp)
    c1, c2, c3 = st.columns(3)
    db_mode = c1.selectbox("Base de datos", ["sqlite (local_store.db)", "none (no guardar)"])
    workers = c2.number_input("Procesos en paralelo", 1, 8, 2)
    persist_rev = c3.checkbox("Guardar también los de baja confianza", value=False)
    if st.button("▶ Procesar lote", type="primary", disabled=not folder, key="run_batch"):
        bar = st.progress(0.0, text="Iniciando…")
        mode_key = db_mode.split(" ")[0]
        out = run_batch(folder, mode_key, int(workers), str(ROOT / "local_store.db"), persist_rev, True,
                        progress=lambda i, n, r: bar.progress(i / n, text=f"{i}/{n} · {Path(r['archivo']).name} → {r['decision']}"))
        st.session_state["batch"] = out
    out = st.session_state.get("batch")
    if out:
        st.success(f"{out['total']} imágenes en {out['segundos']} s → {out['resumen']}")
        df = pd.DataFrame([{"archivo": Path(r["archivo"]).name, "decisión": r["decision"], "lado": r["lado"],
                            "confianza": r["confianza"], "acción BD": a["accion"], "motivo": r["motivo"]}
                           for r, a in zip(out["registros"], out["acciones"])])
        st.dataframe(df, width="stretch", hide_index=True)
        if out["pares_frente_reverso"]:
            st.markdown(f"**{len(out['pares_frente_reverso'])} pares anverso ↔ reverso** cruzados por identidad (MRZ ↔ CURP)")
            st.dataframe(pd.DataFrame(out["pares_frente_reverso"]), width="stretch", hide_index=True)
        tmp_csv = Path(tempfile.gettempdir()) / "reporte_lote.csv"
        write_report(out, tmp_csv)
        st.download_button("Descargar reporte CSV", tmp_csv.read_bytes(), "reporte_lote.csv", "text/csv")
        st.download_button("Descargar reporte JSON (completo)", json.dumps(out, ensure_ascii=False, default=str, indent=1),
                           "reporte_lote.json", "application/json")
