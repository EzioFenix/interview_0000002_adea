# Lector de INE (sin entrenar modelos)

Clasifica imágenes de un banco heterogéneo, se queda solo con credenciales INE, las endereza y limpia,
extrae sus datos (CURP, clave de elector, nombre, domicilio, datos electorales, MRZ y PDF417/QR) y los guarda en SQLite.

## Uso
```bash
bash install.sh      # una sola vez: crea .venv e instala todo
python3 run.py       # abre la interfaz en http://localhost:8501
```
En la interfaz: **Test** (arrastra una imagen y ve cada etapa), **Cámara (celular)** y **Lote** (carpeta o ZIP).
Los datos se guardan en `local_store.db` (tablas `expedientes` y `audit_log`).

## Cómo decide (sin intervención humana)
- **OK**: CURP (regex + dígito verificador) y clave de elector válidos → `expedientes` + `audit_log`.
- **REVISION**: parece INE pero con baja confianza → solo `audit_log`.
- **DESCARTADO**: no es INE → solo `audit_log`.

## Estructura
`ine_pipeline/` (validators, vision, ocr, extract, classify, pipeline, db, batch) · `app.py` (interfaz) ·
`synth/` (banco sintético y evaluación) · `tests/`.
Opcional: `python3 cli.py procesar CARPETA` para lotes por consola (todo se guarda en SQLite local).

## Límites
Probado con imágenes sintéticas; con credenciales reales la precisión depende de la calidad de la foto.
La Ñ y los acentos pueden leerse sin tilde.
