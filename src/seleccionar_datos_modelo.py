import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window
from ultralytics import YOLO

TAREA = "nuevas"

RUTA_MODELO = Path("runs/segment/runs/cantabria/desde_francia_n/weights/best.pt")

RUTA_MANIFEST = Path("data/processed/manifests/teselas_con_edificios.csv")
CARPETA_PNOA = Path("data/pnoa")

CARPETA_TRAIN_VAL = Path("data/labeling/train_val")
CARPETA_TEST = Path("data/labeling/test")
CARPETA_RONDA = Path("data/labeling/ronda_1")

RUTA_YOLO_ANTERIOR = Path("data/processed/cantabria_yolo")

RUTA_NEGATIVAS = Path("data/processed/manifests/negativas_dificiles_modelo.csv")
RUTA_SELECCION_RONDA = Path("data/processed/manifests/seleccion_ronda_1.csv")

CONFIANZA_NEGATIVAS = 0.40
CANTIDAD_NEGATIVAS = 300

CONFIANZA_NUEVAS = 0.75
CANTIDAD_NUEVAS = 200

TAM_TES = 512
TAM_LOTE = 8
SEMILLA = 42


def cargar_nombres_usados_entreno() -> set[str]:
    """Retorna los nombres usados en el anterior train y val."""

    nombres = set()

    for division in ["train", "val"]:
        carpeta = RUTA_YOLO_ANTERIOR / "images" / division

        for ruta_imagen in carpeta.glob("*.png"):
            nombres.add(ruta_imagen.stem)

    return nombres


def cargar_negativas_no_usadas() -> list[Path]:
    """Carga las negativas no utilizadas en el anterior entrenamiento."""

    nombres_usados = cargar_nombres_usados_entreno()
    carpeta_imagenes = CARPETA_TRAIN_VAL / "images"
    carpeta_json = CARPETA_TRAIN_VAL / "annotations"

    negativas = []

    for ruta_json in sorted(carpeta_json.glob("*.json")):
        if ruta_json.stem in nombres_usados:
            continue

        datos = json.loads(ruta_json.read_text(encoding="utf-8"))
        flags = datos.get("flags") or {}

        if flags.get("dudosa", False):
            continue

        tiene_paneles = any(
            forma.get("label") == "panel_solar" and len(forma.get("points", [])) >= 3
            for forma in datos.get("shapes", [])
        )

        if tiene_paneles:
            continue

        ruta_imagen = carpeta_imagenes / f"{ruta_json.stem}.png"

        if not ruta_imagen.exists():
            raise FileNotFoundError(ruta_imagen)

        negativas.append(ruta_imagen)

    return negativas


def seleccionar_negativas_dificiles(
    modelo: YOLO,
    rutas_imagenes: list[Path],
    confianza_minima: float,
    cantidad_maxima: int,
) -> pd.DataFrame:
    """Selecciona negativas donde el modelo encuentra falsennials paneles."""

    seleccionadas = []
    total = len(rutas_imagenes)

    for inicio in range(0, total, TAM_LOTE):
        lote = rutas_imagenes[inicio : inicio + TAM_LOTE]

        resultados = modelo.predict(
            source=[str(ruta) for ruta in lote],
            imgsz=512,
            conf=confianza_minima,
            device=0,
            verbose=False,
        )

        for resultado in resultados:
            if resultado.boxes is None or len(resultado.boxes) == 0:
                continue

            seleccionadas.append(
                {
                    "tile_id": Path(resultado.path).stem,
                    "confianza": resultado.boxes.conf.max().item(),
                    "detecciones": len(resultado.boxes),
                }
            )

        procesadas = min(inicio + TAM_LOTE, total)
        print(
            f"\rProcesadas: {procesadas}/{total} | "
            f"Con fals बेलos positivos: {len(seleccionadas)}",
            end="",
        )

    print()

    seleccion = pd.DataFrame(seleccionadas)

    if seleccion.empty:
        return pd.DataFrame(columns=["tile_id", "confianza", "detecciones"])

    seleccion = seleccion.sort_values(
        by="confianza",
        ascending=False,
    )

    return seleccion.head(cantidad_maxima)


def cargar_manifest() -> pd.DataFrame:
    """Carga las teselas candidatas del manifest."""

    manifest = pd.read_csv(RUTA_MANIFEST)

    if manifest.empty:
        raise ValueError("El manifest esta vacio")

    return manifest


def cargar_ids_usados() -> set[str]:
    """Carga los tile_id que ya se han extraido para etiquetar."""

    ids_usados = set()

    carpetas = [
        CARPETA_TRAIN_VAL / "images",
        CARPETA_TEST / "images",
        CARPETA_RONDA / "images",
    ]

    for carpeta in carpetas:
        for ruta_imagen in carpeta.glob("*.png"):
            ids_usados.add(ruta_imagen.stem)

    return ids_usados


def leer_tesela(registro: pd.Series) -> np.ndarray:
    """Lee una tesela del PNOA sin guardarla todavía."""

    ruta_tif = CARPETA_PNOA / registro["tif"]

    with rasterio.open(ruta_tif) as tif:
        ventana = Window(
            col_off=int(registro["columna"]),
            row_off=int(registro["fila"]),
            width=int(registro["ancho"]),
            height=int(registro["alto"]),
        )

        imagen = tif.read([1, 2, 3], window=ventana)
        imagen = np.moveaxis(imagen, 0, -1)

    return imagen


def guardar_seleccion_ronda(
    seleccion: list[dict],
) -> None:
    """Guarda las teselas aceptadas para poder reanudar."""

    RUTA_SELECCION_RONDA.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(seleccion).to_csv(
        RUTA_SELECCION_RONDA,
        index=False,
    )


def seleccionar_nuevas_por_confianza(
    modelo: YOLO,
    cantidad: int,
    confianza_minima: float,
) -> None:
    """Extrae nuevas teselas seleccionadas por el modelo."""

    manifest = cargar_manifest()
    ids_usados = cargar_ids_usados()

    if RUTA_SELECCION_RONDA.exists():
        seleccion = pd.read_csv(RUTA_SELECCION_RONDA).to_dict("records")
    else:
        seleccion = []

    if len(seleccion) >= cantidad:
        print(f"\nYa existen {len(seleccion)} teselas seleccionadas")
        return

    teselas_validas = manifest[
        (manifest["ancho"] == TAM_TES)
        & (manifest["alto"] == TAM_TES)
        & (~manifest["tile_id"].astype(str).isin(ids_usados))
    ]

    teselas_validas = teselas_validas.sample(
        frac=1,
        random_state=SEMILLA,
    )

    carpeta_imagenes = CARPETA_RONDA / "images"
    carpeta_anotaciones = CARPETA_RONDA / "annotations"

    carpeta_imagenes.mkdir(parents=True, exist_ok=True)
    carpeta_anotaciones.mkdir(parents=True, exist_ok=True)

    revisadas = 0
    seleccionadas_iniciales = len(seleccion)

    for _, registro in teselas_validas.iterrows():
        try:
            imagen = leer_tesela(registro)
            imagen_pil = Image.fromarray(imagen)

            resultado = modelo.predict(
                source=imagen_pil,
                imgsz=512,
                conf=confianza_minima,
                device=0,
                verbose=False,
            )[0]

            revisadas += 1

            if resultado.boxes is None or len(resultado.boxes) == 0:
                continue

            tile_id = str(registro["tile_id"])
            ruta_png = carpeta_imagenes / f"{tile_id}.png"

            imagen_pil.save(ruta_png)

            datos_registro = registro.to_dict()
            datos_registro["confianza_modelo"] = resultado.boxes.conf.max().item()
            datos_registro["detecciones_modelo"] = len(resultado.boxes)

            seleccion.append(datos_registro)
            guardar_seleccion_ronda(seleccion)

            nuevas = len(seleccion) - seleccionadas_iniciales

            print(
                f"\nSeleccionada: {ruta_png.name} | "
                f"Confianza: {datos_registro['confianza_modelo']:.3f} | "
                f"Nuevas: {nuevas}/{cantidad - seleccionadas_iniciales}"
            )

            if len(seleccion) >= cantidad:
                break

        except Exception as error:
            print(f"\nError en {registro['tile_id']}: {error}")

        if revisadas % 100 == 0:
            print(
                f"\rRevisadas por el modelo: {revisadas} | "
                f"Seleccionadas totales: {len(seleccion)}/{cantidad}",
                end="",
            )

    print("\n\nSeleccion terminada")
    print(f"Teselas revisadas: {revisadas}")
    print(f"Teselas seleccionadas: {len(seleccion)}")
    print(f"Imagenes: {carpeta_imagenes}")
    print(f"Manifest: {RUTA_SELECCION_RONDA}")


def ejecutar_negativas(modelo: YOLO) -> None:
    """Busca negativas dificiles entre las etiquetas anteriores."""

    print("\nCargando negativas no utilizadas...")
    negativas = cargar_negativas_no_usadas()
    print(f"Negativas encontradas: {len(negativas)}")

    print("\nBuscando falsos positivos...")
    seleccion = seleccionar_negativas_dificiles(
        modelo=modelo,
        rutas_imagenes=negativas,
        confianza_minima=CONFIANZA_NEGATIVAS,
        cantidad_maxima=CANTIDAD_NEGATIVAS,
    )

    RUTA_NEGATIVAS.parent.mkdir(parents=True, exist_ok=True)
    seleccion.to_csv(RUTA_NEGATIVAS, index=False)

    print("\nSeleccion terminada")
    print(f"Negativas dificiles seleccionadas: {len(seleccion)}")
    print(f"Resultado: {RUTA_NEGATIVAS}")


def main() -> None:
    """Selecciona datos utilizando el modelo de Cantabria."""

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(RUTA_MODELO)

    print("\nCargando modelo...")
    modelo = YOLO(str(RUTA_MODELO))

    if TAREA == "negativas":
        ejecutar_negativas(modelo)

    elif TAREA == "nuevas":
        seleccionar_nuevas_por_confianza(
            modelo=modelo,
            cantidad=CANTIDAD_NUEVAS,
            confianza_minima=CONFIANZA_NUEVAS,
        )

    else:
        raise ValueError("TAREA debe ser 'negativas' o 'nuevas'")


if __name__ == "__main__":
    main()
