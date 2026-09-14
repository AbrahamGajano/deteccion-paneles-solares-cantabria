from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window
from ultralytics import YOLO

RUTA_MODELO = Path(
    "runs/segment/runs/cantabria/desde_cantabria_ronda_1/weights/best.pt"
)

RUTA_MANIFEST = Path("data/processed/manifests/teselas_con_edificios.csv")

RUTA_MUESTRA = Path("data/processed/manifests/muestra_etiquetado.csv")

RUTA_RESULTADO = Path("data/processed/manifests/seleccion_ronda_2.csv")

CARPETA_PNOA = Path("data/pnoa")
CARPETA_SALIDA = Path("data/labeling/ronda_2/images")

CONFIANZA_INTERVALO_MIN = 0.40
CONFIANZA_INTERVALO_MAX = 0.60
CANTIDAD_INTERVALO = 250

CONFIANZA_DETECCIONES = 0.50
MINIMO_DETECCIONES = 4
CANTIDAD_GRANDES = 150

TAM_TES = 512
SEMILLA = 42


def cargar_manifest() -> pd.DataFrame:
    """Carga las teselas disponibles y elimina las ya utilizadas."""

    if not RUTA_MANIFEST.exists():
        raise FileNotFoundError(RUTA_MANIFEST)

    manifest = pd.read_csv(RUTA_MANIFEST)

    manifest = manifest[
        (manifest["ancho"] == TAM_TES) & (manifest["alto"] == TAM_TES)
    ].copy()

    usados = cargar_tile_ids_usados()

    manifest = manifest[~manifest["tile_id"].astype(str).isin(usados)]

    return manifest.sample(
        frac=1,
        random_state=SEMILLA,
    ).reset_index(drop=True)


def cargar_tile_ids_usados() -> set[str]:
    """Busca las teselas que ya han sido utilizadas anteriormente."""

    usados = set()

    if RUTA_MUESTRA.exists():
        muestra = pd.read_csv(RUTA_MUESTRA)

        if "tile_id" in muestra.columns:
            usados.update(muestra["tile_id"].dropna().astype(str))

    carpeta_labeling = Path("data/labeling")

    if carpeta_labeling.exists():
        for ruta in carpeta_labeling.rglob("*"):
            if ruta.suffix.lower() in {".png", ".jpg", ".jpeg", ".json"}:
                usados.add(ruta.stem)

    return usados


def leer_tesela(registro: pd.Series) -> np.ndarray:
    """Extrae una tesela directamente de su ortofoto."""

    ruta_tif = CARPETA_PNOA / str(registro["tif"])

    if not ruta_tif.exists():
        raise FileNotFoundError(ruta_tif)

    ventana = Window(
        col_off=int(registro["columna"]),
        row_off=int(registro["fila"]),
        width=int(registro["ancho"]),
        height=int(registro["alto"]),
    )

    with rasterio.open(ruta_tif) as tif:
        imagen = tif.read(
            [1, 2, 3],
            window=ventana,
        )

    return np.moveaxis(imagen, 0, -1)


def predecir(
    modelo: YOLO,
    imagen: np.ndarray,
) -> tuple[float, int]:
    """Retorna la confianza maxima y el numero de detecciones fiables."""

    imagen_bgr = imagen[:, :, ::-1].copy()

    resultado = modelo.predict(
        source=imagen_bgr,
        imgsz=512,
        conf=CONFIANZA_INTERVALO_MIN,
        batch=1,
        device=0,
        verbose=False,
    )[0]

    if resultado.boxes is None or len(resultado.boxes) == 0:
        return 0.0, 0

    confianzas = resultado.boxes.conf.detach().cpu().tolist()

    confianza_maxima = max(confianzas)

    numero_detecciones = sum(
        confianza >= CONFIANZA_DETECCIONES for confianza in confianzas
    )

    return confianza_maxima, numero_detecciones


def guardar_imagen(
    imagen: np.ndarray,
    tile_id: str,
) -> Path:
    """Guarda la tesela seleccionada como PNG."""

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    ruta_salida = CARPETA_SALIDA / f"{tile_id}.png"
    Image.fromarray(imagen).save(ruta_salida)

    return ruta_salida


def guardar_resultados(registros: list[dict]) -> None:
    """Guarda la ronda y actualiza la lista de teselas utilizadas."""

    if not registros:
        print("\nNo se ha seleccionado ninguna imagen")
        return

    seleccion = pd.DataFrame(registros)

    RUTA_RESULTADO.parent.mkdir(parents=True, exist_ok=True)
    seleccion.to_csv(RUTA_RESULTADO, index=False)

    if RUTA_MUESTRA.exists():
        muestra_anterior = pd.read_csv(RUTA_MUESTRA)
        muestra_total = pd.concat(
            [muestra_anterior, seleccion],
            ignore_index=True,
        )
    else:
        muestra_total = seleccion

    muestra_total = muestra_total.drop_duplicates(
        subset="tile_id",
        keep="last",
    )

    muestra_total.to_csv(RUTA_MUESTRA, index=False)


def seleccionar(
    modelo: YOLO,
    manifest: pd.DataFrame,
) -> list[dict]:
    """Selecciona imágenes ambiguas y posibles instalaciones grandes."""

    seleccionados = []
    cantidad_intervalo = 0
    cantidad_grandes = 0
    total = len(manifest)

    for indice, (_, registro) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        if (
            cantidad_intervalo >= CANTIDAD_INTERVALO
            and cantidad_grandes >= CANTIDAD_GRANDES
        ):
            break

        try:
            imagen = leer_tesela(registro)

            confianza_maxima, numero_detecciones = predecir(
                modelo=modelo,
                imagen=imagen,
            )

            cumple_intervalo = (
                cantidad_intervalo < CANTIDAD_INTERVALO
                and CONFIANZA_INTERVALO_MIN
                <= confianza_maxima
                <= CONFIANZA_INTERVALO_MAX
            )

            cumple_grandes = (
                cantidad_grandes < CANTIDAD_GRANDES
                and numero_detecciones >= MINIMO_DETECCIONES
            )

            if not cumple_intervalo and not cumple_grandes:
                if indice % 100 == 0:
                    print(
                        f"Procesadas: {indice}/{total} | "
                        f"Intervalo: {cantidad_intervalo}/"
                        f"{CANTIDAD_INTERVALO} | "
                        f"Grandes: {cantidad_grandes}/"
                        f"{CANTIDAD_GRANDES}"
                    )

                continue

            tipos = []

            if cumple_intervalo:
                tipos.append("intervalo")
                cantidad_intervalo += 1

            if cumple_grandes:
                tipos.append("muchas_detecciones")
                cantidad_grandes += 1

            tile_id = str(registro["tile_id"])

            ruta_imagen = guardar_imagen(
                imagen=imagen,
                tile_id=tile_id,
            )

            nuevo_registro = registro.to_dict()
            nuevo_registro["confianza_maxima"] = round(
                confianza_maxima,
                4,
            )
            nuevo_registro["numero_detecciones"] = numero_detecciones
            nuevo_registro["tipo_seleccion"] = "_y_".join(tipos)
            nuevo_registro["ruta_imagen"] = str(ruta_imagen)

            seleccionados.append(nuevo_registro)

            print(
                f"Seleccionada: {tile_id} | "
                f"conf={confianza_maxima:.3f} | "
                f"detecciones={numero_detecciones} | "
                f"tipo={nuevo_registro['tipo_seleccion']}"
            )

        except Exception as error:
            print(f"Error procesando {registro['tile_id']}: {error}")

    return seleccionados


def main() -> None:
    """Selecciona las imágenes de la segunda ronda de etiquetado."""

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(RUTA_MODELO)

    if CARPETA_SALIDA.exists() and any(CARPETA_SALIDA.iterdir()):
        raise FileExistsError(f"La carpeta {CARPETA_SALIDA} ya contiene imágenes")

    print("\nCargando teselas disponibles [1/3]")
    manifest = cargar_manifest()
    print(f"Teselas disponibles: {len(manifest)}")

    print("\nCargando modelo [2/3]")
    modelo = YOLO(str(RUTA_MODELO))

    print("\nBuscando imágenes interesantes [3/3]")

    try:
        seleccionados = seleccionar(
            modelo=modelo,
            manifest=manifest,
        )
    except KeyboardInterrupt:
        print("\nProceso detenido por el usuario")
        guardar_resultados(seleccionados)
        return

    guardar_resultados(seleccionados)

    if seleccionados:
        seleccion = pd.DataFrame(seleccionados)

        cantidad_intervalo = seleccion[
            seleccion["tipo_seleccion"].str.contains("intervalo")
        ].shape[0]

        cantidad_grandes = seleccion[
            seleccion["tipo_seleccion"].str.contains("muchas_detecciones")
        ].shape[0]

        print("\nSeleccion terminada")
        print(f"Casos del intervalo: {cantidad_intervalo}")
        print(f"Casos con muchas detecciones: {cantidad_grandes}")
        print(f"Imagenes diferentes: {len(seleccionados)}")
        print(f"Imagenes: {CARPETA_SALIDA}")
        print(f"CSV: {RUTA_RESULTADO}")


if __name__ == "__main__":
    main()
