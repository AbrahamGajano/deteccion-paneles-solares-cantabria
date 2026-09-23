"""Rutas y funciones comunes para trabajar con los manifests."""

from pathlib import Path

import pandas as pd

from paneles_solares.rutas import ruta_proyecto

POOL = ruta_proyecto("data/pool")
LABELING = ruta_proyecto("data/labeling")

MANIFEST = POOL / "manifest.csv"
MANIFEST_LABELING = LABELING / "manifest.csv"

SPLITS = ("train", "val", "test")

CAMPOS = [
    "tile_id",
    "origen",
    "estado",
    "tipo",
    "uso",
    "uso_anterior",
    "instancias",
    "modelo_yolo",
    "confianza_yolo",
    "detecciones_yolo",
    "umbral_yolo",
    "motivo",
    "tif",
    "fila",
    "columna",
    "ancho",
    "alto",
    "num_edificios",
]


def cargar_manifest(ruta: Path = MANIFEST) -> pd.DataFrame:
    """Carga un manifest o crea uno vacío si no existe.

    Args:
        ruta (Path): Ruta del manifest que se quiere cargar.

    Returns:
        pd.DataFrame: Datos guardados en el manifest.
    """

    if not ruta.exists():
        return pd.DataFrame(columns=CAMPOS)

    return pd.read_csv(
        ruta,
        dtype=object,  # Permite actualizar campos con números y conservar los vacíos.
        keep_default_na=False,
    )


def guardar_manifest(
    manifest: pd.DataFrame,
    ruta: Path,
) -> None:
    """Guarda un manifest completo.

    Args:
        manifest (pd.DataFrame): Datos que se quieren guardar.
        ruta (Path): Ruta del archivo CSV.
    """

    ruta.parent.mkdir(parents=True, exist_ok=True)

    manifest.sort_values("tile_id").to_csv(
        ruta,
        index=False,
        encoding="utf-8",
    )


def nueva_fila(registro: dict, origen: str) -> dict:
    """Crea una fila nueva con todos los campos del manifest.

    Args:
        registro (dict): Información original de la tesela.
        origen (str): Motivo por el que se seleccionó la imagen.

    Returns:
        dict: Nueva fila preparada para añadir al manifest.
    """

    fila = {campo: registro.get(campo, "") for campo in CAMPOS}

    fila.update(
        origen=origen,
        estado="pendiente",
        uso="pendiente",
    )

    return fila


def rutas_muestra(carpeta: Path, tile_id: str) -> tuple[Path, Path]:
    """Obtiene las rutas de una imagen y su anotación.

    Args:
        carpeta (Path): Carpeta que contiene images y annotations.
        tile_id (str): Identificador de la muestra.

    Returns:
        tuple[Path, Path]: Ruta del PNG y ruta del JSON.
    """

    imagen = carpeta / "images" / f"{tile_id}.png"
    anotacion = carpeta / "annotations" / f"{tile_id}.json"

    return imagen, anotacion
