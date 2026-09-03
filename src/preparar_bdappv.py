from pathlib import Path

import pandas as pd

RUTA_IMAGENES = Path("./data/extern/bdappv/ign/img")
RUTA_MASCARAS = Path("./data/extern/bdappv/ign/mask")


def buscar_datos() -> pd.DataFrame:
    """Funcion que busca los datos de las imagenes y sus mascaras

    Returns:
        pd.DataFrame: Retorna el df que marca para cada imagen si tiene mascara
    """
    # Listas de todas las imagenes y mascaras
    imagenes = sorted(RUTA_IMAGENES.glob(pattern="*.png"))
    mascaras = sorted(RUTA_MASCARAS.glob(pattern="*.png"))

    # Crear los dataframes
    imgs_dict = []
    for ruta in imagenes:
        img = {"id": ruta.stem, "ruta_imagen": str(ruta)}
        imgs_dict.append(img)

    mask_dict = []
    for ruta in mascaras:
        mask = {"id": ruta.stem, "ruta_mascara": str(ruta)}
        mask_dict.append(mask)

    df_img = pd.DataFrame(imgs_dict)
    df_mask = pd.DataFrame(mask_dict)

    # Igual que en un SQL al hacer el merge si falta en la izq se pondra None
    df_union = df_img.merge(right=df_mask, how="left", on="id")
    # En cada uno que no tenga su igual en mascara se considerara que no tiene
    df_union["tiene_mascara"] = df_union["ruta_mascara"].notna()

    return df_union


def seleccionar_datos(datos: pd.DataFrame, positivos: int, negativos: int):
    """Selecciona positivos y negativos y los divide en train y val."""


def convertir_mascara_yolo(ruta_mascara: Path, ruta_txt: Path) -> None:
    """Convierte las zonas blancas de una máscara en polígonos YOLO."""


def preparar_grupo(datos: pd.DataFrame, grupo: str) -> None:
    """Copia imágenes y genera etiquetas para train o val."""


def guardar_configuracion() -> None:
    """Crea el archivo bdappv.yaml utilizado por YOLO."""


def main() -> None:
    """Ejecuta la preparación completa del dataset."""
