from pathlib import Path

import geopandas as gpd
import pandas as pd

RUTA_MANIFEST = Path("./data/processed/manifests/teselas_con_edificios.csv")
RUTA_SELECCION = Path("./data/processed/manifests/muestra_etiquetado.csv")

CARPETA_IMAGENES = Path("data/labeling/images")


def cargar_manifest(ruta: Path = RUTA_MANIFEST) -> pd.DataFrame:
    """Funcion que carga el manifest csv de las teselas edificadas

    Args:
        ruta (Path, optional): Ruta del csv. Defaults to RUTA_MANIFEST.

    Raises:
        FileNotFoundError: Salta si no se encuentra el manifest
        ValueError: Salta si se encuentra vacio

    Returns:
        pd.DataFrame: Retorna el dataframe del manifest
    """

    try:
        df = pd.read_csv(filepath_or_buffer=ruta)
    except FileNotFoundError:
        print("Csv inexistente")
        raise FileNotFoundError

    if df.empty:
        print("Csv vacio...")
        raise ValueError

    return df


def cargar_seleccion_anterior(ruta: str = RUTA_SELECCION) -> pd.DataFrame:
    """Funcion que retorna el dataframe con la lista de ya seleccionados

    Args:
        ruta (str, optional): Ruta del csv. Defaults to RUTA_SELECCION.

    Returns:
        pd.DataFrame: Retorna el df de los seleccionados previamente
    """
    try:
        return gpd.read_file(ruta)
    except FileNotFoundError:
        print("\nCreando dataframe vacio...")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry")
