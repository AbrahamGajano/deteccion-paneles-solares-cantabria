from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

RUTA_MANIFEST = Path("./data/processed/manifests/teselas_con_edificios.csv")
RUTA_SELECCION = Path("./data/processed/manifests/muestra_etiquetado.csv")

CARPETA_PNOA = Path("data/pnoa")
CARPETA_IMAGENES = Path("data/labeling/images")
TAM_TES = 512


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
        return pd.read_csv(ruta)
    except FileNotFoundError:
        print("\nCreando dataframe vacio...")
        return pd.DataFrame(columns=["tile_id"])


def seleccionar_nuevas_teselas(
    seleccion_anterior: pd.DataFrame,
    manifest: pd.DataFrame,
    cantidad: int,
) -> pd.DataFrame:
    """Funcion que retorna las nuevas teselas candidatas

    Args:
        seleccion_anterior (pd.DataFrame): Anterior seleccion
        manifest (pd.DataFrame): Manifest con todas las teselas
        cantidad (int): Cantidad a extraer

    Returns:
        pd.DataFrame: Dataframe de las elejidas
    """

    # Por simplicidad evitaremos las irregulares en el entrenamiento
    teselas_validas = manifest[
        (manifest["ancho"] == TAM_TES) & (manifest["alto"] == TAM_TES)
    ]

    # Tenemos que encargarnos que no reelejir las ya usadas
    teselas_validas = teselas_validas[
        ~teselas_validas["tile_id"].isin(seleccion_anterior["tile_id"])
    ]

    # En caso de haber menos de las solicitadas
    num = cantidad
    if len(teselas_validas) < cantidad:
        num = len(teselas_validas)
        print(f"Tan se han podido añadir {len(teselas_validas)} ya que no habia mas")

    # Realiza la seleccion aleatoria
    seleccion_actual = teselas_validas.sample(n=num)

    return seleccion_actual


def extraer_tesela(registro: pd.Series) -> Path:
    """Funcion que guarda como png la tesela deseada

    Args:
        registro (pd.Series): Registro de la tesela

    Returns:
        Path: Ruta destino del png
    """

    # Ectraemos la ruta de los tifs y el png
    ruta_tif = CARPETA_PNOA / registro["tif"]
    CARPETA_IMAGENES.mkdir(parents=True, exist_ok=True)
    ruta_png = CARPETA_IMAGENES / f"{registro['tile_id']}.png"

    # Abrimos el tif con rasterio
    with rasterio.open(ruta_tif) as tif:
        # Necesitamos especificar la ventana de pixeles de la tesela
        ventana = Window(
            col_off=int(registro["columna"]),
            row_off=int(registro["fila"]),
            width=int(registro["ancho"]),
            height=int(registro["alto"]),
        )
        # Por defecto retorna color como primera dim pero Pillow requiere como ultima
        imagen = tif.read([1, 2, 3], window=ventana)
        imagen = np.moveaxis(imagen, 0, -1)

        # Guardamos la imagen
        Image.fromarray(imagen).save(ruta_png)

    return ruta_png


def extraer_teselas(teselas: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        teselas (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # Lista de los registros validos
    registros_correctos = []
    total = len(teselas)

    # Recorremos todas las teselas y vamos extrayendolas
    for indx, (_, tesela) in enumerate(teselas.iterrows(), start=1):
        try:
            ruta = extraer_tesela(registro=tesela)
            registros_correctos.append(tesela.to_dict())

            print(f"Tesela: {ruta.name} [{indx}/{total}] guardada como png")

        except Exception as error:
            print(f"Error: {error} en la tesela [{indx}/{total}]")

    return pd.DataFrame(registros_correctos)


def guardar_seleccion(seleccion: pd.DataFrame) -> None:
    """Guarda el csv de la seleccion

    Args:
        seleccion (pd.DataFrame): DataFrame a guardar
    """

    RUTA_SELECCION.parent.mkdir(parents=True, exist_ok=True)
    seleccion.to_csv(path_or_buf=RUTA_SELECCION, index=False)


def main():
    """Programa que selecciona teselas para guardar como png"""
    # Definimos la cantidad de pngs a generar inicial
    cantidad = 1000

    print("\nCargando el manifest [1/5]")
    manifest = cargar_manifest()

    print("\nCargando la seleccion anterior [2/5]")
    seleccion_anterior = cargar_seleccion_anterior()

    print("\nCargando la seleccion actual [3/5]")
    seleccion_actual = seleccionar_nuevas_teselas(
        seleccion_anterior=seleccion_anterior, manifest=manifest, cantidad=cantidad
    )

    print("\nGuardando pngs [4/5]")
    seleccion_correcta = extraer_teselas(seleccion_actual)
    seleccion_total = pd.concat(
        [seleccion_correcta, seleccion_anterior], ignore_index=True
    )
    seleccion_total = seleccion_total.drop_duplicates(subset="tile_id")

    print("\nGuardando cambios [5/5]")
    guardar_seleccion(seleccion_total)


if __name__ == "__main__":
    main()
