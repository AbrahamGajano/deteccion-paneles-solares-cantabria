from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

RUTA_GPKG = "./data/processed/catastro/edificios_cantabria.gpkg"
RUTA_MANIFEST = "./data/processed/manifests/teselas_con_edificios.csv"
TAM_TES = 512


def cargar_edificios(ruta: str = RUTA_GPKG) -> gpd.GeoDataFrame:
    """Funcion que retorna el geodataframe de los edificios del catastro

    Args:
        ruta (str, optional): Ruta del archivo .gpkg a leer. Defaults to RUTA_GPKG.

    Returns:
        gpd.GeoDataFrame: El geodataframe de los edificios
    """
    return gpd.read_file(filename=Path(ruta))


def buscar_ortofotos() -> list[Path]:
    """Funcion que busca todas las ortofotos

    Returns:
        list[Path]: Retorna todas los paths de las ortofotos de cantabria .tif
    """

    ruta = Path("./data/pnoa")
    return sorted(list(ruta.glob(pattern="*.tif")))


def procesar_tif(ruta_tif: Path, edificios: gpd.GeoDataFrame) -> list[dict]:
    """Funcion que retorna los registros de las teselas que intersectan con el catastro

    Args:
        ruta_tif (Path): Ruta del tif
        edificios (gpd.geodataframe): geodataframe de los edificios de cantabria

    Raises:
        ValueError: En caso de que los crs difieran

    Returns:
        list[dict]: Retorna la lista de los registros
    """
    # Creo la lista de los registros
    registros = []
    # Abro con rastrerio el tif
    with rasterio.open(ruta_tif) as tif:
        # Es importante dejar el raise generico en caso de que cambiamos del PNOA a otro
        if edificios.crs != tif.crs:
            raise ValueError("Los CRS del gpkg y el tif no coinciden")

        # Limitamos los edificios a los encontrados en el tif
        poligono_tif = box(*tif.bounds)
        # Creamos el indice por optimizacion sino me tardaria demasiado el codigo
        indice = edificios.sindex
        # Con que una parte intersecte lo añadimos
        posiciones = indice.query(poligono_tif, predicate="intersects")

        # Seleccionamos todos los edificios del tif para no tratar con todo el tif
        edificios_tif = edificios.iloc[posiciones]
        if edificios_tif.empty:
            return registros

        indice_tif = edificios_tif.sindex

        for fila in range(0, tif.height, TAM_TES):
            for columna in range(0, tif.width, TAM_TES):
                # A modo de precaucion por si tocamos los limites ya que
                # raro seria que sean cuadrados perfectos
                ancho = min(TAM_TES, tif.width - columna)
                alto = min(TAM_TES, tif.height - fila)

                # Creamos la ventana de PIXELES
                ventana = Window(
                    col_off=columna, row_off=fila, width=ancho, height=alto
                )
                # No nos valen pixeles para comparar con el geometry
                limites = tif.window_bounds(ventana)

                # Repetimos el proceso de antes pero para el subgrupo de
                # edificios del tif
                poligono_tes = box(*limites)

                edificios_tesela = indice_tif.query(
                    poligono_tes, predicate="intersects"
                )

                if len(edificios_tesela) == 0:
                    continue

                registro = {
                    "tile_id": f"{ruta_tif.stem}_f{fila}_c{columna}",
                    "tif": ruta_tif.name,
                    "fila": fila,
                    "columna": columna,
                    "ancho": ancho,
                    "alto": alto,
                    "num_edificios": len(edificios_tesela),
                }
                registros.append(registro)

    return registros


def guardar_manifest(registros: list[dict], ruta: str = RUTA_MANIFEST) -> None:
    """Funcion que guarda el manifest de las teselas en csv

    Args:
        registros (list[dict]): Los registros de las teselas
        ruta (str, optional): La ruta donde guardar el csv. Defaults to RUTA_MANIFEST.
    """
    ruta = Path(ruta)
    # Definire como columnas las entradas del diccionario anterior
    columnas = ["tile_id", "tif", "fila", "columna", "ancho", "alto", "num_edificios"]

    # En caso de existir no lo tomo como error
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Creo el dataframe
    manifest = pd.DataFrame(data=registros, columns=columnas)
    # Guardo el csv
    manifest.to_csv(ruta, index=False)


def main():
    """Programa que guarda las posiciones de todas las teselas de interes y
    descarta aquellas no construidas mediante la informacion del catastro
    """
    print("\nCargando edificios [1/4]...")
    edificios = cargar_edificios()

    print("\nBuscando ortofotos [2/4]...")
    tifs = buscar_ortofotos()

    print("\nProcesando tifs [3/4]...")
    registros_totales = []
    for num, tif in enumerate(tifs):
        print(f"Procesando tif [{num + 1}/{len(tifs)}]: {tif.name}")
        registros_tif = procesar_tif(ruta_tif=tif, edificios=edificios)
        registros_totales.extend(registros_tif)

    print("\nGuardando manifest [4/4]...")
    guardar_manifest(registros=registros_totales)


if __name__ == "__main__":
    main()
