"""Índice de teselas PNOA que intersectan edificios de Catastro."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from paneles_solares.rutas import ruta_proyecto

RUTA_GPKG = ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg")
RUTA_MANIFEST = ruta_proyecto("data/manifests/teselas.csv")
CARPETA_PNOA = ruta_proyecto("data/pnoa")

TAM_TES = 512


def cargar_edificios(ruta: Path = RUTA_GPKG) -> gpd.GeoDataFrame:
    """Lee los edificios de Catastro.

    Args:
        ruta: Archivo GeoPackage con las geometrías de los edificios.

    Returns:
        Tabla de edificios con su sistema de coordenadas.
    """

    return gpd.read_file(ruta)


def buscar_ortofotos(carpeta: Path = CARPETA_PNOA) -> list[Path]:
    """Busca los archivos TIF de la carpeta PNOA.

    Args:
        carpeta: Carpeta donde están descargadas las ortofotos.

    Returns:
        Rutas de las ortofotos, ordenadas por nombre.
    """

    return sorted(carpeta.glob("*.tif"))


def obtener_teselas(tif, edificios: gpd.GeoDataFrame) -> list[dict]:
    """Recorre una ortofoto y registra las teselas que contienen edificios.

    Args:
        tif: Ortofoto abierta con rasterio.
        edificios: Edificios situados dentro de esta ortofoto.

    Returns:
        Registros con la posición, tamaño y número de edificios de cada tesela.
    """

    registros = []

    # El índice permite buscar edificios sin recorrer toda la tabla cada vez.
    indice_edificios = edificios.sindex
    ruta_tif = Path(tif.name)

    for fila in range(0, tif.height, TAM_TES):
        for columna in range(0, tif.width, TAM_TES):
            # En los bordes puede quedar una tesela menor que TAM_TES.
            ancho = min(TAM_TES, tif.width - columna)
            alto = min(TAM_TES, tif.height - fila)

            ventana = Window(col_off=columna, row_off=fila, width=ancho, height=alto)
            limites = tif.window_bounds(ventana)
            poligono = box(*limites)

            posiciones = indice_edificios.query(poligono, predicate="intersects")
            num_edificios = len(posiciones)

            if num_edificios == 0:
                continue

            registro = {
                "tile_id": f"{ruta_tif.stem}_f{fila}_c{columna}",
                "tif": ruta_tif.name,
                "fila": fila,
                "columna": columna,
                "ancho": ancho,
                "alto": alto,
                "num_edificios": num_edificios,
            }
            registros.append(registro)

    return registros


def procesar_tif(ruta_tif: Path, edificios: gpd.GeoDataFrame) -> list[dict]:
    """Busca primero los edificios de la ortofoto y después sus teselas.

    Args:
        ruta_tif: Archivo TIF que se va a consultar.
        edificios: Tabla completa de edificios de Catastro.

    Returns:
        Teselas de esta ortofoto que intersectan algún edificio.
    """

    with rasterio.open(ruta_tif) as tif:
        if edificios.crs != tif.crs:
            raise ValueError("Los CRS del gpkg y el tif no coinciden")

        poligono_tif = box(*tif.bounds)
        posiciones = edificios.sindex.query(poligono_tif, predicate="intersects")
        edificios_tif = edificios.iloc[posiciones]

        if edificios_tif.empty:
            return []

        return obtener_teselas(tif, edificios_tif)


def guardar_manifest(registros: list[dict], ruta: Path = RUTA_MANIFEST) -> None:
    """Guarda el índice de teselas como CSV.

    Args:
        registros: Teselas encontradas al recorrer las ortofotos.
        ruta: Archivo CSV de salida.
    """

    columnas = ["tile_id", "tif", "fila", "columna", "ancho", "alto", "num_edificios"]
    manifest = pd.DataFrame(registros, columns=columnas)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(ruta, index=False)


def main() -> None:
    """Crea el índice que se usa después para solicitar imágenes de etiquetado."""

    print("\nCargando edificios [1/4]...")
    edificios = cargar_edificios(RUTA_GPKG)

    print("\nBuscando ortofotos [2/4]...")
    tifs = buscar_ortofotos(CARPETA_PNOA)

    print("\nProcesando tifs [3/4]...")
    registros_totales = []

    for numero, tif in enumerate(tifs, start=1):
        print(f"Procesando tif [{numero}/{len(tifs)}]: {tif.name}")

        registros_tif = procesar_tif(tif, edificios)
        registros_totales.extend(registros_tif)

    print("\nGuardando manifest [4/4]...")
    guardar_manifest(registros_totales, RUTA_MANIFEST)


if __name__ == "__main__":
    main()
