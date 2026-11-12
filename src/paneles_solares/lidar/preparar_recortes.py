"""Prepara cuadrados de contexto y las hojas LiDAR necesarias para cubrirlos."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from paneles_solares.rutas import ruta_proyecto

PANELES = ruta_proyecto("runs/indicadores/seguimiento_historico/atribucion.gpkg")
CAPA = "paneles"
ANHO = 2023
MARGEN = 10
SALIDA = ruta_proyecto("data/lidar/preparacion")

CATALOGO = (
    "https://geoservicios.cantabria.es/inspire/rest/services/"
    "Servicio_de_Descarga/MapServer/68/query"
)
DESCARGAS = "https://geoservicios.cantabria.es/Series2/LIDAR_2023/LASZ"


def cargar_paneles(ruta: Path, capa: str, anho: int) -> gpd.GeoDataFrame:
    """Lee los IDs y polígonos de la capa y campaña indicadas, en EPSG:25830."""

    paneles = gpd.read_file(ruta, layer=capa, where=f"anio = {anho}")

    return paneles[["id_panel", "id_instalacion", "anio", "geometry"]].to_crs(25830)


def crear_recortes(paneles: gpd.GeoDataFrame, margen: float) -> gpd.GeoDataFrame:
    """Construye un cuadrado de contexto para cada detección.

    Params:
        paneles: Polígonos originales en EPSG:25830.
        margen: Contexto mínimo, en metros, a cada lado de la detección.

    Returns:
        gpd.GeoDataFrame: Cuadrados con sus identificadores y límites métricos.
    """

    recortes = paneles.drop(columns="geometry").copy()
    cuadrados = []

    for geom in paneles.geometry:
        xmin, ymin, xmax, ymax = geom.bounds

        centro_x = (xmin + xmax) / 2
        centro_y = (ymin + ymax) / 2
        lado = max(xmax - xmin, ymax - ymin) + 2 * margen

        cuadrado = box(
            centro_x - lado / 2,
            centro_y - lado / 2,
            centro_x + lado / 2,
            centro_y + lado / 2,
        )
        cuadrados.append(cuadrado)

    recortes = gpd.GeoDataFrame(recortes, geometry=cuadrados, crs=paneles.crs)
    recortes["margen_m"] = margen
    recortes[["xmin", "ymin", "xmax", "ymax"]] = recortes.geometry.bounds

    return recortes


def consultar_hojas() -> gpd.GeoDataFrame:
    """Consulta las huellas y los nombres de archivo del catálogo oficial de 2023.

    Returns:
        gpd.GeoDataFrame: Hojas disponibles en el índice, con su URL de descarga.
    """

    registros = []
    desplazamiento = 0

    while True:
        respuesta = requests.get(
            CATALOGO,
            params={
                "where": "1=1",
                "outFields": "OBJECTID,LASZ",
                "returnGeometry": "true",
                "outSR": 25830,
                "orderByFields": "OBJECTID",
                "resultOffset": desplazamiento,
                "resultRecordCount": 1000,
                "f": "geojson",
            },
            timeout=60,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()

        if "error" in datos:
            raise RuntimeError(datos["error"])

        registros.extend(datos["features"])

        if not datos.get("exceededTransferLimit", False):
            break

        desplazamiento += len(datos["features"])

    hojas = gpd.GeoDataFrame.from_features(registros, crs="EPSG:25830")
    hojas = hojas.rename(columns={"LASZ": "nombre_zip"})
    hojas["id_hoja"] = hojas.nombre_zip.str.removesuffix(".zip")
    hojas["url"] = DESCARGAS + "/" + hojas.nombre_zip

    return hojas[["id_hoja", "nombre_zip", "url", "geometry"]]


def relacionar_hojas(
    recortes: gpd.GeoDataFrame,
    hojas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Relaciona cada cuadrado con todas las hojas necesarias para cubrirlo.

    Params:
        recortes: Cuadrados de contexto de las detecciones.
        hojas: Huellas oficiales del catálogo LiDAR.

    Returns:
        pd.DataFrame: Una fila por pareja de panel y hoja, sin duplicados.
    """

    cruces = gpd.sjoin(
        recortes[["id_panel", "geometry"]],
        hojas[["id_hoja", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    return (
        pd.DataFrame(cruces[["id_panel", "id_hoja"]])
        .drop_duplicates()
        .sort_values(["id_hoja", "id_panel"])
        .reset_index(drop=True)
    )


def guardar_preparacion(
    paneles: gpd.GeoDataFrame,
    recortes: gpd.GeoDataFrame,
    hojas: gpd.GeoDataFrame,
    relaciones: pd.DataFrame,
    salida: Path,
) -> None:
    """Guarda las geometrías y el plan de descarga, sin descargar ningún ZIP.

    Params:
        paneles: Detecciones originales de la campaña.
        recortes: Cuadrados que se usarán para recortar LiDAR y ortofotos.
        hojas: Catálogo oficial de hojas LiDAR.
        relaciones: Parejas de panel y hoja necesarias.
        salida: Carpeta donde guardar el GeoPackage y los CSV.
    """

    salida.mkdir(parents=True, exist_ok=True)

    necesarias = hojas[hojas.id_hoja.isin(relaciones.id_hoja)].copy()
    necesarias = necesarias.sort_values("id_hoja")
    recuentos = relaciones.groupby("id_hoja").size()
    necesarias["n_recortes"] = necesarias.id_hoja.map(recuentos)

    geopackage = salida / "recortes.gpkg"
    paneles.to_file(geopackage, layer="paneles", driver="GPKG", index=False)
    recortes.to_file(geopackage, layer="recortes", driver="GPKG", index=False)
    necesarias.to_file(geopackage, layer="hojas", driver="GPKG", index=False)

    relaciones.to_csv(salida / "relaciones.csv", index=False, encoding="utf-8")
    necesarias.drop(columns="geometry").to_csv(
        salida / "hojas_descarga.csv", index=False, encoding="utf-8"
    )

    sin_hoja = recortes[~recortes.id_panel.isin(relaciones.id_panel)]
    print(f"Paneles: {len(paneles)}")
    print(f"Hojas necesarias: {len(necesarias)}")
    print(f"Recortes sin hoja en el catálogo: {len(sin_hoja)}")
    print(f"Preparación guardada en: {salida}")


def main() -> None:
    """Prepara las ventanas de 2023 y su plan de descarga LiDAR."""

    paneles = cargar_paneles(PANELES, CAPA, ANHO)
    recortes = crear_recortes(paneles, MARGEN)
    hojas = consultar_hojas()
    relaciones = relacionar_hojas(recortes, hojas)

    guardar_preparacion(paneles, recortes, hojas, relaciones, SALIDA)


if __name__ == "__main__":
    main()
