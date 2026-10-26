"""Descarga las edificaciones CBT23 y las parcelas catastrales de 2023."""

import hashlib
import json
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely import make_valid, normalize

from paneles_solares.rutas import EDIFICIOS_CATASTRO, huella_archivo, ruta_proyecto

REST = "https://geoservicios.cantabria.es/inspire/rest/services"
HOJAS = f"{REST}/Servicio_de_Descarga/MapServer/67"
PARCELAS = f"{REST}/Cdt/MapServer/15"
DESCARGAS = "https://geoservicios.cantabria.es/Series2/CBT23_LULC23"
ORIGINALES = ruta_proyecto("data/raw/catastro/2023")


def consultar(url: str, parametros: dict) -> dict:
    """Consulta el servicio y muestra también los errores que devuelve como JSON."""
    respuesta = requests.get(url, params={"f": "json", **parametros}, timeout=120)
    respuesta.raise_for_status()
    datos = respuesta.json()
    if "error" in datos:
        raise RuntimeError(f"{url}: {datos['error']}")
    return datos


def descargar_hoja(fila: dict, carpeta: Path) -> tuple[gpd.GeoDataFrame, dict]:
    """Reutiliza el ZIP de una hoja y lee sus edificios sin extraer otro GeoPackage."""
    destino = carpeta / "hojas" / fila["GPKG"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DESCARGAS}/GPKG/{destino.name}"

    if not destino.exists():
        temporal = destino.with_suffix(".part")
        with requests.get(url, stream=True, timeout=120) as respuesta:
            respuesta.raise_for_status()
            with temporal.open("wb") as archivo:
                for bloque in respuesta.iter_content(1024 * 1024):
                    archivo.write(bloque)
        temporal.replace(destino)

    with zipfile.ZipFile(destino) as archivo:
        nombre = next(n for n in archivo.namelist() if n.endswith(".gpkg"))
        licencia = carpeta / "licencia.txt"
        if not licencia.exists():
            licencia.write_text(
                archivo.read("Licencia de uso.txt").decode("cp1252"),
                encoding="utf-8",
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Measured .*geometry types.*")
        edificios = gpd.read_file(
            f"/vsizip/{destino.resolve().as_posix()}/{nombre}",
            layer="CBT23_Edificaciones",
            force_2d=True,
        )
    edificios["hoja_origen"] = fila["Num_Hoja"]
    fuente = {
        "hoja": fila["Num_Hoja"],
        "url": url,
        "sha256": huella_archivo(destino),
        "elementos": len(edificios),
    }
    return edificios, fuente


def descargar_pagina(inicio: int, carpeta: Path) -> Path:
    """Guarda una página del parcelario completo de 2023, si todavía falta."""
    destino = carpeta / f"{inicio:07d}.geojson"
    if not destino.exists():
        datos = consultar(
            PARCELAS + "/query",
            {
                "f": "geojson",
                "where": "1=1",
                "outSR": 25830,
                "outFields": "OBJECTID,MUNICIPIO,MASA,TIPO,PARCELA,ID_PARCELA",
                "orderByFields": "OBJECTID",
                "resultOffset": inicio,
                "resultRecordCount": 1000,
            },
        )
        temporal = destino.with_suffix(".part")
        temporal.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        temporal.replace(destino)
    return destino


def descargar_parcelas(carpeta: Path) -> gpd.GeoDataFrame:
    """Une las páginas del parcelario, conservando referencias y códigos oficiales."""
    carpeta.mkdir(parents=True, exist_ok=True)
    total = consultar(
        PARCELAS + "/query",
        {
            "where": "1=1",
            "returnCountOnly": "true",
        },
    )["count"]

    with ThreadPoolExecutor(max_workers=3) as ejecutor:
        paginas = ejecutor.map(
            lambda inicio: descargar_pagina(inicio, carpeta), range(0, total, 1000)
        )
        tablas = [gpd.read_file(ruta) for ruta in paginas]
    parcelas = gpd.GeoDataFrame(pd.concat(tablas, ignore_index=True), crs=25830)
    if len(parcelas) != total or parcelas.OBJECTID.duplicated().any():
        raise ValueError(
            "La descarga del parcelario está incompleta o tiene duplicados"
        )

    parcelas = parcelas.rename(
        columns={
            "ID_PARCELA": "referencia_catastral",
            "MUNICIPIO": "municipio_catastro",
        }
    )
    parcelas.geometry = make_valid(parcelas.geometry)
    parcelas["id_parcela"] = "cp2023_" + parcelas.OBJECTID.astype(str)
    parcelas["anio_fuente"] = 2023
    return parcelas


def unir_edificios(tablas: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Crea identificadores estables y elimina polígonos repetidos entre hojas."""
    edificios = gpd.GeoDataFrame(pd.concat(tablas, ignore_index=True), crs=25830)
    edificios.geometry = normalize(make_valid(edificios.geometry))
    edificios = edificios.explode(index_parts=False).reset_index(drop=True)
    edificios = edificios[edificios.geom_type == "Polygon"].copy()
    edificios["id_edificio"] = edificios.geometry.apply(
        lambda geometria: "cbt23_" + hashlib.sha256(geometria.wkb).hexdigest()[:24]
    )
    edificios = edificios.drop_duplicates("id_edificio").reset_index(drop=True)
    edificios["anio_fuente"] = 2023
    edificios["fuente"] = "Gobierno de Cantabria, CBT23, derivado de Catastro"
    edificios["uso_fuente"] = "CBT23/LULC23; uso cartográfico, no uso registral"
    return edificios


def relacionar(objetos, entidades, id_objeto, id_entidad):
    """Relaciona polígonos por solape, conservando todos los candidatos."""
    parejas = gpd.sjoin(
        objetos[[id_objeto, "geometry"]],
        entidades[[id_entidad, "geometry"]],
        how="inner",
        predicate="intersects",
    )
    geometrias = entidades.geometry.loc[parejas.index_right]
    intersecciones = parejas.geometry.intersection(geometrias, align=False)
    parejas["interseccion_m2"] = intersecciones.area
    parejas["fraccion_objeto"] = parejas.interseccion_m2 / parejas.geometry.area
    columnas = [id_objeto, id_entidad, "interseccion_m2", "fraccion_objeto"]
    return parejas.loc[parejas.interseccion_m2 > 0, columnas].reset_index(drop=True)


def preparar(salida: Path = EDIFICIOS_CATASTRO, originales: Path = ORIGINALES) -> None:
    """Prepara la referencia regional. No modifica etiquetas ni resultados del modelo."""
    originales.mkdir(parents=True, exist_ok=True)
    indice = consultar(
        HOJAS + "/query",
        {
            "where": "1=1",
            "outFields": "GPKG,Num_Hoja",
            "returnGeometry": "false",
            "orderByFields": "OBJECTID",
            "resultRecordCount": 2000,
        },
    )
    if indice.get("exceededTransferLimit"):
        raise ValueError("El índice de hojas CBT23 está incompleto")

    filas = [elemento["attributes"] for elemento in indice["features"]]
    with ThreadPoolExecutor(max_workers=3) as ejecutor:
        resultados = list(
            ejecutor.map(lambda fila: descargar_hoja(fila, originales), filas)
        )
    tablas, fuentes = zip(*resultados)
    edificios = unir_edificios(list(tablas))
    parcelas = descargar_parcelas(originales / "parcelas")

    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = salida.with_name(".catastro_preparando.gpkg")
    temporal.unlink(missing_ok=True)
    edificios.to_file(temporal, layer="edificios", index=False)
    parcelas.to_file(temporal, layer="parcelas", index=False)
    relaciones = relacionar(edificios, parcelas, "id_edificio", "id_parcela")
    relaciones = relaciones.merge(parcelas[["id_parcela", "referencia_catastral"]])
    relaciones.to_csv(salida.parent / "edificios_parcelas.csv", index=False)
    temporal.replace(salida)

    pd.DataFrame(fuentes).to_csv(originales / "descargas.csv", index=False)
    fuente = {
        "anio_referencia": 2023,
        "descargado_utc": datetime.now(timezone.utc).isoformat(),
        "edificios": len(edificios),
        "parcelas": len(parcelas),
        "hojas": len(filas),
        "edificaciones": DESCARGAS,
        "parcelario": PARCELAS,
        "metodologia": f"{DESCARGAS}/DOC_AUX/Documentacion_LULC_Cantabria_v2_0.pdf",
        "fecha_edificaciones": "Imágenes de agosto y LiDAR de octubre de 2023",
        "fecha_parcelario": "Principios de 2023; anuncio del 01-02-2023",
        "limitacion": "CBT23 es cartografía derivada, no un volcado INSPIRE BU histórico.",
        "licencia": str(originales / "licencia.txt"),
        "sha256_gpkg": huella_archivo(salida),
    }
    salida.with_suffix(".json").write_text(
        json.dumps(fuente, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Referencia de 2023: {len(edificios)} unidades y {len(parcelas)} parcelas")


if __name__ == "__main__":
    preparar()
