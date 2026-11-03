"""Descarga hojas LiDAR y conserva sus aportaciones a los cuadrados de contexto.

Cada hoja tiene una carpeta de salida. Si un panel cruza dos hojas, sus dos
recortes se guardan por separado; este módulo todavía no los une.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from shutil import copyfileobj
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import requests

from paneles_solares.rutas import ruta_proyecto

PREPARACION = ruta_proyecto("data/lidar/preparacion")
TEMPORALES = ruta_proyecto("data/lidar/temporales")
SALIDA = ruta_proyecto("data/lidar/recortes_parciales")
PUNTOS_POR_BLOQUE = 500_000


def descargar_zip(url: str, destino: Path) -> Path:
    """Descarga el ZIP por bloques o reutiliza una descarga terminada.

    Params:
        url: Enlace guardado en hojas_descarga.csv.
        destino: Ruta del ZIP, incluido su nombre de archivo.

    Returns:
        Path: Ruta del ZIP descargado en disco.
    """

    if destino.exists():
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(".zip.part")

    with requests.get(url, stream=True, timeout=120) as respuesta:
        respuesta.raise_for_status()

        with parcial.open("wb") as archivo:
            for bloque in respuesta.iter_content(chunk_size=1024**2):
                if bloque:
                    archivo.write(bloque)

    parcial.replace(destino)

    return destino


def extraer_laz(ruta_zip: Path, carpeta: Path) -> Path:
    """Extrae únicamente el LAZ y lo guarda con el nombre de la hoja.

    Params:
        ruta_zip: ZIP descargado que contiene un único LAZ.
        carpeta: Directorio temporal de extracción.

    Returns:
        Path: Ruta del LAZ extraído en disco.
    """

    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{ruta_zip.stem}.laz"

    if destino.exists():
        return destino

    with ZipFile(ruta_zip) as archivo:
        archivos_laz = [
            nombre for nombre in archivo.namelist()
            if nombre.lower().endswith(".laz")
        ]

        if len(archivos_laz) != 1:
            raise ValueError(f"Se esperaba un LAZ en {ruta_zip.name}.")

        parcial = destino.with_suffix(".laz.part")

        with archivo.open(archivos_laz[0]) as origen, parcial.open("wb") as salida:
            copyfileobj(origen, salida, length=1024**2)

    parcial.replace(destino)

    return destino


def recortar_hoja(
    ruta_laz: Path,
    ventanas: pd.DataFrame,
    carpeta: Path,
    puntos_por_bloque: int = PUNTOS_POR_BLOQUE,
) -> dict:
    """Escribe los puntos de cada ventana sin cargar la hoja completa en RAM.

    Params:
        ruta_laz: Nube original de una hoja.
        ventanas: Tabla con id_panel, xmin, ymin, xmax e ymax en EPSG:25830.
        carpeta: Directorio de recortes parciales de esta hoja.
        puntos_por_bloque: Número máximo de puntos leídos en cada iteración.

    Returns:
        dict: Número de puntos escritos por panel, incluidos los recortes vacíos.
    """

    carpeta.mkdir(parents=True, exist_ok=True)
    escritores = {}
    recuentos = {panel_id: 0 for panel_id in ventanas.id_panel}

    # ExitStack cierra todos los escritores, también si ocurre un error.
    with laspy.open(ruta_laz) as lector, ExitStack() as archivos:
        crs = lector.header.parse_crs()

        if crs is None or crs.to_epsg() != 25830:
            raise ValueError("La hoja LiDAR debe estar en EPSG:25830.")

        for puntos in lector.chunk_iterator(puntos_por_bloque):
            x = np.asarray(puntos.x)
            y = np.asarray(puntos.y)

            for ventana in ventanas.itertuples(index=False):
                dentro = (
                    (x >= ventana.xmin)
                    & (x <= ventana.xmax)
                    & (y >= ventana.ymin)
                    & (y <= ventana.ymax)
                )
                cantidad = int(dentro.sum())

                if cantidad == 0:
                    continue

                panel_id = ventana.id_panel

                if panel_id not in escritores:
                    destino = carpeta / f"{panel_id}.laz.part"
                    escritores[panel_id] = archivos.enter_context(
                        laspy.open(
                            destino, mode="w", header=lector.header,
                            do_compress=True,
                            laz_backend=laspy.LazBackend.Lazrs,
                        )
                    )

                # Se conservan todos los atributos, sin filtrar clases ni alturas.
                escritores[panel_id].write_points(puntos[dentro])
                recuentos[panel_id] += cantidad

        if lector.header.evlrs:
            for escritor in escritores.values():
                escritor.write_evlrs(lector.header.evlrs)

    # Los nombres definitivos solo se asignan después de cerrar los escritores.
    for panel_id, cantidad in recuentos.items():
        if cantidad:
            parcial = carpeta / f"{panel_id}.laz.part"
            parcial.replace(carpeta / f"{panel_id}.laz")

    return recuentos


def procesar_hoja(
    hoja: pd.Series,
    ventanas: pd.DataFrame,
    temporales: Path,
    salida: Path,
) -> None:
    """Procesa una hoja y elimina sus temporales únicamente al terminar bien.

    Params:
        hoja: Fila con id_hoja, nombre_zip y url.
        ventanas: Cuadrados asociados a esta hoja, con sus límites métricos.
        temporales: Carpeta para los ZIP y LAZ originales.
        salida: Carpeta raíz de los recortes parciales.
    """

    carpeta = salida / hoja.id_hoja
    resumen = carpeta / "resumen.json"
    plan = {"url": hoja.url, "ventanas": ventanas.to_dict(orient="records")}

    if resumen.exists():
        anterior = json.loads(resumen.read_text(encoding="utf-8"))

        if anterior["plan"] != plan:
            raise ValueError("El plan cambió: utiliza otra carpeta con --salida.")

        completos = all(
            (carpeta / f"{panel_id}.laz").exists()
            for panel_id, cantidad in anterior["puntos"].items() if cantidad
        )

        if completos:
            print("  Ya terminada.", flush=True)
            return

    print("  Descargando ZIP...", flush=True)
    ruta_zip = descargar_zip(hoja.url, temporales / hoja.nombre_zip)
    print("  Extrayendo LAZ...", flush=True)
    ruta_laz = extraer_laz(ruta_zip, temporales)
    print(f"  Recortando {len(ventanas)} ventanas...", flush=True)
    recuentos = recortar_hoja(ruta_laz, ventanas, carpeta)

    # Esta marca permite omitir las hojas completadas al ejecutar de nuevo.
    parcial = resumen.with_suffix(".json.part")
    parcial.write_text(
        json.dumps({"plan": plan, "puntos": recuentos}, indent=2), encoding="utf-8"
    )
    parcial.replace(resumen)

    ruta_laz.unlink()
    ruta_zip.unlink()

    vacios = sum(cantidad == 0 for cantidad in recuentos.values())
    print(f"  Terminada: {sum(recuentos.values())} puntos; {vacios} vacíos.", flush=True)


def main() -> None:
    """Lee la preparación y procesa una hoja por defecto, o todas con --limite 0."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=1, help="Hojas a procesar; 0 = todas.")
    parser.add_argument("--hoja", help="Identificador de una hoja concreta.")
    parser.add_argument("--preparacion", type=Path, default=PREPARACION)
    parser.add_argument("--temporales", type=Path, default=TEMPORALES)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    if args.limite < 0:
        parser.error("--limite debe ser 0 o un número positivo.")

    preparacion = ruta_proyecto(args.preparacion)
    hojas = pd.read_csv(preparacion / "hojas_descarga.csv")
    relaciones = pd.read_csv(preparacion / "relaciones.csv")
    recortes = gpd.read_file(preparacion / "recortes.gpkg", layer="recortes")

    if args.hoja:
        hojas = hojas[hojas.id_hoja == args.hoja]

        if hojas.empty:
            parser.error("La hoja indicada no aparece en hojas_descarga.csv.")

    if args.limite:
        hojas = hojas.head(args.limite)

    errores = []

    for numero, (_, hoja) in enumerate(hojas.iterrows(), start=1):
        print(f"[{numero}/{len(hojas)}] {hoja.id_hoja}", flush=True)
        paneles = relaciones.loc[relaciones.id_hoja == hoja.id_hoja, "id_panel"]
        ventanas = recortes.loc[
            recortes.id_panel.isin(paneles),
            ["id_panel", "xmin", "ymin", "xmax", "ymax"],
        ].sort_values("id_panel")

        try:
            procesar_hoja(
                hoja, ventanas, ruta_proyecto(args.temporales), ruta_proyecto(args.salida)
            )
        except (requests.RequestException, BadZipFile, OSError, ValueError,
                laspy.errors.LaspyException) as error:
            errores.append(hoja.id_hoja)
            print(f"  ERROR: {error}", flush=True)

    if errores:
        raise SystemExit(f"Hojas pendientes ({len(errores)}): {', '.join(errores)}")


if __name__ == "__main__":
    main()
