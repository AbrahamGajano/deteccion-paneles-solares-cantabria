"""Descarga hojas LiDAR y conserva sus aportaciones a los cuadrados de contexto.

Cada hoja tiene una carpeta de salida. Si un panel cruza dos hojas, sus dos
recortes se guardan por separado; este módulo todavía no los une.
"""

import argparse
import ctypes
import json
import logging
import os
import time
from contextlib import ExitStack
from pathlib import Path
from shutil import copyfileobj, disk_usage
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
INFORMES = ruta_proyecto("runs/lidar/descarga")
LOGGER = logging.getLogger(__name__)


class EspacioInsuficiente(RuntimeError):
    """Indica que hay que detener el lote para conservar espacio libre."""


def comprobar_espacio(carpeta: Path, reserva_gb: float, necesario: int = 0) -> None:
    """Comprueba que una escritura dejará la reserva de disco configurada.

    Params:
        carpeta: Directorio existente en el disco que se quiere comprobar.
        reserva_gb: Espacio libre que se quiere conservar, en GiB.
        necesario: Bytes adicionales que se prevé escribir.
    """

    libre = disk_usage(carpeta).free

    if libre - necesario < reserva_gb * 1024**3:
        raise EspacioInsuficiente(
            f"Quedan {libre / 1024**3:.2f} GiB libres en {carpeta}. "
            f"Se reservan {reserva_gb:g} GiB para el sistema."
        )


def descargar_zip(url: str, destino: Path, reserva_gb: float = 5) -> Path:
    """Descarga el ZIP por bloques directamente a su ruta de destino.

    Params:
        url: Enlace guardado en hojas_descarga.csv.
        destino: Ruta del ZIP, incluido su nombre de archivo.
        reserva_gb: Espacio libre mínimo que debe conservar el disco.

    Returns:
        Path: Ruta del ZIP descargado en disco.
    """

    destino.parent.mkdir(parents=True, exist_ok=True)
    comprobar_espacio(destino.parent, reserva_gb)

    with requests.get(url, stream=True, timeout=120) as respuesta:
        respuesta.raise_for_status()
        comprobar_espacio(
            destino.parent, reserva_gb, int(respuesta.headers.get("Content-Length", 0))
        )

        with destino.open("wb") as archivo:
            for bloque in respuesta.iter_content(chunk_size=1024**2):
                if bloque:
                    comprobar_espacio(destino.parent, reserva_gb, len(bloque))
                    archivo.write(bloque)

    return destino


def extraer_laz(ruta_zip: Path, carpeta: Path, reserva_gb: float = 5) -> Path:
    """Extrae únicamente el LAZ y lo guarda con el nombre de la hoja.

    Params:
        ruta_zip: ZIP descargado que contiene un único LAZ.
        carpeta: Directorio temporal de extracción.
        reserva_gb: Espacio libre mínimo que debe conservar el disco.

    Returns:
        Path: Ruta del LAZ extraído en disco.
    """

    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{ruta_zip.stem}.laz"

    with ZipFile(ruta_zip) as archivo:
        archivos_laz = [
            nombre for nombre in archivo.namelist()
            if nombre.lower().endswith(".laz")
        ]

        if len(archivos_laz) != 1:
            raise ValueError(f"Se esperaba un LAZ en {ruta_zip.name}.")

        comprobar_espacio(carpeta, reserva_gb, archivo.getinfo(archivos_laz[0]).file_size)

        with archivo.open(archivos_laz[0]) as origen, destino.open("wb") as salida:
            copyfileobj(origen, salida, length=1024**2)

    return destino


def recortar_hoja(
    ruta_laz: Path,
    ventanas: pd.DataFrame,
    carpeta: Path,
    puntos_por_bloque: int = PUNTOS_POR_BLOQUE,
    reserva_gb: float = 5,
) -> dict:
    """Escribe los puntos de cada ventana sin cargar la hoja completa en RAM.

    Params:
        ruta_laz: Nube original de una hoja.
        ventanas: Tabla con id_panel, xmin, ymin, xmax e ymax en EPSG:25830.
        carpeta: Directorio de recortes parciales de esta hoja.
        puntos_por_bloque: Número máximo de puntos leídos en cada iteración.
        reserva_gb: Espacio libre mínimo que debe conservar el disco.

    Returns:
        dict: Número de puntos escritos por panel, incluidos los recortes vacíos.
    """

    carpeta.mkdir(parents=True, exist_ok=True)
    escritores = {}
    recuentos = dict.fromkeys(ventanas.id_panel, 0)

    with laspy.open(ruta_laz) as lector, ExitStack() as archivos:
        crs = lector.header.parse_crs()

        if crs is None or crs.to_epsg() != 25830:
            raise ValueError("La hoja LiDAR debe estar en EPSG:25830.")

        for puntos in lector.chunk_iterator(puntos_por_bloque):
            comprobar_espacio(carpeta, reserva_gb)
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
                    destino = carpeta / f"{panel_id}.laz"
                    escritores[panel_id] = archivos.enter_context(
                        laspy.open(
                            destino, mode="w", header=lector.header,
                            do_compress=True,
                            laz_backend=laspy.LazBackend.Lazrs,
                        )
                    )

                escritores[panel_id].write_points(puntos[dentro])
                recuentos[panel_id] += cantidad

        if lector.header.evlrs:
            for escritor in escritores.values():
                escritor.write_evlrs(lector.header.evlrs)

    return recuentos


def procesar_hoja(
    hoja: pd.Series,
    ventanas: pd.DataFrame,
    temporales: Path,
    salida: Path,
    reserva_gb: float = 5,
) -> None:
    """Procesa una hoja y limpia sus ZIP y LAZ temporales, incluso si falla.

    Params:
        hoja: Fila con id_hoja, nombre_zip y url.
        ventanas: Cuadrados asociados a esta hoja, con sus límites métricos.
        temporales: Carpeta para los ZIP y LAZ originales.
        salida: Carpeta raíz de los recortes parciales.
        reserva_gb: Espacio libre mínimo que debe conservar el disco.
    """

    carpeta = salida / hoja.id_hoja
    resumen = carpeta / "resumen.json"
    plan = {"url": hoja.url, "ventanas": ventanas.to_dict(orient="records")}
    ruta_zip = temporales / hoja.nombre_zip
    ruta_laz = temporales / f"{Path(hoja.nombre_zip).stem}.laz"

    if resumen.exists():
        try:
            anterior = json.loads(resumen.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            anterior = None

        if anterior is not None:
            if anterior["plan"] != plan:
                raise ValueError("El plan cambió: utiliza otra carpeta con --salida.")

            completos = set(anterior["puntos"]) == set(ventanas.id_panel)

            for panel_id, cantidad in anterior["puntos"].items():
                if not cantidad:
                    continue
                try:
                    with laspy.open(carpeta / f"{panel_id}.laz") as lector:
                        completos = completos and lector.header.point_count == cantidad
                except (OSError, laspy.errors.LaspyException):
                    completos = False

            if completos:
                ruta_zip.unlink(missing_ok=True)
                ruta_laz.unlink(missing_ok=True)
                LOGGER.info("  Ya terminada.")
                return

    resumen.unlink(missing_ok=True)

    try:
        LOGGER.info("  Descargando ZIP...")
        descargar_zip(hoja.url, ruta_zip, reserva_gb)
        LOGGER.info("  Extrayendo LAZ...")
        extraer_laz(ruta_zip, temporales, reserva_gb)
        LOGGER.info("  Recortando %s ventanas...", len(ventanas))
        recuentos = recortar_hoja(ruta_laz, ventanas, carpeta, reserva_gb=reserva_gb)

        resumen.write_text(
            json.dumps({"plan": plan, "puntos": recuentos}, indent=2), encoding="utf-8"
        )
    finally:
        ruta_laz.unlink(missing_ok=True)
        ruta_zip.unlink(missing_ok=True)

    vacios = sum(cantidad == 0 for cantidad in recuentos.values())
    LOGGER.info("  Terminada: %s puntos; %s vacíos.", sum(recuentos.values()), vacios)


def procesar_con_reintentos(
    hoja: pd.Series, ventanas: pd.DataFrame, temporales: Path, salida: Path,
    reintentos: int, reserva_gb: float,
) -> str:
    """Repite errores temporales y devuelve el error si la hoja queda pendiente.

    Params:
        hoja: Fila del plan de descarga.
        ventanas: Cuadrados asociados a la hoja.
        temporales: Directorio de trabajo para una sola hoja.
        salida: Directorio de los recortes parciales.
        reintentos: Número máximo de intentos por hoja, incluido el primero.
        reserva_gb: Espacio libre mínimo que debe conservar el disco.

    Returns:
        str: Texto vacío si terminó bien; motivo del fallo en otro caso.
    """

    for intento in range(1, reintentos + 1):
        try:
            procesar_hoja(hoja, ventanas, temporales, salida, reserva_gb)
            return ""
        except (requests.RequestException, BadZipFile, OSError, ValueError,
                EOFError, laspy.errors.LaspyException) as error:
            if isinstance(error, OSError) and (
                error.errno == 28 or getattr(error, "winerror", None) == 112
            ):
                raise EspacioInsuficiente("El sistema ha indicado que el disco está lleno.") from error

            LOGGER.error("  Intento %s/%s: %s", intento, reintentos, error)
            permanente = isinstance(error, ValueError)

            if isinstance(error, requests.HTTPError):
                permanente = error.response.status_code not in (408, 429, 500, 502, 503, 504)

            if permanente or intento == reintentos:
                return f"{type(error).__name__}: {error}"

            time.sleep(10)

    return ""


def main() -> None:
    """Lee la preparación y procesa una hoja por defecto, o todas con --limite 0."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=1, help="Hojas a procesar; 0 = todas.")
    parser.add_argument("--hoja", help="Identificador de una hoja concreta.")
    parser.add_argument("--preparacion", type=Path, default=PREPARACION)
    parser.add_argument("--temporales", type=Path, default=TEMPORALES)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--reintentos", type=int, default=3, help="Intentos máximos por hoja.")
    parser.add_argument("--reserva-gb", type=float, default=5, help="GiB libres que se deben conservar.")
    parser.add_argument("--informes", type=Path, default=INFORMES)
    args = parser.parse_args()

    if args.limite < 0:
        parser.error("--limite debe ser 0 o un número positivo.")

    if args.reintentos < 1 or args.reserva_gb < 0:
        parser.error("Los intentos deben ser positivos y la reserva no puede ser negativa.")

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

    temporales = ruta_proyecto(args.temporales)
    salida = ruta_proyecto(args.salida)
    informes = ruta_proyecto(args.informes)

    for carpeta in (temporales, salida, informes):
        carpeta.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(informes / "descarga.log", encoding="utf-8"),
        ],
    )
    archivo_errores = informes / "errores.csv"
    errores = {}

    if archivo_errores.exists():
        anteriores = pd.read_csv(archivo_errores, keep_default_na=False)
        errores = dict(zip(anteriores.id_hoja, anteriores.error))

    energia = ctypes.windll.kernel32 if os.name == "nt" else None

    if energia is not None:
        energia.SetThreadExecutionState(0x80000001)

    try:
        for numero, (_, hoja) in enumerate(hojas.iterrows(), start=1):
            LOGGER.info("[%s/%s] %s", numero, len(hojas), hoja.id_hoja)
            paneles = relaciones.loc[relaciones.id_hoja == hoja.id_hoja, "id_panel"]
            ventanas = recortes.loc[
                recortes.id_panel.isin(paneles),
                ["id_panel", "xmin", "ymin", "xmax", "ymax"],
            ].sort_values("id_panel")

            detener = False

            try:
                error = procesar_con_reintentos(
                    hoja, ventanas, temporales, salida, args.reintentos, args.reserva_gb
                )
            except EspacioInsuficiente as fallo:
                error = str(fallo)
                detener = True

            if error:
                errores[hoja.id_hoja] = error
            else:
                errores.pop(hoja.id_hoja, None)

            pd.DataFrame(
                list(errores.items()), columns=["id_hoja", "error"]
            ).to_csv(archivo_errores, index=False, encoding="utf-8")

            if detener:
                LOGGER.error("Lote detenido por espacio: %s", error)
                raise SystemExit("Libera espacio y ejecuta el mismo comando para continuar.")
    finally:
        if energia is not None:
            energia.SetThreadExecutionState(0x80000000)

    pendientes = set(hojas.id_hoja) & set(errores)

    if pendientes:
        raise SystemExit(f"Hojas pendientes: {len(pendientes)}. Consulta {archivo_errores}")

    LOGGER.info("Lote terminado. Todas las hojas seleccionadas están completas.")


if __name__ == "__main__":
    main()
