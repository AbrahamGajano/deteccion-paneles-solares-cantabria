"""Crea una muestra reproducible con imágenes, nubes LAZ y formularios vacíos."""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.merge import merge
from rasterio.transform import array_bounds
from shapely.geometry import box

from paneles_solares.rutas import ruta_proyecto

PREPARACION = ruta_proyecto("data/lidar/preparacion")
RECORTES = ruta_proyecto("data/lidar/recortes_parciales")
ORTOFOTOS = ruta_proyecto("data/pnoa")
SALIDA = ruta_proyecto("data/lidar/revision")


def inventariar_casos(preparacion: Path, recortes: Path) -> gpd.GeoDataFrame:
    """Busca paneles con todas sus aportaciones LiDAR terminadas.

    Params:
        preparacion: Carpeta con recortes.gpkg y relaciones.csv.
        recortes: Carpeta con los resultados organizados por hoja.

    Returns:
        gpd.GeoDataFrame: Ventanas completas, geometría del panel y rutas LAZ.
    """

    relaciones = pd.read_csv(preparacion / "relaciones.csv")
    ventanas = gpd.read_file(preparacion / "recortes.gpkg", layer="recortes")
    paneles = gpd.read_file(preparacion / "recortes.gpkg", layer="paneles")
    resumenes = {}

    for ruta in sorted(recortes.glob("*/resumen.json")):
        resumenes[ruta.parent.name] = json.loads(ruta.read_text(encoding="utf-8"))

    disponibles = {}

    for panel_id, grupo in relaciones.groupby("id_panel", sort=True):
        rutas = []
        completo = True

        for hoja in grupo.id_hoja:
            resumen = resumenes.get(hoja)

            if resumen is None or panel_id not in resumen["puntos"]:
                completo = False
                break

            ruta = recortes / hoja / f"{panel_id}.laz"

            if resumen["puntos"][panel_id] > 0:
                if not ruta.exists():
                    completo = False
                    break
                rutas.append(ruta)

        if completo:
            disponibles[panel_id] = rutas

    casos = ventanas[ventanas.id_panel.isin(disponibles)].copy()
    casos["rutas_laz"] = casos.id_panel.map(disponibles)
    geometrias = paneles.set_index("id_panel").geometry
    casos["poligono"] = casos.id_panel.map(geometrias.to_wkt())

    return casos.sort_values("id_panel").reset_index(drop=True)


def indexar_ortofotos(carpeta: Path) -> gpd.GeoDataFrame:
    """Lee las extensiones de los TIFF sin cargar sus imágenes completas.

    Params:
        carpeta: Directorio de ortofotos locales en EPSG:25830.

    Returns:
        gpd.GeoDataFrame: Ruta y huella de cada ortofoto.
    """

    registros = []

    for ruta in sorted(carpeta.glob("*.tif")):
        with rasterio.open(ruta) as imagen:
            if imagen.crs.to_epsg() != 25830:
                raise ValueError(f"La ortofoto {ruta.name} no está en EPSG:25830.")
            registros.append({"ruta": ruta, "geometry": box(*imagen.bounds)})

    return gpd.GeoDataFrame(registros, columns=["ruta", "geometry"], crs=25830)


def extraer_imagen(ventana, ortofotos: gpd.GeoDataFrame, destino: Path) -> tuple:
    """Extrae RGB del contexto y une hojas de ortofoto si es necesario.

    Params:
        ventana: Fila con la geometría del cuadrado de contexto.
        ortofotos: Índice de los TIFF locales.
        destino: Ruta del PNG que se quiere guardar.

    Returns:
        tuple: Límites reales del PNG en metros: xmin, ymin, xmax, ymax.
    """

    rutas = ortofotos.loc[ortofotos.intersects(ventana.geometry), "ruta"].tolist()

    if not rutas:
        raise ValueError("No hay ortofoto local para este contexto.")

    datos, transformacion = merge(
        rutas, bounds=ventana.geometry.bounds, indexes=[1, 2, 3], masked=True
    )

    if np.ma.getmaskarray(datos).all(axis=0).any():
        raise ValueError("La ortofoto no cubre todo el contexto.")

    rgb = np.moveaxis(datos.filled(0), 0, -1).astype(np.uint8)
    Image.fromarray(rgb).save(destino)

    return array_bounds(rgb.shape[0], rgb.shape[1], transformacion)


def unir_nubes(rutas: list, destino: Path, crs) -> int:
    """Reúne las aportaciones del panel sin perder sus atributos.

    Params:
        rutas: LAZ del mismo panel procedentes de una o varias hojas.
        destino: Ruta del LAZ conjunto.
        crs: Referencia espacial usada si el caso no tiene ningún punto.

    Returns:
        int: Número de puntos guardados, sin duplicados exactos.
    """

    if not rutas:
        cabecera = laspy.LasHeader(point_format=8, version="1.4")
        cabecera.add_crs(crs)
        laspy.LasData(cabecera).write(destino)
        return 0

    nubes = [laspy.read(ruta) for ruta in rutas]
    cabecera = nubes[0].header.copy()
    escalas = np.min([nube.header.scales for nube in nubes], axis=0)
    cabecera.scales = escalas

    for nube in nubes:
        if nube.header.point_format != cabecera.point_format:
            raise ValueError("Las aportaciones tienen formatos de puntos diferentes.")
        if nube.header.parse_crs() != crs:
            raise ValueError("Las aportaciones tienen un CRS diferente al contexto.")
        nube.change_scaling(scales=escalas, offsets=cabecera.offsets)

    registros = np.concatenate([nube.points.array for nube in nubes])
    _, indices = np.unique(registros, return_index=True)
    registros = registros[np.sort(indices)]
    resultado = laspy.LasData(cabecera)
    resultado.points = laspy.ScaleAwarePointRecord(
        registros, cabecera.point_format, cabecera.scales, cabecera.offsets
    )
    resultado.write(destino)

    return len(registros)


def preparar_muestra(
    preparacion: Path, recortes: Path, ortofotos: Path,
    salida: Path, cantidad: int = 100, semilla: int = 42,
) -> Path:
    """Selecciona casos al azar y crea los archivos que leerá el visor.

    Params:
        preparacion: Carpeta del plan de recortes.
        recortes: Carpeta de aportaciones LiDAR descargadas.
        ortofotos: Carpeta de TIFF locales.
        salida: Carpeta nueva de revisión; una muestra existente se conserva.
        cantidad: Máximo de casos a seleccionar.
        semilla: Valor que permite repetir el mismo sorteo.

    Returns:
        Path: CSV con rutas relativas y una columna formulario para el JSON.
    """

    manifest = salida / "casos.csv"

    if manifest.exists():
        print(f"Se conserva la muestra y sus respuestas: {manifest}")
        return manifest

    casos = inventariar_casos(preparacion, recortes)

    if casos.empty:
        raise ValueError("Todavía no hay casos con todas sus hojas terminadas.")

    muestra = casos.sample(n=min(cantidad, len(casos)), random_state=semilla)
    indice = indexar_ortofotos(ortofotos)
    (salida / "imagenes").mkdir(parents=True, exist_ok=True)
    (salida / "nubes").mkdir(parents=True, exist_ok=True)
    filas = []
    errores = []

    for numero, caso in enumerate(muestra.itertuples(), start=1):
        print(f"[{numero}/{len(muestra)}] {caso.id_panel}", flush=True)
        imagen = Path("imagenes") / f"{caso.id_panel}.png"
        laz = Path("nubes") / f"{caso.id_panel}.laz"

        try:
            limites = extraer_imagen(caso, indice, salida / imagen)
            puntos = unir_nubes(caso.rutas_laz, salida / laz, casos.crs)
        except (ValueError, OSError, rasterio.errors.RasterioError,
                laspy.errors.LaspyException) as error:
            errores.append({"id_panel": caso.id_panel, "error": str(error)})
            print(f"  No preparado: {error}", flush=True)
            continue

        filas.append({
            "id_panel": caso.id_panel,
            "id_instalacion": caso.id_instalacion,
            "imagen": imagen.as_posix(),
            "laz": laz.as_posix(),
            "xmin": limites[0], "ymin": limites[1],
            "xmax": limites[2], "ymax": limites[3],
            "poligono": caso.poligono,
            "n_puntos": puntos,
            "formulario": "",
        })

    if errores:
        pd.DataFrame(errores).to_csv(salida / "errores.csv", index=False)

    if not filas:
        raise ValueError("No se pudo preparar ningún caso. Consulta errores.csv.")

    pd.DataFrame(filas).to_csv(manifest, index=False, encoding="utf-8")
    print(f"Casos completos disponibles: {len(casos)}. Muestra preparada: {len(filas)}.")
    print(f"CSV: {manifest}")

    return manifest


def main() -> None:
    """Prepara hasta 100 casos; usa otra carpeta para crear una muestra nueva."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cantidad", type=int, default=100)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--preparacion", type=Path, default=PREPARACION)
    parser.add_argument("--recortes", type=Path, default=RECORTES)
    parser.add_argument("--ortofotos", type=Path, default=ORTOFOTOS)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    if args.cantidad < 1:
        parser.error("--cantidad debe ser mayor que cero.")

    preparar_muestra(
        ruta_proyecto(args.preparacion), ruta_proyecto(args.recortes),
        ruta_proyecto(args.ortofotos), ruta_proyecto(args.salida),
        args.cantidad, args.semilla,
    )


if __name__ == "__main__":
    main()
