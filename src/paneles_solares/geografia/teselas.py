"""Crea el índice PNOA de todo el territorio, incluidas zonas sin edificios."""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely import area, box, intersection, union_all

from paneles_solares.rutas import (
    EDIFICIOS_CATASTRO,
    INDICE_TESELAS,
    MUNICIPIOS_CANTABRIA,
    PNOA,
    huella_archivo,
    ruta_proyecto,
)

TAM_TES = 512


def obtener_teselas(tif, edificios, territorio=None) -> pd.DataFrame:
    """Describe ventanas de 512 píxeles sin descartarlas por ausencia de edificios."""
    if edificios.crs != tif.crs or not np.allclose(tif.res, (0.15, 0.15)):
        raise ValueError(f"Se espera EPSG:25830 y 0,15 m: {tif.name}")

    filas, columnas = np.meshgrid(
        np.arange(0, tif.height, TAM_TES),
        np.arange(0, tif.width, TAM_TES),
        indexing="ij",
    )
    tabla = pd.DataFrame({"fila": filas.ravel(), "columna": columnas.ravel()})
    tabla["ancho"] = np.minimum(TAM_TES, tif.width - tabla.columna)
    tabla["alto"] = np.minimum(TAM_TES, tif.height - tabla.fila)

    transformacion = tif.transform
    xmin = transformacion.c + tabla.columna * transformacion.a
    ymax = transformacion.f + tabla.fila * transformacion.e
    geometrias = box(
        xmin, ymax + tabla.alto * transformacion.e,
        xmin + tabla.ancho * transformacion.a, ymax,
    )
    if territorio is not None:
        activas = area(intersection(geometrias, territorio)) > 0
        tabla = tabla.loc[activas].reset_index(drop=True)
        geometrias = geometrias[activas]

    parejas = edificios.sindex.query(geometrias, predicate="intersects")
    tabla["num_edificios"] = np.bincount(parejas[0], minlength=len(tabla))
    nombre = Path(tif.name)
    tabla["tile_id"] = [
        f"{nombre.stem}_f{fila}_c{columna}"
        for fila, columna in zip(tabla.fila, tabla.columna)
    ]
    tabla["tif"] = nombre.name
    tabla["prioridad"] = np.where(
        tabla.num_edificios > 0, "edificaciones", "resto_territorio"
    )
    tabla["anio_catastro"] = 2023
    return tabla[[
        "tile_id", "tif", "fila", "columna", "ancho", "alto",
        "num_edificios", "prioridad", "anio_catastro",
    ]]


def preparar(salida: Path = INDICE_TESELAS, bbox=None) -> pd.DataFrame:
    """Guarda el índice y las capas de cobertura nominal de las ortofotos locales."""
    edificios = gpd.read_file(EDIFICIOS_CATASTRO, layer="edificios")
    region = union_all(gpd.read_file(MUNICIPIOS_CANTABRIA, layer="cantabria").geometry)
    if bbox is not None:
        region = region.intersection(box(*bbox))

    registros, hojas = [], []
    tifs = sorted(PNOA.glob("*.tif"))
    for numero, ruta in enumerate(tifs, 1):
        with rasterio.open(ruta) as tif:
            zona = region.intersection(box(*tif.bounds))
            if zona.is_empty:
                continue
            # Los solapes se resuelven al inferir: otra hoja puede cubrir bordes negros.
            tabla = obtener_teselas(tif, edificios, zona)
            registros.append(tabla)
            hojas.append({
                "tif": ruta.name,
                "teselas": len(tabla),
                "area_nominal_m2": zona.area,
                "geometry": zona,
            })
            print(f"[{numero}/{len(tifs)}] {ruta.name}: {len(tabla)} teselas")

    indice = pd.concat(registros, ignore_index=True)
    salida.parent.mkdir(parents=True, exist_ok=True)
    indice.to_csv(salida, index=False)
    cobertura = union_all([hoja["geometry"] for hoja in hojas])
    sin_cobertura = region.difference(cobertura)

    gpkg = salida.with_suffix(".gpkg")
    gpkg.unlink(missing_ok=True)
    gpd.GeoDataFrame(hojas, crs=25830).to_file(gpkg, layer="hojas", index=False)
    for nombre, geometria in (("cobertura_pnoa", cobertura), ("sin_ortofoto", sin_cobertura)):
        gpd.GeoDataFrame(geometry=[geometria], crs=25830).to_file(
            gpkg, layer=nombre, index=False
        )

    resumen = {
        "anio": 2023,
        "bbox": bbox,
        "teselas": len(indice),
        "con_edificaciones": int(indice.num_edificios.gt(0).sum()),
        "sin_edificaciones": int(indice.num_edificios.eq(0).sum()),
        "hojas_locales": len(tifs),
        "hojas_utilizadas": len(hojas),
        "sin_ortofoto_m2": sin_cobertura.area,
        "referencia_sha256": huella_archivo(EDIFICIOS_CATASTRO),
    }
    salida.with_suffix(".json").write_text(
        json.dumps(resumen, indent=2), encoding="utf-8"
    )
    return indice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=float, nargs=4)
    args = parser.parse_args()
    salida = ruta_proyecto("runs/prueba_2023/teselas.csv") if args.bbox else INDICE_TESELAS
    preparar(salida, args.bbox)


if __name__ == "__main__":
    main()

