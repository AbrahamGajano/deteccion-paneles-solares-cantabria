"""Actualiza los datos derivados después de preparar la referencia geográfica."""

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely import from_wkb, union_all
from shapely.geometry import box

from paneles_solares.datos.coleccion import MANIFEST, MANIFEST_LABELING
from paneles_solares.datos.seguimiento import BASE, AlmacenSeguimiento
from paneles_solares.geografia.teselas import preparar
from paneles_solares.rutas import (
    DATASETS,
    EDIFICIOS_CATASTRO,
    INDICADORES_SEGUIMIENTO,
    INDICADORES_UNET,
    MUNICIPIOS_CANTABRIA,
    PNOA,
    RAIZ,
    huella_archivo,
    retirar_panel,
)


class GeografiaSeguimiento:
    """Catastro describe la detección; su relación no confirma ni excluye paneles."""

    def __init__(self, proximidad: float | None = None):
        """Carga Catastro y los límites municipales para describir predicciones.

        Args:
            proximidad: Distancia descriptiva en metros, o None si no se ha estimado.
        """

        self.edificios = gpd.read_file(EDIFICIOS_CATASTRO, layer="edificios").set_index(
            "id_edificio", drop=False
        )
        self.edificios = self.edificios[~self.edificios.geometry.to_wkb().duplicated()]
        self.municipios = gpd.read_file(MUNICIPIOS_CANTABRIA, layer="municipios")
        self.proximidad = proximidad

    def describir(self, geometria) -> dict:
        """Mide intersección y distancia sin recortar ni desplazar la predicción.

        Args:
            geometria: Componente original en EPSG:25830.

        Returns:
            dict: Municipio, relación, fracción, distancia y asociaciones candidatas.
        """

        posiciones = self.edificios.sindex.query(geometria, predicate="intersects")
        coincidentes = self.edificios.iloc[posiciones]
        fraccion = (
            geometria.intersection(union_all(coincidentes.geometry)).area
            / geometria.area
            if len(posiciones)
            else 0
        )
        _, distancias = self.edificios.sindex.nearest(
            geometria, return_distance=True, return_all=True
        )
        distancia = float(distancias.min()) if len(distancias) else None
        edificios = [str(i) for i in coincidentes.index]
        cubiertas = []
        for idx, fila in coincidentes.iterrows():
            partes = (
                list(fila.geometry.geoms)
                if hasattr(fila.geometry, "geoms")
                else [fila.geometry]
            )
            for n, parte in enumerate(sorted(partes, key=lambda g: g.bounds), 1):
                if parte.intersection(geometria).area > 0:
                    cubiertas.append(f"catastro_{idx}_parte_{n}")
        if fraccion >= 0.8:
            relacion = "interseccion_clara" if len(edificios) == 1 else "ambigua"
        elif fraccion > 0:
            relacion = "parcial_desplazado" if len(edificios) == 1 else "ambigua"
        elif (
            self.proximidad is not None
            and distancia is not None
            and distancia <= self.proximidad
        ):
            relacion = "proximo"
        else:
            relacion = "exterior" if distancia is not None else "sin_asociacion"
        municipales = self.municipios.sindex.query(
            geometria.representative_point(), predicate="intersects"
        )
        municipio = self.municipios.iloc[municipales[0]] if len(municipales) else None
        return {
            "relacion": relacion,
            "edificios": edificios,
            "cubiertas": cubiertas,
            "fraccion": float(fraccion),
            "distancia": distancia,
            "municipio": municipio.municipio
            if municipio is not None
            else "Sin municipio resuelto",
            "codigo_municipal": str(municipio.codigo) if municipio is not None else "",
        }


def contar_edificios(manifest: Path, edificios: gpd.GeoDataFrame) -> None:
    """Actualiza el recuento de edificios sin cambiar las muestras ni sus etiquetas."""
    tabla = pd.read_csv(manifest, keep_default_na=False)
    validacion = "tif_referencia" in tabla
    campo_tif = "tif_referencia" if validacion else "tif"

    for nombre, grupo in tabla.groupby(campo_tif):
        geometrias = []
        with rasterio.open(PNOA / nombre) as tif:
            for fila in grupo.itertuples():
                if validacion:
                    ventana = Window(
                        fila.columna_referencia, fila.fila_referencia, 512, 512
                    )
                else:
                    ventana = Window(fila.columna, fila.fila, fila.ancho, fila.alto)
                geometrias.append(box(*tif.window_bounds(ventana)))

        parejas = edificios.sindex.query(geometrias, predicate="intersects")
        tabla.loc[grupo.index, "num_edificios"] = np.bincount(
            parejas[0], minlength=len(grupo)
        )
    tabla.to_csv(manifest, index=False)


def actualizar_seguimiento(base: Path = BASE) -> None:
    """Recalcula asociaciones conservando máscaras, agrupaciones y respuestas."""
    if not base.exists():
        return

    geografia = GeografiaSeguimiento()
    almacen = AlmacenSeguimiento(base)
    try:
        with almacen.db:
            for tabla in ("componentes", "emplazamientos"):
                campos = [
                    "relacion",
                    "edificios",
                    "cubiertas",
                    "municipio",
                    "codigo_municipal",
                ]
                if tabla == "componentes":
                    campos += ["fraccion", "distancia"]

                cambios = []
                for fila in almacen.filas(f"SELECT id, geometria FROM {tabla}"):
                    info = geografia.describir(from_wkb(fila["geometria"]))
                    valores = [
                        json.dumps(info[campo])
                        if isinstance(info[campo], list)
                        else info[campo]
                        for campo in campos
                    ]
                    cambios.append((*valores, fila["id"]))

                asignaciones = ", ".join(f"{campo}=?" for campo in campos)
                almacen.db.executemany(
                    f"UPDATE {tabla} SET {asignaciones} WHERE id=?", cambios
                )
            almacen.db.execute(
                "DELETE FROM metadatos WHERE clave LIKE 'cursor_region_%'"
            )

        almacen.poner_meta(
            "referencia_geografica",
            {
                "anio": 2023,
                "sha256": huella_archivo(EDIFICIOS_CATASTRO),
            },
        )
    finally:
        almacen.cerrar()


def main() -> None:
    """Regenera el índice, los manifiestos y el contexto del seguimiento."""
    if not MUNICIPIOS_CANTABRIA.exists():
        from paneles_solares.geografia.municipios import main as preparar_municipios

        preparar_municipios()

    edificios = gpd.read_file(EDIFICIOS_CATASTRO, layer="edificios")
    preparar()
    for manifest in (
        MANIFEST,
        MANIFEST_LABELING,
        DATASETS / "validacion_20cm/manifest.csv",
    ):
        if manifest.exists():
            contar_edificios(manifest, edificios)
    actualizar_seguimiento()

    # Estos indicadores requieren repetir la inferencia con la nueva referencia.
    carpetas = [
        INDICADORES_UNET,
        INDICADORES_SEGUIMIENTO,
        INDICADORES_UNET.with_name(".unet_actual_avance"),
        INDICADORES_UNET.with_name(".unet_actual_preparando"),
    ]

    for carpeta in carpetas:
        destino = carpeta.resolve()
        if not destino.is_relative_to((RAIZ / "runs").resolve()):
            raise ValueError(f"Resultado fuera de runs: {destino}")
        if destino.exists():
            shutil.rmtree(destino)
    retirar_panel()
    print(
        "Contexto geográfico actualizado; pendiente regenerar indicadores regionales."
    )


if __name__ == "__main__":
    main()
