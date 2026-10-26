"""Genera los mapas de la memoria desde el inventario mostrado en el panel."""

import argparse

import geopandas as gpd
import matplotlib
import pandas as pd

from paneles_solares.rutas import MUNICIPIOS_CANTABRIA, VISUALIZACION, ruta_proyecto

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

FIGURAS = ruta_proyecto("docs/figuras")


def guardar_mapa(
    municipios: gpd.GeoDataFrame,
    columna: str,
    titulo: str,
    leyenda: str,
    subtitulo: str,
    colores: str,
    gamma: float,
    archivo: str,
) -> None:
    """Dibuja los municipios con una escala que distingue los valores bajos.

    Args:
        municipios: Indicadores y geometrías municipales.
        columna: Campo numérico que se representa.
        titulo: Título visible del mapa.
        leyenda: Etiqueta y unidades de la escala de color.
        subtitulo: Texto breve sobre los datos y supuestos del mapa.
        colores: Nombre de la paleta de Matplotlib.
        gamma: Exponente de la escala de color para distinguir valores bajos.
        archivo: Nombre del PNG dentro de docs/figuras.
    """

    maximo = municipios[columna].max()
    if pd.isna(maximo):
        maximo = 1

    figura, eje = plt.subplots(figsize=(11, 6.2))
    municipios.plot(
        ax=eje,
        column=columna,
        cmap=colores,
        norm=PowerNorm(gamma=gamma, vmin=0, vmax=max(1, maximo)),
        edgecolor="#64748b",
        linewidth=0.32,
        legend=True,
        legend_kwds={"label": leyenda, "shrink": 0.74},
        missing_kwds={"color": "#dddddd"},
    )
    eje.set_axis_off()
    eje.set_title(titulo, fontsize=15, pad=15)
    figura.text(0.06, 0.06, subtitulo, fontsize=9, color="#475569")
    figura.tight_layout()
    figura.savefig(FIGURAS / archivo, dpi=200, facecolor="white")
    plt.close(figura)


def main() -> None:
    """Exporta la campaña elegida, o la más reciente si no se indica."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anio", type=int)
    args = parser.parse_args()

    resumen = pd.read_csv(
        VISUALIZACION / "resumen_municipal.csv", dtype={"codigo": str}
    )
    anio = args.anio if args.anio is not None else int(resumen.anio.max())
    tabla = resumen[
        (resumen.modo == "inventario")
        & (resumen.anio == anio)
        & (resumen.uso == "*")
        & (resumen.revision == "*")
        & (~resumen.codigo.isin(["*", "fuera"]))
    ]
    if tabla.empty:
        parser.error(f"No hay resultados de {anio}; genera antes el panel")

    FIGURAS.mkdir(parents=True, exist_ok=True)
    municipios = gpd.read_file(MUNICIPIOS_CANTABRIA, layer="municipios")
    municipios["codigo"] = municipios.codigo.astype(str)
    municipios = municipios.merge(tabla, on="codigo", how="left")
    nota = f"Inventario observado de {anio}; no censo regional. Gris: sin datos."
    guardar_mapa(
        municipios,
        "superficie_m2",
        "Superficie fotovoltaica detectada por municipio",
        "Superficie horizontal observada (m²)",
        nota,
        "YlGnBu",
        0.65,
        "mapa_detecciones.png",
    )

    guardar_mapa(
        municipios,
        "produccion_central_mwh",
        "Producción fotovoltaica anual estimada por municipio",
        "Producción central estimada (MWh/año)",
        nota + " Escenario central PVGIS.",
        "YlOrRd",
        0.60,
        "mapa_produccion.png",
    )


if __name__ == "__main__":
    main()
