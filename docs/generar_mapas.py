"""Genera los dos mapas estáticos de la memoria desde las salidas actuales."""

import geopandas as gpd
import matplotlib
import pandas as pd

from paneles_solares.rutas import INDICADORES_UNET, ruta_proyecto

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
    """Dibuja los municipios con una escala que distingue los valores bajos."""
    figura, eje = plt.subplots(figsize=(11, 6.2))
    municipios.plot(
        ax=eje,
        column=columna,
        cmap=colores,
        norm=PowerNorm(gamma=gamma, vmin=0, vmax=municipios[columna].max()),
        edgecolor="#64748b",
        linewidth=0.32,
        legend=True,
        legend_kwds={"label": leyenda, "shrink": 0.74},
    )
    eje.set_axis_off()
    eje.set_title(titulo, fontsize=15, pad=15)
    figura.text(0.06, 0.06, subtitulo, fontsize=9, color="#475569")
    figura.tight_layout()
    figura.savefig(FIGURAS / archivo, dpi=200, facecolor="white")
    plt.close(figura)


def main() -> None:
    """Lee los indicadores definitivos y sobrescribe los mapas de la memoria."""
    FIGURAS.mkdir(parents=True, exist_ok=True)
    municipios = gpd.read_file(INDICADORES_UNET / "municipios.gpkg")
    guardar_mapa(
        municipios,
        "edificios_con_deteccion_por_1000",
        "Edificios con detección fotovoltaica por municipio",
        "Edificios con detección por 1.000 analizados",
        "U-Net · PNOA 2023 · tasa por municipio",
        "YlGnBu",
        0.65,
        "mapa_detecciones.png",
    )

    produccion = pd.read_csv(
        INDICADORES_UNET / "produccion_municipal.csv", dtype={"codigo": str}
    )
    mapa_energia = municipios[["codigo", "municipio", "geometry"]].merge(
        produccion[["codigo", "produccion_anual_central_kwh"]],
        on="codigo",
        validate="one_to_one",
    )
    mapa_energia["produccion_mwh"] = (
        mapa_energia.produccion_anual_central_kwh / 1000
    )
    guardar_mapa(
        mapa_energia,
        "produccion_mwh",
        "Producción fotovoltaica anual estimada por municipio",
        "Producción central estimada (MWh/año)",
        "U-Net · PNOA 2023 · PVGIS 5.3 · escenario central",
        "YlOrRd",
        0.60,
        "mapa_produccion.png",
    )


if __name__ == "__main__":
    main()
