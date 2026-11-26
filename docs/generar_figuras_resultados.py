"""Genera las figuras del análisis de la campaña de 2023."""

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from generar_mapas import FIGURAS, guardar_mapa, plt
from paneles_solares.rutas import CACHE_PVGIS_HORARIO, VISUALIZACION, ruta_proyecto
from paneles_solares.visualizacion.series_temporales import obtener_serie_horaria, parametros_horarios

SERIES = VISUALIZACION / "series_temporales" / "inventario_2023_meteo_2023"


def mapa_regional() -> None:
    """Representa la superficie detectada por municipio en la pasada regional."""

    archivo = ruta_proyecto("runs/indicadores/unet_actual/municipios.gpkg")
    municipios = gpd.read_file(archivo)
    municipios["superficie_ha"] = municipios.superficie_fv_detectada_total_m2 / 10_000
    guardar_mapa(
        municipios,
        "superficie_ha",
        "Detección fotovoltaica regional en Cantabria, 2023",
        "Superficie horizontal detectada (ha)",
        "Superficie absoluta. La extensión analizada se registra por municipio.",
        "YlGnBu",
        0.65,
        FIGURAS / "mapa_deteccion_regional_2023.png",
    )


def evolucion_por_campania(resumen: pd.DataFrame) -> None:
    """Representa el área disponible en el inventario de cada campaña."""

    tabla = resumen[
        (resumen.modo == "inventario") & (resumen.codigo == "*")
        & (resumen.uso == "*") & (resumen.revision == "*")
    ].set_index("anio").loc[[2014, 2017, 2020, 2023]]
    figura, eje = plt.subplots(figsize=(9.2, 3.3), layout="constrained")
    barras = eje.bar(tabla.index.astype(str), tabla.superficie_m2, color="#2468a0", width=0.58)
    barras[-1].set_facecolor("#f3d6c4")
    barras[-1].set_edgecolor("#a84d16")
    barras[-1].set_hatch("//")
    valores = [f"{valor:,.0f}".replace(",", ".") for valor in tabla.superficie_m2]
    eje.bar_label(barras, labels=valores, padding=3)
    eje.set(
        title="Inventario de seguimiento por campaña",
        ylabel="Superficie horizontal (m²)",
        ylim=(0, tabla.superficie_m2.max() * 1.17),
    )
    eje.ticklabel_format(axis="y", style="plain")
    eje.grid(axis="y", alpha=0.22)
    eje.set_axisbelow(True)
    eje.spines[["top", "right"]].set_visible(False)
    figura.savefig(FIGURAS / "superficie_por_campania.png", dpi=220)
    plt.close(figura)


def usos_cartograficos(resumen: pd.DataFrame) -> None:
    """Compara la superficie horizontal y la de módulos por uso cartográfico."""

    tabla = resumen[
        (resumen.modo == "inventario") & (resumen.anio == 2023)
        & (resumen.codigo == "*") & (resumen.revision == "*")
    ].set_index("uso")
    usos = {
        "Industrial o manufacturero o logístico-almacenaje": "Industrial y logística",
        "Residencial en vivienda no colectiva": "Vivienda no colectiva",
        "Sin dato cartográfico": "Sin uso cartográfico asignado",
        "Comercial": "Comercial",
    }
    seleccion = tabla.loc[list(usos)].rename(index=usos).iloc[::-1]
    figura, eje = plt.subplots(figsize=(9.2, 3.1), layout="constrained")
    posiciones = np.arange(len(seleccion))
    eje.barh(posiciones + 0.17, seleccion.superficie_m2 / 10_000, height=0.30,
             color="#abcde4", label="Horizontal")
    barras = eje.barh(posiciones - 0.17, seleccion.superficie_modulos_m2 / 10_000, height=0.30,
                      color="#19746e", label="Módulos, con inclinación")
    eje.bar_label(
        barras,
        labels=[f"{n:,} fichas".replace(",", ".") for n in seleccion.con_mascara],
        padding=4,
    )
    eje.set(
        title="Inventario de seguimiento de 2023",
        xlabel="Superficie (ha)",
        yticks=posiciones,
        yticklabels=seleccion.index,
        xlim=(0, 8.0),
    )
    eje.legend(loc="lower right", frameon=False)
    eje.grid(axis="x", alpha=0.22)
    eje.set_axisbelow(True)
    eje.spines[["top", "right"]].set_visible(False)
    figura.savefig(FIGURAS / "superficie_por_uso_2023.png", dpi=220)
    plt.close(figura)


def distribucion_lidar() -> None:
    """Representa las inclinaciones y los azimuts aceptados por polígono."""

    factores = pd.read_csv(VISUALIZACION / "factores_energia.csv")
    campania = factores[factores.anio == 2023]
    inclinaciones = campania.inclinacion_estimada.dropna()
    azimuts = campania.loc[campania.origen_orientacion == "estimada", "azimut_lidar_grados"]
    sectores = ((azimuts + 22.5) % 360 // 45).astype(int)
    cuentas = sectores.value_counts().reindex(range(8), fill_value=0)

    figura = plt.figure(figsize=(9, 3.6), layout="constrained")
    figura.suptitle("Inventario de seguimiento de 2023")
    rejilla = figura.add_gridspec(1, 2, width_ratios=[1.2, 1])
    inclinacion = figura.add_subplot(rejilla[0, 0])
    orientacion = figura.add_subplot(rejilla[0, 1], projection="polar")
    inclinacion.hist(inclinaciones, bins=range(0, 51, 5), color="#2468a0", edgecolor="white")
    inclinacion.axvline(inclinaciones.median(), ls="--", lw=1.5, color="#a84d16", label="Mediana")
    inclinacion.set(
        title=f"Inclinación · {len(inclinaciones):,} polígonos".replace(",", "."),
        xlabel="Grados desde la horizontal",
        ylabel="Polígonos",
        xticks=range(0, 51, 10),
        xlim=(0, 50),
    )
    inclinacion.legend(frameon=False)
    inclinacion.grid(axis="y", alpha=0.22)
    inclinacion.set_axisbelow(True)
    inclinacion.spines[["top", "right"]].set_visible(False)

    angulos = np.deg2rad(np.arange(0, 360, 45))
    orientacion.set_theta_zero_location("N")
    orientacion.set_theta_direction(-1)
    orientacion.bar(angulos, cuentas, width=np.deg2rad(40), color="#19746e", alpha=0.88)
    orientacion.set_xticks(angulos, ["N", "NE", "E", "SE", "S", "SO", "O", "NO"])
    orientacion.set_yticks([200, 400, 600, 800])
    orientacion.set_rlabel_position(15)
    orientacion.set_title(
        f"Orientación · {len(azimuts):,} polígonos".replace(",", "."), pad=22
    )
    figura.savefig(FIGURAS / "angulos_lidar_2023.png", dpi=220)
    plt.close(figura)


def produccion_mensual() -> None:
    """Suma la energía mensual de los tres escenarios con meteorología de 2023."""

    datos = pd.read_csv(SERIES / "mensual_instalaciones.csv")
    mensual = datos.groupby(["mes", "escenario"]).energia_kwh.sum().unstack() / 1e6
    figura, eje = plt.subplots(figsize=(8.8, 3.1), layout="constrained")
    eje.fill_between(mensual.index, mensual["min"], mensual["max"],
                     color="#abcde4", label="Sensibilidad · 8–12 % de pérdidas")
    eje.plot(mensual.index, mensual.central, "o-", color="#2468a0", label="Referencia · 10 %")
    eje.set(
        xticks=range(1, 13),
        xticklabels="Ene Feb Mar Abr May Jun Jul Ago Sep Oct Nov Dic".split(),
        ylabel="Energía estimada (GWh)",
        ylim=(0, mensual["max"].max() * 1.29),
    )
    eje.legend(ncol=3, loc="upper center", frameon=False)
    eje.grid(axis="y", alpha=0.22)
    eje.set_axisbelow(True)
    eje.spines[["top", "right"]].set_visible(False)
    figura.savefig(FIGURAS / "produccion_mensual_2023.png", dpi=220)
    plt.close(figura)


def curvas_santander() -> None:
    """Compara geometrías con el mismo recurso meteorológico y potencia nominal."""

    referencias = pd.read_csv(SERIES / "referencias.csv", dtype={"codigo": str})
    punto = referencias[referencias.codigo == "34063939075"].iloc[0]
    comparaciones = [
        ("Orientación con inclinación de 30°", [(30, -50, "Sureste"), (30, 50, "Suroeste")]),
        ("Inclinación con orientación sur", [(10, 0, "10° al sur"), (30, 0, "30° al sur")]),
    ]
    figura, ejes = plt.subplots(2, 2, figsize=(9.2, 5.3), sharey=True, layout="constrained")
    with requests.Session() as sesion:
        for columna, (titulo, casos) in enumerate(comparaciones):
            for (inclinacion, orientacion, etiqueta), (color, estilo) in zip(
                casos, [("#2468a0", "-"), ("#a84d16", "--")]
            ):
                fila = SimpleNamespace(
                    latitud=punto.latitud, longitud=punto.longitud,
                    inclinacion_pvgis_grados=inclinacion, orientacion_grados=orientacion,
                )
                _, serie = obtener_serie_horaria(sesion, parametros_horarios(fila, 2023), CACHE_PVGIS_HORARIO)
                serie = serie.tz_convert("Europe/Madrid")
                print(f"{etiqueta} | anual {serie.sum():.1f} kWh")
                for renglon, (mes, nombre) in enumerate([(6, "Junio"), (12, "Diciembre")]):
                    mensual = serie[serie.index.month == mes]
                    perfil = mensual.groupby(mensual.index.hour).mean()
                    eje = ejes[renglon, columna]
                    eje.plot(perfil.index, perfil, label=etiqueta, c=color, ls=estilo, lw=2)
                    eje.set_title(f"{nombre} · {titulo}", fontsize=10)
                    antes = mensual[mensual.index.hour < 14].sum() / mensual.sum()
                    print(
                        f"  {nombre}: {mensual.sum():.1f} kWh | pico {perfil.idxmax():02d}:00"
                        f" | antes 14 h {antes:.1%} | potencia a las 13 h {perfil[13]:.3f} kW/kWp"
                    )
    for eje in ejes.flat:
        eje.set(xlim=(5, 23), ylim=(0, 0.75), xticks=range(6, 23, 4))
        eje.legend(loc="upper right", frameon=False, fontsize=8)
        eje.grid(axis="y", alpha=0.22)
        eje.spines[["top", "right"]].set_visible(False)
    figura.supxlabel("Hora local")
    figura.supylabel("Potencia horaria media (kW por kWp)")
    figura.savefig(FIGURAS / "perfil_horario_santander_2023.png", dpi=220)
    plt.close(figura)


if __name__ == "__main__":
    mapa_regional()
    resumen = pd.read_csv(VISUALIZACION / "resumen_municipal.csv", dtype={"codigo": str})
    evolucion_por_campania(resumen)
    usos_cartograficos(resumen)
    distribucion_lidar()
    produccion_mensual()
    curvas_santander()
