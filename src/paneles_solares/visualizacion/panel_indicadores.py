"""Panel municipal y resumen gráfico de los indicadores U-Net de Cantabria."""

import json

import folium
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from branca.element import Element
from folium.plugins import HeatMap

from paneles_solares.rutas import (
    INDICADORES_UNET,
    PRODUCCION_CONFIG,
    PRODUCCION_MUNICIPAL,
    version_indicadores,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA = INDICADORES_UNET
MUNICIPIOS = CARPETA / "municipios.gpkg"
EDIFICIOS = CARPETA / "edificios.gpkg"
SALIDA_HTML = CARPETA / "panel_indicadores.html"
SALIDA_PNG = CARPETA / "resumen_indicadores.png"

TAM_CELDA = 1000  # Metros; agrupa los edificios positivos para el mapa de calor.
SIMPLIFICACION_MAPA = 40  # Metros; afecta solo a la geometría dibujada en HTML.

INDICADORES = {
    "superficie_fv_detectada_total_m2": ("Superficie fotovoltaica detectada", "m²"),
    "superficie_fv_corregida_test_total_m2": (
        "Superficie corregida según el test",
        "m²",
    ),
    "edificios_con_deteccion": ("Edificios con detección", "edificios"),
    "porcentaje_edificios_con_deteccion": ("Edificios con detección", "%"),
    "edificios_con_deteccion_por_1000": (
        "Edificios con detección por cada 1.000",
        "edificios",
    ),
    "porcentaje_cubierta_ocupado": ("Cubierta ocupada por fotovoltaica", "%"),
    "superficie_fv_media_positivo_m2": (
        "Superficie detectada media por edificio positivo",
        "m²",
    ),
    "superficie_fv_mediana_positivo_m2": (
        "Superficie detectada mediana por edificio positivo",
        "m²",
    ),
}

INDICADORES_ENERGIA = {
    "potencia_central_mwp": ("Potencia pico central estimada", "MWp"),
    "produccion_anual_central_mwh": ("Producción anual central estimada", "MWh"),
}
INDICADOR_INICIAL_ENERGIA = "produccion_anual_central_mwh"

# Campo numérico, texto del GeoJSON, etiqueta visible, decimales y unidad.
TOOLTIPS = [
    ("edificios_analizados", "texto_analizados", "Edificios analizados", 0, ""),
    ("edificios_con_deteccion", "texto_detectados", "Edificios con detección", 0, ""),
    ("porcentaje_edificios_con_deteccion", "texto_porcentaje_edificios", "Edificios con detección (%)", 2, " %"),
    ("superficie_fv_detectada_total_m2", "texto_superficie_detectada", "Superficie detectada", 2, " m²"),
    ("superficie_fv_corregida_test_total_m2", "texto_superficie_corregida", "Superficie corregida (test)", 2, " m²"),
    ("edificios_con_deteccion_por_1000", "texto_por_1000", "Edificios con detección por 1.000", 2, ""),
    ("porcentaje_cubierta_ocupado", "texto_cubierta_ocupada", "Cubierta ocupada", 2, " %"),
    ("superficie_fv_media_positivo_m2", "texto_media", "Media por edificio positivo", 2, " m²"),
    ("superficie_fv_mediana_positivo_m2", "texto_mediana", "Mediana por edificio positivo", 2, " m²"),
]
TOOLTIPS_ENERGIA = [
    (f"potencia_{escenario}_mwp", f"texto_potencia_{escenario}", f"Potencia pico {escenario} estimada", 3, " MWp")
    for escenario in ("conservador", "central", "superior")
] + [
    (f"produccion_anual_{escenario}_mwh", f"texto_produccion_{escenario}", f"Producción anual {escenario} estimada", 2, " MWh/año")
    for escenario in ("conservador", "central", "superior")
] + [
    ("rendimiento_anual_kwh_kwp", "texto_rendimiento_anual", "Rendimiento anual PVGIS", 2, " kWh/kWp"),
]

COLORES = ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#cc4c02"]
COLOR_SIN_EDIFICIOS = "#9ca3af"
COLOR_SIN_DETECCIONES = "#d9efe2"


def cargar_datos() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Lee los resultados geográficos de la U-Net sin usar CSV por tesela."""
    for ruta in (MUNICIPIOS, EDIFICIOS):
        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta {ruta}. Ejecuta primero "
                "python -m paneles_solares.modelos.indicadores_unet"
            )
    municipios = gpd.read_file(MUNICIPIOS)
    positivos = gpd.read_file(
        EDIFICIOS,
        where="presencia_fotovoltaica = 1",
        columns=["superficie_fv_detectada_m2", "geometry"],
    )
    return municipios, positivos


def integrar_produccion(municipios: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict | None]:
    """Incorpora solo resúmenes energéticos compatibles con los GeoPackage actuales."""
    if not all(ruta.is_file() for ruta in (PRODUCCION_CONFIG, PRODUCCION_MUNICIPAL)):
        return municipios, None
    config = json.loads(PRODUCCION_CONFIG.read_text(encoding="utf-8"))
    if config["indicadores_entrada"] != version_indicadores():
        return municipios, None
    energia = pd.read_csv(PRODUCCION_MUNICIPAL, dtype={"codigo": str})
    if energia.codigo.duplicated().any() or set(energia.codigo) != set(municipios.codigo.astype(str)):
        raise ValueError("El resumen energético no contiene exactamente los municipios del GeoPackage")
    municipios = municipios.merge(
        energia.drop(columns=["municipio", "superficie_fv_detectada_m2", "superficie_fv_corregida_test_m2"]),
        on="codigo", validate="one_to_one",
    )
    for escenario in ("conservador", "central", "superior"):
        municipios[f"potencia_{escenario}_mwp"] = municipios[f"potencia_{escenario}_kwp"] / 1000
        municipios[f"produccion_anual_{escenario}_mwh"] = municipios[f"produccion_anual_{escenario}_kwh"] / 1000
    return municipios, config


def formatear_numero(valor: float, decimales: int = 2) -> str:
    """Representa un número con miles y decimales en formato español."""
    return (
        f"{valor:,.{decimales}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    )


def preparar_tooltips(municipios: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Añade textos legibles sin modificar los indicadores numéricos."""
    datos = municipios.copy()
    campos = TOOLTIPS + (TOOLTIPS_ENERGIA if "potencia_central_kwp" in datos else [])
    for columna, texto, _, decimales, unidad in campos:
        datos[texto] = [
            (
                "Sin datos"
                if pd.isna(valor)
                else formatear_numero(valor, decimales) + unidad
            )
            for valor in datos[columna]
        ]
    sin_detecciones = datos.edificios_con_deteccion.eq(0)
    datos.loc[sin_detecciones, ["texto_media", "texto_mediana"]] = "Sin detecciones"
    if "potencia_central_kwp" in datos:
        datos.loc[~datos.consultado_pvgis, "texto_rendimiento_anual"] = "Sin consulta PVGIS"
    return datos


def crear_escalas(
    municipios: gpd.GeoDataFrame, indicadores: dict[str, tuple[str, str]]
) -> dict[str, dict]:
    """Calcula cortes por cuantiles entre municipios con detección."""
    escalas = {}
    positivos = municipios.loc[municipios.edificios_con_deteccion.gt(0)]
    for columna, (titulo, unidad) in indicadores.items():
        valores = positivos[columna].dropna().to_numpy()
        cortes = (
            np.unique(np.quantile(valores, [0.2, 0.4, 0.6, 0.8])).tolist()
            if len(valores)
            else []
        )
        escalas[columna] = {"titulo": titulo, "unidad": unidad, "cortes": cortes}
    return escalas


def color_municipio(propiedades: dict, columna: str, cortes: list[float]) -> str:
    """Distingue falta de análisis, ausencia de detección y valor positivo."""
    if propiedades["edificios_analizados"] == 0:
        return COLOR_SIN_EDIFICIOS
    if propiedades["edificios_con_deteccion"] == 0:
        return COLOR_SIN_DETECCIONES
    return COLORES[min(np.searchsorted(cortes, propiedades[columna], side="left"), 4)]


def texto_resumen(municipios: gpd.GeoDataFrame, con_energia: bool) -> str:
    """Crea las cifras generales a partir de los indicadores municipales."""
    analizados = int(municipios.edificios_analizados.sum())
    detectados = int(municipios.edificios_con_deteccion.sum())
    superficie = municipios.superficie_fv_detectada_total_m2.sum()
    corregida = municipios.superficie_fv_corregida_test_total_m2.sum()
    cubierta = municipios.superficie_cubiertas_analizadas_m2.sum()
    porcentaje = 100 * detectados / analizados if analizados else 0
    ocupacion = 100 * superficie / cubierta if cubierta else 0
    cifras = [
        ("Edificios analizados", formatear_numero(analizados, 0)),
        ("Edificios con detección", formatear_numero(detectados, 0)),
        ("Edificios con detección", f"{formatear_numero(porcentaje)} %"),
        ("Superficie detectada", f"{formatear_numero(superficie)} m²"),
        ("Superficie corregida · test", f"{formatear_numero(corregida)} m²"),
        ("Cubiertas analizadas", f"{formatear_numero(cubierta)} m²"),
        ("Cubierta ocupada", f"{formatear_numero(ocupacion)} %"),
    ]
    energia_html = ""
    if con_energia:
        cifras_energia = []
        for escenario in ("conservador", "central", "superior"):
            cifras_energia.extend([
                (f"Potencia pico {escenario} estimada", f"{formatear_numero(municipios[f'potencia_{escenario}_mwp'].sum())} MWp"),
                (f"Producción anual {escenario} estimada", f"{formatear_numero(municipios[f'produccion_anual_{escenario}_mwh'].sum() / 1000)} GWh"),
            ])
        tarjetas_energia = "".join(
            f'<div class="cifra"><strong>{valor}</strong><span>{nombre}</span></div>'
            for nombre, valor in cifras_energia
        )
        energia_html = (
            '<h3>Potencia pico y producción estimadas</h3>'
            f'<div class="cifras">{tarjetas_energia}</div>'
        )
    tarjetas = "".join(
        f'<div class="cifra"><strong>{valor}</strong><span>{nombre}</span></div>'
        for nombre, valor in cifras
    )
    return (
        '<div id="resumen-panel"><h2>Cantabria · U-Net</h2>'
        f'{energia_html}<h3>Detecciones y superficies</h3>'
        f'<div class="cifras">{tarjetas}</div></div>'
    )


def puntos_calor(positivos: gpd.GeoDataFrame) -> list[list[float]]:
    """Agrupa puntos interiores de edificios positivos en celdas de 1 km."""
    if positivos.empty:
        return []
    puntos = positivos.representative_point()
    celdas = pd.DataFrame(
        {
            "x": puntos.x,
            "y": puntos.y,
            "celda_x": (puntos.x // TAM_CELDA).astype(int),
            "celda_y": (puntos.y // TAM_CELDA).astype(int),
        }
    )
    agrupadas = celdas.groupby(["celda_x", "celda_y"], as_index=False).agg(
        x=("x", "mean"), y=("y", "mean"), edificios=("x", "size")
    )
    ubicaciones = gpd.GeoSeries(
        gpd.points_from_xy(agrupadas.x, agrupadas.y), crs=positivos.crs
    ).to_crs("EPSG:4326")
    return [
        [punto.y, punto.x, int(peso)]
        for punto, peso in zip(ubicaciones, agrupadas.edificios)
    ]


def controles_html(
    municipios: gpd.GeoDataFrame, config_energia: dict | None,
) -> str:
    """Prepara el selector, la leyenda y el resumen general."""
    con_energia = config_energia is not None
    opciones = "".join(
        f'<option value="{columna}">{titulo} ({unidad})</option>'
        for columna, (titulo, unidad) in INDICADORES.items()
    )
    if con_energia:
        opciones = (
            '<optgroup label="Potencia y producción estimadas">'
            + "".join(
                f'<option value="{columna}"'
                f'{" selected" if columna == INDICADOR_INICIAL_ENERGIA else ""}>'
                f'{titulo} ({unidad})</option>'
                for columna, (titulo, unidad) in INDICADORES_ENERGIA.items()
            )
            + '</optgroup><optgroup label="Detecciones y superficies">'
            + opciones + '</optgroup>'
        )
    nota_energia = (
        "<p class='nota-panel'>La potencia pico y la producción son estimaciones, "
        "no potencia instalada conocida ni energía medida. La superficie corregida "
        "utiliza el sesgo global del test (−7,54 %). PVGIS usa inclinación de "
        f"{config_energia['inclinacion_grados']}°, orientación "
        f"{config_energia['orientacion_pvgis_grados']}° (0° = sur) y pérdidas del "
        f"{config_energia['perdidas_pct']} % como hipótesis comunes; aún no se conoce "
        "la inclinación ni orientación real de cada cubierta. El perfil horario es "
        "representativo, no una medición real.</p>"
        if con_energia else ""
    )
    return f"""
    <style>
      .panel-ui {{position:fixed;z-index:1000;background:#fff;color:#1f2937;
        border:1px solid #d1d5db;border-radius:9px;box-shadow:0 2px 12px #0002;
        font:13px Arial,sans-serif;padding:14px;box-sizing:border-box}}
      #resumen-panel {{position:fixed;top:15px;left:62px;width:355px;z-index:1000;
        background:#fff;border:1px solid #d1d5db;border-radius:9px;
        box-shadow:0 2px 12px #0002;padding:14px;font:13px Arial,sans-serif;
        max-height:58vh;overflow:auto}}
      #resumen-panel h2 {{margin:0 0 10px;font-size:17px}}
      #resumen-panel h3 {{margin:12px 0 8px;font-size:13px}}
      .cifras {{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
      .cifra {{background:#f8fafc;border-radius:5px;padding:7px}}
      .cifra strong {{display:block;font-size:15px}}
      .cifra span {{font-size:11px;color:#475569}}
      #selector-panel {{top:15px;right:15px;width:340px}}
      #selector-panel label {{display:block;font-weight:bold;font-size:15px;margin-bottom:8px}}
      #indicador-select {{width:100%;padding:7px;border:1px solid #9ca3af;border-radius:5px}}
      #titulo-indicador {{margin:10px 0 0;font-weight:bold}}
      #leyenda-panel {{bottom:22px;right:15px;width:340px;max-height:43vh;overflow:auto}}
      #leyenda-panel h3 {{margin:0 0 7px;font-size:14px}}
      .leyenda-fila {{display:flex;align-items:center;gap:8px;margin:4px 0}}
      .leyenda-color {{width:17px;height:13px;border:1px solid #6665;flex:none}}
      .nota-panel {{font-size:11px;line-height:1.35;color:#475569;margin:8px 0 0}}
      @media(max-width:760px) {{ #resumen-panel {{width:250px;max-height:35vh}}
        #selector-panel,#leyenda-panel {{width:245px}}}}
    </style>
    {texto_resumen(municipios, con_energia)}
    <div class="panel-ui" id="selector-panel">
      <label for="indicador-select">Indicador municipal</label>
      <select id="indicador-select">{opciones}</select>
      <p id="titulo-indicador"></p>
      {"<p class='nota-panel'>La potencia pico y la producción son estimaciones; consulta las hipótesis en la leyenda.</p>" if con_energia else "<p class='nota-panel'>Sin producción compatible: ejecuta estimar_energia.py y vuelve a generar este panel.</p>"}
    </div>
    <div class="panel-ui" id="leyenda-panel">
      <h3>Leyenda</h3><div id="leyenda-filas"></div>
      <p class="nota-panel">Cortes por cuantiles de municipios con detección.
      Los valores reales aparecen al pasar sobre cada municipio.</p>
      <p class="nota-panel">Superficie planimétrica. La superficie corregida es una
      calibración experimental del sesgo global del test (−7,54 %).
      Límites municipales: IGN/CNIG, CC BY 4.0.</p>
      {nota_energia}
    </div>
    """


def selector_javascript(capa: folium.GeoJson, escalas: dict[str, dict]) -> str:
    """Cambia el estilo de un único GeoJSON y reconstruye su leyenda."""
    return f"""
    document.addEventListener('DOMContentLoaded', function () {{
      const capa = {capa.get_name()};
      const escalas = {json.dumps(escalas, ensure_ascii=False)};
      const colores = {json.dumps(COLORES)};
      const gris = '{COLOR_SIN_EDIFICIOS}';
      const sinDetecciones = '{COLOR_SIN_DETECCIONES}';
      const selector = document.getElementById('indicador-select');
      const formato = new Intl.NumberFormat('es-ES', {{maximumFractionDigits:2}});
      function fila(color, texto) {{
        return '<div class="leyenda-fila"><span class="leyenda-color" style="background:' +
          color + '"></span><span>' + texto + '</span></div>';
      }}
      function actualizar() {{
        const clave = selector.value;
        const escala = escalas[clave];
        capa.setStyle(function (feature) {{
          const datos = feature.properties;
          let color = gris;
          if (datos.edificios_analizados > 0) {{
            color = sinDetecciones;
            if (datos.edificios_con_deteccion > 0) {{
              const indice = escala.cortes.findIndex(corte => datos[clave] <= corte);
              color = colores[indice < 0 ? escala.cortes.length : indice];
            }}
          }}
          return {{fillColor:color,fillOpacity:0.75,color:'#64748b',weight:1}};
        }});
        document.getElementById('titulo-indicador').textContent =
          escala.titulo + ' · ' + escala.unidad;
        let filas = fila(gris, 'Sin edificios analizados');
        filas += fila(sinDetecciones, 'Analizados sin detecciones');
        escala.cortes.forEach(function (corte, indice) {{
          const inferior = indice ? '> ' + formato.format(escala.cortes[indice-1]) + ' y ' : '';
          filas += fila(colores[indice], inferior + '≤ ' + formato.format(corte) + ' ' + escala.unidad);
        }});
        if (escala.cortes.length) {{
          filas += fila(colores[escala.cortes.length], '> ' +
            formato.format(escala.cortes[escala.cortes.length-1]) + ' ' + escala.unidad);
        }}
        document.getElementById('leyenda-filas').innerHTML = filas;
      }}
      selector.addEventListener('change', actualizar);
      actualizar();
    }});
    """


def crear_panel(
    municipios: gpd.GeoDataFrame, positivos: gpd.GeoDataFrame,
    config_energia: dict | None = None,
) -> folium.Map:
    """Crea el mapa municipal único y la capa opcional de calor."""
    con_energia = config_energia is not None
    indicadores = INDICADORES | (INDICADORES_ENERGIA if con_energia else {})
    escalas = crear_escalas(municipios, indicadores)
    inicial = INDICADOR_INICIAL_ENERGIA if con_energia else next(iter(indicadores))
    limites = municipios.to_crs("EPSG:4326").total_bounds
    mapa = folium.Map(tiles=None, zoom_control=True)
    mapa.fit_bounds([[limites[1], limites[0]], [limites[3], limites[2]]])

    poligonos = preparar_tooltips(municipios)
    poligonos["geometry"] = poligonos.geometry.simplify(
        SIMPLIFICACION_MAPA, preserve_topology=True
    )
    tooltips = TOOLTIPS + (TOOLTIPS_ENERGIA if con_energia else [])
    campos_tooltip = ["municipio"] + [texto for _, texto, _, _, _ in tooltips]
    aliases = ["Municipio"] + [titulo for _, _, titulo, _, _ in tooltips]
    capa = folium.GeoJson(
        poligonos.to_crs("EPSG:4326").to_json(),
        name="Municipios",
        control=False,
        style_function=lambda feature: {
            "fillColor": color_municipio(
                feature["properties"], inicial, escalas[inicial]["cortes"]
            ),
            "fillOpacity": 0.75,
            "color": "#64748b",
            "weight": 1,
        },
        highlight_function=lambda _: {"weight": 2, "color": "#111827"},
        tooltip=folium.GeoJsonTooltip(
            fields=campos_tooltip, aliases=aliases, labels=True
        ),
    ).add_to(mapa)

    calor = puntos_calor(positivos)
    if calor:
        capa_calor = folium.FeatureGroup(
            name=f"Calor de edificios con detección ({len(positivos):,})", show=False
        ).add_to(mapa)
        HeatMap(calor, radius=18, blur=15, min_opacity=0.25).add_to(capa_calor)
        folium.LayerControl(position="bottomleft", collapsed=False).add_to(mapa)

    mapa.get_root().html.add_child(Element(controles_html(municipios, config_energia)))
    mapa.get_root().script.add_child(Element(selector_javascript(capa, escalas)))
    return mapa


def crear_resumen_png(
    municipios: gpd.GeoDataFrame, positivos: gpd.GeoDataFrame
) -> None:
    """Dibuja cuatro gráficos para la memoria técnica."""
    figura, ejes = plt.subplots(2, 2, figsize=(16, 12))
    figura.subplots_adjust(
        left=0.16, right=0.98, top=0.92, bottom=0.09, hspace=0.22, wspace=0.36
    )
    top_superficie = municipios.nlargest(
        15, "superficie_fv_corregida_test_total_m2"
    ).iloc[::-1]
    ejes[0, 0].barh(
        top_superficie.municipio,
        top_superficie.superficie_fv_corregida_test_total_m2,
        color="#cc4c02",
    )
    ejes[0, 0].set(
        title="15 municipios con más superficie corregida",
        xlabel="Superficie corregida según el test (m²)",
    )

    top_tasa = municipios.nlargest(15, "edificios_con_deteccion_por_1000").iloc[::-1]
    ejes[0, 1].barh(
        top_tasa.municipio, top_tasa.edificios_con_deteccion_por_1000, color="#20705c"
    )
    ejes[0, 1].set(
        title="15 municipios con más edificios con detección por cada 1.000",
        xlabel="Edificios con detección por cada 1.000 analizados",
    )

    superficies = positivos.superficie_fv_detectada_m2.loc[lambda valores: valores > 0]
    if not superficies.empty:
        minimo, maximo = superficies.min(), superficies.max()
        limites = np.geomspace(minimo, maximo, 31) if maximo > minimo else 10
        ejes[1, 0].hist(superficies, bins=limites, color="#2e78a6", edgecolor="white")
        if maximo > minimo:
            ejes[1, 0].set_xscale("log")
    ejes[1, 0].set(
        title="Superficie detectada por edificio positivo",
        xlabel="Superficie sin corregir (m²; escala logarítmica)",
        ylabel="Edificios",
    )

    ejes[1, 1].scatter(
        municipios.edificios_con_deteccion,
        municipios.superficie_fv_corregida_test_total_m2,
        s=34,
        alpha=0.7,
        color="#8c4b9f",
    )
    ejes[1, 1].set(
        title="Edificios con detección y superficie corregida",
        xlabel="Edificios con detección",
        ylabel="Superficie corregida según el test (m²)",
    )
    for eje in ejes.flat:
        eje.grid(axis="x", alpha=0.2)
        eje.set_axisbelow(True)
        eje.tick_params(labelsize=9)
    figura.suptitle(
        "Indicadores fotovoltaicos de Cantabria · U-Net sobre PNOA 2023", fontsize=17
    )
    figura.text(
        0.5,
        0.025,
        "Superficies planimétricas. La corrección es una calibración experimental del sesgo global del test (−7,54 %).",
        ha="center",
        fontsize=9,
    )
    figura.savefig(SALIDA_PNG, dpi=180, facecolor="white")
    plt.close(figura)


def main() -> None:
    """Sobrescribe el panel HTML y el resumen PNG a partir de los GeoPackage."""
    municipios, positivos = cargar_datos()
    municipios, config_energia = integrar_produccion(municipios)
    crear_panel(municipios, positivos, config_energia).save(str(SALIDA_HTML))
    crear_resumen_png(municipios, positivos)
    print(f"Panel municipal: {SALIDA_HTML}")
    print(f"Resumen gráfico: {SALIDA_PNG}")


if __name__ == "__main__":
    main()
