"""Visor Dash para comprobar resultados de planos."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
from dash import Dash, Input, Output, State, ctx, dcc, html
from shapely import intersects_xy, wkt

from paneles_solares.lidar.estimar_planos import agregar_plano
from paneles_solares.lidar.preparar_revision import (
    PREPARACION, RECORTES, ORTOFOTOS, inventariar_casos,
    indexar_ortofotos, extraer_imagen, leer_nubes,
)
from paneles_solares.lidar.visor import figura_ortofoto, leer_casos
from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/lidar/prueba_pueblos_20261105/revision")
RESULTADOS = ruta_proyecto("runs/lidar/cantabria_20261112_v2")


def leer_resultado(id_panel: str, resultados: Path) -> dict | None:
    """Lee id_panel/resumen.json; devuelve None si todavía no existe."""

    ruta = resultados / id_panel / "resumen.json"

    if not ruta.exists():
        return None

    return json.loads(ruta.read_text(encoding="utf-8"))


def figura_ortofoto_regional(caso: pd.Series, ortofotos) -> tuple:
    """Extrae la ortofoto del contexto y devuelve su figura y límites reales.

    Params:
        caso: Detección del inventario con su ventana geográfica.
        ortofotos: Índice de los TIFF locales.

    Returns:
        tuple: Figura y límites xmin, ymin, xmax, ymax. Sin cobertura local,
            devuelve una figura informativa y los límites originales.
    """

    limites = tuple(caso[campo] for campo in ["xmin", "ymin", "xmax", "ymax"])
    with TemporaryDirectory(prefix="visor_lidar_") as temporal:
        carpeta = Path(temporal)
        ruta = carpeta / "ortofoto.png"
        try:
            limites = extraer_imagen(caso, ortofotos, ruta, cobertura_completa=False)
        except (ValueError, OSError, rasterio.errors.RasterioError) as error:
            figura = go.Figure()
            figura.add_annotation(text=f"Sin ortofoto local: {error}",
                                  x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            return figura, limites

        imagen = caso.copy()
        imagen["imagen"] = "ortofoto.png"
        for campo, valor in zip(["xmin", "ymin", "xmax", "ymax"], limites):
            imagen[campo] = valor

        return figura_ortofoto(imagen, carpeta), limites


def figura_nube(
    caso: pd.Series,
    carpeta: Path,
    resultado: dict | None,
    resultados: Path = RESULTADOS,
) -> go.Figure:
    """Dibuja la nube 3D, la detección y los planos guardados del ajuste.

    Params:
        caso: Fila con laz o rutas_laz, poligono y límites geográficos del recorte.
        carpeta: Directorio de la muestra; la ruta LAZ es relativa a él.
        resultado: Datos del ajuste asociado, o None si no está calculado.
        resultados: Directorio que contiene los archivos cubierta.csv por caso.

    Returns:
        go.Figure: Nube en coordenadas locales con los puntos interiores en rojo.
            El contorno se proyecta en la base de la vista, sin representar la
            altura del panel. Los planos interior y exterior aparecen en naranja
            y azul. El encuadre vertical utiliza los percentiles 1 y 99.
    """

    fig = go.Figure()

    if "rutas_laz" in caso.index:
        nube = leer_nubes(caso.rutas_laz, caso.crs_lidar)
    else:
        nube = laspy.read(carpeta / caso.laz)

    if not len(nube.points):
        return fig

    x = np.asarray(nube.x)
    y = np.asarray(nube.y)
    z = np.asarray(nube.z)

    geom = wkt.loads(caso.poligono)

    dentro = intersects_xy(geom, x, y)

    inferior, superior = np.percentile(z, [1, 99])
    margen = max(1.0, (superior - inferior) * 0.1)
    origen_z = inferior - margen
    altura = superior + margen - origen_z
    origen = np.array([caso.xmin, caso.ymin, origen_z])
    puntos = np.column_stack([x, y, z])

    fig.add_trace(
        go.Scatter3d(
            x=x[~dentro] - caso.xmin,
            y=y[~dentro] - caso.ymin,
            z=z[~dentro] - origen_z,
            mode="markers",
            name="Contexto",
            marker={
                "size": 2,
                "color": z[~dentro],
                "colorscale": "Viridis",
                "cmin": inferior,
                "cmax": max(superior, inferior + 0.1),
                "showscale": False,
            },
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x[dentro] - caso.xmin,
            y=y[dentro] - caso.ymin,
            z=z[dentro] - origen_z,
            mode="markers",
            name="Interior de la detección",
            marker={"size": 4, "color": "red"},
        )
    )

    poligonos = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]

    for poligono in poligonos:
        for anillo in [poligono.exterior, *poligono.interiors]:
            borde_x, borde_y = anillo.xy
            fig.add_trace(
                go.Scatter3d(
                    x=np.asarray(borde_x) - caso.xmin,
                    y=np.asarray(borde_y) - caso.ymin,
                    z=np.zeros(len(borde_x)),
                    mode="lines",
                    line={"color": "red", "dash": "dash"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if resultado and resultado.get("normal") is not None:
        centro = np.asarray(resultado["centro"])
        normal = np.asarray(resultado["normal"])
        distancias = np.abs((puntos - centro) @ normal)
        compatibles = dentro & (distancias <= resultado["umbral_m"])

        if compatibles.sum() >= 3:
            agregar_plano(
                fig,
                puntos[compatibles] - origen,
                centro - origen,
                normal,
                "#ffad33",
                "Plano interior candidato",
            )

    cubierta = resultado.get("cubierta") if resultado else None
    ruta_cubierta = resultados / caso.id_panel / "cubierta.csv"

    if cubierta and ruta_cubierta.exists():
        puntos_cubierta = pd.read_csv(ruta_cubierta)[["x", "y", "z"]].to_numpy()

        if len(puntos_cubierta) >= 3:
            locales = puntos_cubierta - origen
            fig.add_trace(
                go.Scatter3d(
                    x=locales[:, 0],
                    y=locales[:, 1],
                    z=locales[:, 2],
                    mode="markers",
                    name="Fragmento exterior asociado",
                    marker={"size": 3, "color": "#1645ab", "symbol": "diamond"},
                )
            )
            agregar_plano(
                fig,
                locales,
                np.asarray(cubierta["centro"]) - origen,
                np.asarray(cubierta["normal"]),
                "#168be0",
                "Plano exterior",
            )

    dimensiones = np.array([caso.xmax - caso.xmin, caso.ymax - caso.ymin, altura])
    proporciones = dimensiones / dimensiones.max()
    fig.update_layout(
        scene={
            "xaxis_title": "Este local (m)",
            "yaxis_title": "Norte local (m)",
            "zaxis": {"title": "Altura relativa (m)", "range": [0, altura]},
            "aspectmode": "manual",
            "aspectratio": dict(zip(["x", "y", "z"], proporciones)),
        },
        uirevision=caso.id_panel,
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 0, "r": 0, "t": 15, "b": 0},
    )

    return fig


def formatear_numero(valor: float | None, unidad: str = "") -> str:
    """Formatea con dos decimales y unidad; solo None se muestra como «Sin dato»."""

    return "Sin dato" if valor is None else f"{valor:.2f}{unidad}"


def mostrar_datos(caso: pd.Series, resultado: dict | None) -> list:
    """Prepara los textos de la tercera columna usando componentes html.

    Params:
        caso: Fila del panel actual, para mostrar su identificador.
        resultado: JSON del ajuste, o None si está sin procesar.

    Returns:
        list: Componentes como html.P con inclinación candidata, estimada,
            utilizada, origen, motivo y porcentaje de inliers. Los valores
            desconocidos se representan mediante texto.
    """

    datos = [html.P(caso.id_panel, style={"overflowWrap": "anywhere"})]

    if resultado is None:
        return datos + [html.P("Este caso todavía no tiene resultados calculados.")]

    medidas = [
        (
            "Inclinación candidata",
            formatear_numero(resultado.get("inclinacion_candidata"), "°"),
        ),
        (
            "Inclinación estimada",
            formatear_numero(resultado.get("inclinacion_estimada"), "°"),
        ),
        (
            "Inclinación utilizada",
            formatear_numero(resultado.get("inclinacion_utilizada"), "°"),
        ),
        ("Origen", resultado["origen"]),
        ("Motivo", resultado["motivo"].replace("_", " ")),
        (
            "Inliers interiores",
            formatear_numero(resultado.get("porcentaje_compatible"), " %"),
        ),
        (
            "RMSE interior",
            formatear_numero(resultado.get("rmse_perpendicular_m"), " m"),
        ),
    ]
    datos.extend(
        html.P([html.Strong(f"{nombre}: "), valor]) for nombre, valor in medidas
    )

    cubierta = resultado.get("cubierta")
    if cubierta:
        datos.extend(
            [
                html.H4("Comparación con el exterior"),
                html.P(
                    [
                        html.Strong("Ángulo entre planos: "),
                        formatear_numero(cubierta["angulo_entre_planos_grados"], "°"),
                    ]
                ),
                html.P(
                    [
                        html.Strong("Separación p90: "),
                        formatear_numero(cubierta["separacion_p90_m"], " m"),
                    ]
                ),
            ]
        )
    else:
        datos.append(html.P("Sin plano exterior asociado."))

    return datos


def crear_layout() -> html.Div:
    """Define los componentes de la página y su distribución en tres columnas.

    Returns:
        html.Div: Contenedor con dcc.Graph(id="ortofoto"), dcc.Graph(id="nube")
            y html.Div(id="datos"), botones con id="atras" e id="siguiente"
            y dcc.Store(id="indice", data=0) con la posición actual.
    """

    return html.Div(
        [
            html.H2("Revisión de resultados LiDAR"),
            dcc.Store(id="indice", data=0),
            html.Div(
                [
                    dcc.Dropdown(
                        id="filtro",
                        options=[
                            {"label": "Todos los resultados", "value": "todos"},
                            {"label": "Inclinación estimada", "value": "estimada"},
                            {"label": "Valor predeterminado", "value": "predeterminada"},
                        ],
                        value="todos", clearable=False,
                        style={"width": "250px"},
                    ),
                    html.Button("Atrás", id="atras", n_clicks=0),
                    html.Button("Siguiente", id="siguiente", n_clicks=0),
                    html.Button("Aleatorio", id="aleatorio", n_clicks=0),
                ],
                style={"display": "flex", "gap": "10px", "marginBottom": "15px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Ortofoto"),
                            dcc.Graph(id="ortofoto", style={"height": "72vh"}),
                        ],
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.H3("Nube y planos"),
                            dcc.Graph(id="nube", style={"height": "72vh"}),
                        ],
                        style={"minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.H3("Resultado"),
                            html.Div(id="datos"),
                            html.Hr(),
                            html.P(
                                "Rojo: puntos dentro del polígono detectado. "
                                "El contorno discontinuo muestra su proyección XY."
                            ),
                            html.P(
                                "Los planos dibujados son candidatos geométricos. "
                                "Una inclinación predeterminada es un valor supuesto."
                            ),
                            html.P("Los huecos de cobertura de la ortofoto se muestran en negro."),
                        ]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr 280px",
                    "gap": "18px",
                },
            ),
        ],
        style={"fontFamily": "Arial, sans-serif", "margin": "20px"},
    )


def registrar_callbacks(
    app: Dash, casos: pd.DataFrame, carpeta: Path, resultados: Path, ortofotos=None,
) -> None:
    """Conecta los botones con las figuras y los datos de la página.

    Params:
        app: Aplicación Dash cuyo layout ya contiene los identificadores indicados.
        casos: Tabla de la muestra, accesible desde la función interior.
        carpeta: Directorio de imágenes y nubes.
        resultados: Directorio de los ajustes guardados.
        ortofotos: Índice de TIFF para el inventario regional; None en una muestra.
    """

    @app.callback(
        Output("indice", "data"),
        Output("ortofoto", "figure"),
        Output("nube", "figure"),
        Output("datos", "children"),
        Input("atras", "n_clicks"),
        Input("siguiente", "n_clicks"),
        Input("aleatorio", "n_clicks"),
        Input("filtro", "value"),
        State("indice", "data"),
    )
    def actualizar_vista(
        atras: int | None, siguiente: int | None, aleatorio: int | None,
        filtro: str, indice: int,
    ) -> tuple:
        """Elige la fila y devuelve las cuatro salidas en el orden del decorador.

        Params:
            atras: Clics acumulados del botón Atrás; puede empezar como None.
            siguiente: Clics acumulados del botón Siguiente.
            aleatorio: Clics acumulados del botón Aleatorio.
            filtro: Todos los casos, estimaciones o valores predeterminados.
            indice: Posición actual recibida desde dcc.Store mediante State.

        Returns:
            tuple: Índice, figura de ortofoto, figura de nube y lista de textos.
                La navegación permanece dentro de los límites de la tabla.
                La primera llamada muestra el caso 0.
        """

        seleccion = casos if filtro == "todos" else casos[casos.origen == filtro]
        if seleccion.empty:
            return 0, go.Figure(), go.Figure(), [html.P("No hay casos con este filtro.")]

        if ctx.triggered_id == "filtro":
            indice = 0
        elif ctx.triggered_id == "atras":
            indice -= 1
        elif ctx.triggered_id == "siguiente":
            indice += 1
        elif ctx.triggered_id == "aleatorio":
            indice = int(np.random.default_rng().integers(len(seleccion)))

        indice = max(0, min(indice, len(seleccion) - 1))
        caso = seleccion.iloc[indice].copy()
        resultado = leer_resultado(caso.id_panel, resultados)
        if ortofotos is None:
            foto = figura_ortofoto(caso, carpeta)
        else:
            foto, limites = figura_ortofoto_regional(caso, ortofotos)
            for campo, valor in zip(["xmin", "ymin", "xmax", "ymax"], limites):
                caso[campo] = valor
        datos = [
            html.P(f"Caso {indice + 1} de {len(seleccion)} · Total disponible: {len(casos)}"),
            *mostrar_datos(caso, resultado),
        ]

        return (
            indice,
            foto,
            figura_nube(caso, carpeta, resultado, resultados),
            datos,
        )


def crear_app(
    carpeta: Path = CARPETA, resultados: Path = RESULTADOS,
    cantabria: bool = True, preparacion: Path = PREPARACION,
    recortes: Path = RECORTES, ortofotos: Path = ORTOFOTOS,
) -> Dash:
    """Reúne los datos, el layout y los callbacks en una aplicación.

    Params:
        carpeta: Directorio de la muestra.
        resultados: Directorio de los ajustes guardados.
        cantabria: Si es True, utiliza el inventario regional; si no, casos.csv.
        preparacion: Directorio del inventario de detecciones y ventanas.
        recortes: Directorio de las aportaciones LAZ por hoja.
        ortofotos: Directorio de los TIFF locales.

    Returns:
        Dash: Aplicación con los casos cargados, el layout y los callbacks registrados.
    """

    indice_ortofotos = None
    if cantabria:
        casos = inventariar_casos(preparacion, recortes)
        casos["crs_lidar"] = [casos.crs] * len(casos)
        indice_ortofotos = indexar_ortofotos(ortofotos)
    else:
        casos = leer_casos(carpeta)

    tabla = pd.read_csv(resultados / "resumen.csv", keep_default_na=False)
    casos = casos.merge(tabla[["id_panel", "origen"]], on="id_panel", how="left")
    casos["origen"] = casos.origen.fillna("")

    if casos.empty:
        raise ValueError("No hay casos en casos.csv.")

    app = Dash(__name__)
    app.title = "Resultados LiDAR"
    app.layout = crear_layout()
    registrar_callbacks(app, casos, carpeta, resultados, indice_ortofotos)

    return app


def main() -> None:
    """Arranca el visor local con las rutas elegidas, en el puerto 8052 por defecto."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=CARPETA)
    parser.add_argument("--resultados", type=Path, default=RESULTADOS)
    parser.add_argument("--muestra", action="store_true", help="Utiliza casos.csv en lugar del inventario regional.")
    parser.add_argument("--preparacion", type=Path, default=PREPARACION)
    parser.add_argument("--recortes", type=Path, default=RECORTES)
    parser.add_argument("--ortofotos", type=Path, default=ORTOFOTOS)
    parser.add_argument("--puerto", type=int, default=8052)
    args = parser.parse_args()

    app = crear_app(
        ruta_proyecto(args.carpeta), ruta_proyecto(args.resultados),
        not args.muestra, ruta_proyecto(args.preparacion),
        ruta_proyecto(args.recortes), ruta_proyecto(args.ortofotos),
    )
    app.run(host="127.0.0.1", port=args.puerto, debug=False, threaded=False)


if __name__ == "__main__":
    main()
