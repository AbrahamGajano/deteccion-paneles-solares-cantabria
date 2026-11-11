"""Visor Dash para comprobar resultados de planos."""

import argparse
import base64
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from shapely import wkt, intersects_xy

from paneles_solares.lidar.estimar_planos import agregar_plano
from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/lidar/prueba_pueblos_20261105/revision")
RESULTADOS = ruta_proyecto("runs/lidar/validacion_100")


def leer_casos(carpeta: Path) -> pd.DataFrame:
    """Lee casos.csv conservando el orden y los campos vacíos como texto.

    Params:
        carpeta: Directorio de la muestra, con casos.csv, imagenes/ y nubes/.

    Returns:
        pd.DataFrame: Tabla con id_panel, imagen, laz, poligono y límites XY.
    """

    return pd.read_csv(carpeta / "casos.csv", keep_default_na=False)


def leer_resultado(id_panel: str, resultados: Path) -> dict | None:
    """Lee resultados/id_panel/resumen.json del estimador.

    Params:
        id_panel: Identificador del caso que se muestra.
        resultados: Directorio de salida de estimar_planos.py.

    Returns:
        dict | None: JSON leído, o None si el caso todavía no tiene resultado.
            centro, normal y cubierta también pueden ser None dentro del JSON.
    """

    ruta = resultados / id_panel / "resumen.json"

    if not ruta.exists():
        return None

    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def figura_ortofoto(caso: pd.Series, carpeta: Path) -> go.Figure:
    """Construye una figura con el PNG y el contorno original del panel.

    Params:
        caso: Fila con imagen, poligono en WKT y límites xmin/ymin/xmax/ymax.
        carpeta: Directorio de la muestra; la ruta de imagen es relativa a él.

    Returns:
        go.Figure: Imagen orientada al norte y contorno en coordenadas locales.
    """

    ruta_img = carpeta / caso.imagen

    with open(ruta_img, "rb") as imagen:
        datos_imagen = base64.b64encode(imagen.read()).decode("ascii")

    ancho = caso.xmax - caso.xmin
    alto = caso.ymax - caso.ymin
    fig = go.Figure()

    fig.add_layout_image(
        source=f"data:image/png;base64,{datos_imagen}",
        xref="x", yref="y", x=0, y=alto,
        sizex=ancho, sizey=alto,
        xanchor="left", yanchor="top", sizing="stretch", layer="below",
    )

    geometria = wkt.loads(caso.poligono)

    poligonos = (
        list(geometria.geoms) if geometria.geom_type == "MultiPolygon" else [geometria]
    )

    for poligono in poligonos:
        for anillo in [poligono.exterior, *poligono.interiors]:
            x, y = anillo.xy
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(x) - caso.xmin,
                    y=np.asarray(y) - caso.ymin,
                    mode="lines",
                    line={"color": "red"},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        margin={"l": 45, "r": 10, "t": 15, "b": 40},
        xaxis={"title": "Este local (m)", "range": [0, ancho]},
        yaxis={"title": "Norte local (m)", "range": [0, alto],
               "autorange": False, "scaleanchor": "x"},
    )

    return fig


def figura_nube(
    caso: pd.Series, carpeta: Path, resultado: dict | None,
    resultados: Path = RESULTADOS,
) -> go.Figure:
    """Dibuja la nube 3D, la detección y los planos guardados del ajuste.

    Params:
        caso: Fila con laz, poligono y límites geográficos del recorte.
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

    ruta_laz = carpeta / caso.laz

    nube = laspy.read(ruta_laz)

    if not len(nube.points):
        return fig

    x = np.asarray(nube.x)
    y = np.asarray(nube.y)
    z = np.asarray(nube.z)

    geom = wkt.loads(caso.poligono)

    dentro = intersects_xy(geom, x, y)

    inferior, superior = np.percentile(z, [1, 99])
    margen = max(1., (superior - inferior) * 0.1)
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
            marker={"size": 2, "color": z[~dentro], "colorscale": "Viridis",
                    "cmin": inferior, "cmax": max(superior, inferior + 0.1),
                    "showscale": False},
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
            fig.add_trace(go.Scatter3d(
                x=np.asarray(borde_x) - caso.xmin,
                y=np.asarray(borde_y) - caso.ymin,
                z=np.zeros(len(borde_x)),
                mode="lines",
                line={"color": "red", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ))

    if resultado and resultado.get("normal") is not None:
        centro = np.asarray(resultado["centro"])
        normal = np.asarray(resultado["normal"])
        distancias = np.abs((puntos - centro) @ normal)
        compatibles = dentro & (distancias <= resultado["umbral_m"])

        if compatibles.sum() >= 3:
            agregar_plano(fig, puntos[compatibles] - origen, centro - origen,
                           normal, "#ffad33", "Plano interior candidato")

    cubierta = resultado.get("cubierta") if resultado else None
    ruta_cubierta = resultados / caso.id_panel / "cubierta.csv"

    if cubierta and ruta_cubierta.exists():
        puntos_cubierta = pd.read_csv(ruta_cubierta)[["x", "y", "z"]].to_numpy()

        if len(puntos_cubierta) >= 3:
            locales = puntos_cubierta - origen
            fig.add_trace(go.Scatter3d(
                x=locales[:, 0], y=locales[:, 1], z=locales[:, 2],
                mode="markers", name="Fragmento exterior asociado",
                marker={"size": 3, "color": "#1645ab", "symbol": "diamond"},
            ))
            agregar_plano(fig, locales, np.asarray(cubierta["centro"]) - origen,
                           np.asarray(cubierta["normal"]), "#168be0", "Plano exterior")

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
    """Expresa una medida con dos decimales o indica que es desconocida.

    Params:
        valor: Medida numérica, o None si no está disponible.
        unidad: Sufijo con la unidad de la medida.

    Returns:
        str: Valor formateado, conservando el cero como medida válida.
    """

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
        ("Inclinación candidata", formatear_numero(resultado.get("inclinacion_candidata"), "°")),
        ("Inclinación estimada", formatear_numero(resultado.get("inclinacion_estimada"), "°")),
        ("Inclinación utilizada", formatear_numero(resultado.get("inclinacion_utilizada"), "°")),
        ("Origen", resultado["origen"]),
        ("Motivo", resultado["motivo"].replace("_", " ")),
        ("Inliers interiores", formatear_numero(resultado.get("porcentaje_compatible"), " %")),
        ("RMSE interior", formatear_numero(resultado.get("rmse_perpendicular_m"), " m")),
    ]
    datos.extend(html.P([html.Strong(f"{nombre}: "), valor]) for nombre, valor in medidas)

    cubierta = resultado.get("cubierta")
    if cubierta:
        datos.extend([
            html.H4("Comparación con el exterior"),
            html.P([html.Strong("Ángulo entre planos: "),
                    formatear_numero(cubierta["angulo_entre_planos_grados"], "°")]),
            html.P([html.Strong("Separación p90: "),
                    formatear_numero(cubierta["separacion_p90_m"], " m")]),
        ])
    else:
        datos.append(html.P("Sin plano exterior asociado."))

    return datos


def crear_layout() -> html.Div:
    """Define los componentes de la página y su distribución en tres columnas.

    Params:
        Ninguno.

    Returns:
        html.Div: Contenedor con dcc.Graph(id="ortofoto"), dcc.Graph(id="nube")
            y html.Div(id="datos"), botones con id="atras" e id="siguiente"
            y dcc.Store(id="indice", data=0) con la posición actual.
    """

    return html.Div([
        html.H2("Revisión de resultados LiDAR"),
        dcc.Store(id="indice", data=0),
        html.Div([
            html.Button("Atrás", id="atras", n_clicks=0),
            html.Button("Siguiente", id="siguiente", n_clicks=0),
        ], style={"display": "flex", "gap": "10px", "marginBottom": "15px"}),
        html.Div([
            html.Div([
                html.H3("Ortofoto"),
                dcc.Graph(id="ortofoto", style={"height": "72vh"}),
            ], style={"minWidth": 0}),
            html.Div([
                html.H3("Nube y planos"),
                dcc.Graph(id="nube", style={"height": "72vh"}),
            ], style={"minWidth": 0}),
            html.Div([
                html.H3("Resultado"),
                html.Div(id="datos"),
                html.Hr(),
                html.P("Rojo: puntos dentro del polígono detectado. "
                       "El contorno discontinuo muestra su proyección XY."),
                html.P("Los planos dibujados son candidatos geométricos. "
                       "Una inclinación predeterminada es un valor supuesto."),
            ]),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 280px", "gap": "18px"}),
    ], style={"fontFamily": "Arial, sans-serif", "margin": "20px"})


def registrar_callbacks(
    app: Dash, casos: pd.DataFrame, carpeta: Path, resultados: Path
) -> None:
    """Conecta los botones con las figuras y los datos de la página.

    Params:
        app: Aplicación Dash cuyo layout ya contiene los identificadores indicados.
        casos: Tabla de la muestra, accesible desde la función interior.
        carpeta: Directorio de imágenes y nubes.
        resultados: Directorio de los ajustes guardados.

    Returns:
        None: El decorador registra la función que Dash ejecutará al navegar.
    """

    @app.callback(
        Output("indice", "data"),
        Output("ortofoto", "figure"),
        Output("nube", "figure"),
        Output("datos", "children"),
        Input("atras", "n_clicks"),
        Input("siguiente", "n_clicks"),
        State("indice", "data"),
    )
    def actualizar_vista(
        atras: int | None, siguiente: int | None, indice: int
    ) -> tuple:
        """Elige la fila y devuelve las cuatro salidas en el orden del decorador.

        Params:
            atras: Clics acumulados del botón Atrás; puede empezar como None.
            siguiente: Clics acumulados del botón Siguiente.
            indice: Posición actual recibida desde dcc.Store mediante State.

        Returns:
            tuple: Índice, figura de ortofoto, figura de nube y lista de textos.
                La navegación permanece dentro de los límites de la tabla.
                La primera llamada muestra el caso 0.
        """

        if ctx.triggered_id == "atras":
            indice -= 1
        elif ctx.triggered_id == "siguiente":
            indice += 1

        indice = max(0, min(indice, len(casos) - 1))
        caso = casos.iloc[indice]
        resultado = leer_resultado(caso.id_panel, resultados)
        datos = [html.P(f"Caso {indice + 1} de {len(casos)}"),
                 *mostrar_datos(caso, resultado)]

        return (
            indice,
            figura_ortofoto(caso, carpeta),
            figura_nube(caso, carpeta, resultado, resultados),
            datos,
        )


def crear_app(carpeta: Path = CARPETA, resultados: Path = RESULTADOS) -> Dash:
    """Reúne los datos, el layout y los callbacks en una aplicación.

    Params:
        carpeta: Directorio de la muestra.
        resultados: Directorio de los ajustes guardados.

    Returns:
        Dash: Aplicación con los casos cargados, el layout y los callbacks registrados.
    """

    casos = leer_casos(carpeta)

    if casos.empty:
        raise ValueError("No hay casos en casos.csv.")

    app = Dash(__name__)
    app.title = "Resultados LiDAR"
    app.layout = crear_layout()
    registrar_callbacks(app, casos, carpeta, resultados)

    return app


def main() -> None:
    """Arranca el visor con las rutas y el puerto recibidos por terminal.

    Params:
        Ninguno.

    Returns:
        None: Servidor local en el puerto indicado, 8052 por defecto.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=CARPETA)
    parser.add_argument("--resultados", type=Path, default=RESULTADOS)
    parser.add_argument("--puerto", type=int, default=8052)
    args = parser.parse_args()

    app = crear_app(ruta_proyecto(args.carpeta), ruta_proyecto(args.resultados))
    app.run(host="127.0.0.1", port=args.puerto, debug=False, threaded=False)


if __name__ == "__main__":
    main()
