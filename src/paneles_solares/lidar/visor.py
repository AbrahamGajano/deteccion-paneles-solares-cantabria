"""Visor local de ortofotos, nubes 3D y formularios guardados en un CSV."""

import argparse
import base64
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from shapely import intersects_xy, wkt

from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/lidar/revision")
CAMPOS = ["coincidencia", "visibilidad", "montaje", "confianza", "notas"]
OPCIONES = {
    "coincidencia": [
        "Aparentemente alineado", "Desplazamiento ligero", "Desplazamiento grande",
        "Correspondencia dudosa", "Sin correspondencia", "No evaluable",
    ],
    "visibilidad": [
        {"label": "Panel separado de la cubierta en la nube", "value": "Panel distinguible"},
        {"label": "Superficie continua: panel y cubierta no separables",
         "value": "Solo cubierta distinguible"},
        {"label": "No sé qué superficie corresponde al panel",
         "value": "Asociación espacial dudosa"},
        {"label": "Puntos insuficientes", "value": "Puntos insuficientes"},
        {"label": "Sin puntos", "value": "Sin puntos"},
        {"label": "No interpretable", "value": "No interpretable"},
    ],
    "montaje": [
        "Aparentemente coplanar con cubierta", "Estructura independiente",
        "Varios planos", "En suelo", "Indeterminado",
    ],
    "confianza": ["Alta", "Media", "Baja"],
}


def leer_casos(carpeta: Path) -> pd.DataFrame:
    """Lee la muestra manteniendo los formularios vacíos como texto vacío.

    Params:
        carpeta: Directorio que contiene casos.csv.

    Returns:
        pd.DataFrame: Casos en el mismo orden en que fueron seleccionados.
    """

    return pd.read_csv(carpeta / "casos.csv", keep_default_na=False)


def leer_formulario(texto: str) -> dict:
    """Recupera las respuestas o prepara campos vacíos para un caso pendiente.

    Params:
        texto: JSON de la columna formulario; vacío si todavía no hay respuestas.

    Returns:
        dict: Valores de todos los campos del formulario.
    """

    formulario = dict.fromkeys(CAMPOS, "")

    if texto:
        formulario.update(json.loads(texto))

    return formulario


def guardar_formulario(carpeta: Path, indice: int, formulario: dict) -> pd.DataFrame:
    """Guarda las respuestas del caso actual en la columna formulario del CSV.

    Params:
        carpeta: Directorio de la muestra.
        indice: Posición de la fila que se está revisando.
        formulario: Respuestas; un formulario completamente vacío sigue pendiente.

    Returns:
        pd.DataFrame: Tabla actualizada, ya guardada en disco.
    """

    casos = leer_casos(carpeta)
    valores = {campo: formulario.get(campo) or "" for campo in CAMPOS}
    texto = json.dumps(valores, ensure_ascii=False) if any(valores.values()) else ""
    casos.loc[indice, "formulario"] = texto
    casos.to_csv(carpeta / "casos.csv", index=False, encoding="utf-8")

    return casos


def figura_ortofoto(caso: pd.Series, carpeta: Path) -> go.Figure:
    """Muestra el PNG con la detección original dibujada sobre él.

    Params:
        caso: Fila del CSV con imagen, límites geográficos y polígono original.
        carpeta: Directorio de la muestra.

    Returns:
        go.Figure: Ortofoto orientada al norte, con ejes locales en metros.
    """

    ruta = carpeta / caso.imagen
    contenido = base64.b64encode(ruta.read_bytes()).decode("ascii")

    ancho_m = caso.xmax - caso.xmin
    alto_m = caso.ymax - caso.ymin
    figura = go.Figure()
    figura.add_layout_image(
        source=f"data:image/png;base64,{contenido}",
        xref="x", yref="y", x=0, y=alto_m,
        sizex=ancho_m, sizey=alto_m,
        xanchor="left", yanchor="top", sizing="stretch", layer="below",
    )
    geometria = wkt.loads(caso.poligono)
    poligonos = list(geometria.geoms) if geometria.geom_type == "MultiPolygon" else [geometria]

    for poligono in poligonos:
        for anillo in [poligono.exterior, *poligono.interiors]:
            x, y = anillo.xy
            figura.add_trace(go.Scatter(
                x=np.asarray(x) - caso.xmin, y=np.asarray(y) - caso.ymin,
                mode="lines", line={"color": "#ff3355", "width": 2},
                showlegend=False, hoverinfo="skip",
            ))

    figura.update_layout(
        margin={"l": 45, "r": 10, "t": 15, "b": 40},
        xaxis={"title": "Este local (m)", "range": [0, ancho_m]},
        yaxis={"title": "Norte local (m)", "range": [0, alto_m],
               "autorange": False, "scaleanchor": "x"},
    )

    return figura


def limites_altura(z: np.ndarray) -> tuple:
    """Calcula un encuadre vertical que no dependa de unos pocos extremos.

    Params:
        z: Cotas de los puntos de la nube, en metros.

    Returns:
        tuple: Límites de los percentiles 1 y 99 ampliados un 10 %, mínimo 1 m.
    """

    inferior, superior = np.percentile(z, [1, 99])
    margen = max(1.0, (superior - inferior) * 0.1)

    return float(inferior - margen), float(superior + margen)


def proporciones_escena(ancho: float, fondo: float, altura: float) -> dict:
    """Expresa las dimensiones reales como proporciones manejables por la cámara.

    Params:
        ancho: Extensión del eje X, en metros.
        fondo: Extensión del eje Y, en metros.
        altura: Extensión del eje Z, en metros.

    Returns:
        dict: Proporciones X/Y/Z con la dimensión mayor igual a uno.
    """

    mayor = max(ancho, fondo, altura)

    return {"x": ancho / mayor, "y": fondo / mayor, "z": altura / mayor}


def figura_nube(caso: pd.Series, carpeta: Path) -> go.Figure:
    """Dibuja la nube y destaca la huella del panel con un encuadre por percentiles.

    Params:
        caso: Fila del CSV con la ruta LAZ y el origen horizontal del contexto.
        carpeta: Directorio de la muestra.

    Returns:
        go.Figure: Nube 3D con encuadre por percentiles y detección resaltada.
    """

    nube = laspy.read(carpeta / caso.laz)
    figura = go.Figure()

    if len(nube.points):
        x = np.asarray(nube.x)
        y = np.asarray(nube.y)
        z = np.asarray(nube.z)
        geometria = wkt.loads(caso.poligono)
        dentro = intersects_xy(geometria, x, y)
        inferior, superior = limites_altura(z)
        origen_z = inferior
        fuera = int(((z < inferior) | (z > superior)).sum())
        texto_habitual = f"Escala habitual · {fuera} de {len(z)} puntos fuera del encuadre"
        figura.add_trace(go.Scatter3d(
            x=x[~dentro] - caso.xmin, y=y[~dentro] - caso.ymin,
            z=z[~dentro] - origen_z, mode="markers", name="Contexto",
            marker={"size": 2, "color": z[~dentro], "colorscale": "Viridis",
                    "cmin": inferior, "cmax": superior,
                    "colorbar": {"title": "Cota (m)", "thickness": 12}},
        ))
        figura.add_trace(go.Scatter3d(
            x=x[dentro] - caso.xmin, y=y[dentro] - caso.ymin,
            z=z[dentro] - origen_z, mode="markers",
            name=f"Bajo detección 2D ({int(dentro.sum())} puntos)",
            marker={"size": 4, "color": "#ff3355"},
            hovertemplate=(
                "Dentro de la huella XY original<br>"
                "Este: %{x:.2f} m<br>Norte: %{y:.2f} m<br>"
                "Altura relativa: %{z:.2f} m<extra></extra>"
            ),
        ))
        poligonos = list(geometria.geoms) if geometria.geom_type == "MultiPolygon" else [geometria]

        for poligono in poligonos:
            for anillo in [poligono.exterior, *poligono.interiors]:
                borde_x, borde_y = anillo.xy
                figura.add_trace(go.Scatter3d(
                    x=np.asarray(borde_x) - caso.xmin,
                    y=np.asarray(borde_y) - caso.ymin,
                    z=np.zeros(len(borde_x)), mode="lines",
                    line={"color": "#ff3355", "width": 4, "dash": "dash"},
                    showlegend=False,
                    hovertemplate="Huella XY en el plano de referencia; no es la altura del panel<extra></extra>",
                ))
    else:
        figura.add_annotation(text="Este caso no tiene puntos LiDAR", showarrow=False)

    figura.update_layout(
        margin={"l": 0, "r": 0, "t": 25, "b": 35},
        legend={"orientation": "h", "x": 0, "y": -0.06},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "Este local (m)", "range": [0, caso.xmax - caso.xmin]},
            "yaxis": {"title": "Norte local (m)", "range": [0, caso.ymax - caso.ymin]},
            "zaxis_title": "Altura relativa (m)",
            "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.8}},
        },
    )

    if len(nube.points):
        proporcion_habitual = proporciones_escena(
            caso.xmax - caso.xmin, caso.ymax - caso.ymin, superior - inferior
        )
        figura.update_layout(
            scene_zaxis_range=[0, superior - origen_z],
            scene_zaxis_autorange=False,
            scene_aspectmode="manual",
            scene_aspectratio=proporcion_habitual,
        )
        figura.add_annotation(
            text=texto_habitual, x=0, y=1, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False,
            font={"size": 11},
        )

    return figura


def mostrar_caso(casos: pd.DataFrame, indice: int, carpeta: Path) -> tuple:
    """Prepara los valores de pantalla para una fila del CSV.

    Params:
        casos: Tabla de la muestra con las respuestas actuales.
        indice: Posición del caso que se quiere mostrar.
        carpeta: Directorio de la muestra.

    Returns:
        tuple: Índice, título, figuras, formulario, progreso y estado de botones.
    """

    caso = casos.iloc[indice]
    formulario = leer_formulario(caso.formulario)
    respondidos = casos.formulario.ne("").sum()
    titulo = f"Caso {indice + 1} de {len(casos)} · {caso.id_panel} · {caso.n_puntos} puntos"
    estado = f"Con respuestas: {respondidos}/{len(casos)}. Cambios guardados en casos.csv."

    return (
        indice, titulo, figura_ortofoto(caso, carpeta), figura_nube(caso, carpeta),
        *[formulario[campo] for campo in CAMPOS], estado,
        indice == 0, indice == len(casos) - 1,
    )


def crear_app(carpeta: Path) -> Dash:
    """Construye las tres columnas y conecta los controles con el guardado.

    Params:
        carpeta: Muestra creada por preparar_revision.py.

    Returns:
        Dash: Aplicación lista para ejecutarse en el navegador local.
    """

    app = Dash(__name__)
    app.title = "Revisión LiDAR"

    def crear_pantalla():
        """Abre el primer formulario vacío cada vez que se carga la página."""

        casos = leer_casos(carpeta)
        pendientes = casos.index[casos.formulario.eq("")].tolist()
        indice = pendientes[0] if pendientes else 0
        valores = mostrar_caso(casos, indice, carpeta)
        formulario = []

        for campo, etiqueta in [
            ("coincidencia", "Coincidencia espacial"),
            ("visibilidad", "¿Se separa el panel de la cubierta en la nube?"),
            ("montaje", "Montaje aparente (si se puede determinar)"), ("confianza", "Confianza"),
        ]:
            formulario.extend([
                html.Label(etiqueta, style={"display": "block", "marginTop": "18px"}),
                dcc.Dropdown(
                    id=campo, options=OPCIONES[campo],
                    value=valores[4 + CAMPOS.index(campo)],
                    placeholder="Sin responder", clearable=True,
                ),
            ])

        formulario.extend([
            html.P(
                "Una superficie continua no confirma que sean coplanares. "
                "Si no puedes determinar el montaje, elige Indeterminado.",
                style={"fontSize": "12px", "color": "#555"},
            ),
            html.Label("Observaciones", style={"display": "block", "marginTop": "18px"}),
            dcc.Textarea(id="notas", value=valores[8],
                         style={"width": "100%", "height": "130px"}),
            html.P("Cada cambio se guarda automáticamente. Vaciar todos los campos deja el caso pendiente."),
            html.Div([
                html.Button("Atrás", id="atras", n_clicks=0, disabled=valores[10]),
                html.Button("Guardar", id="guardar", n_clicks=0),
                html.Button("Siguiente", id="siguiente", n_clicks=0, disabled=valores[11]),
            ], style={"display": "flex", "gap": "10px"}),
            html.P(valores[9], id="estado"),
        ])

        return html.Div([
            dcc.Store(id="indice", data=indice),
            html.H2("Revisión de paneles con LiDAR"),
            html.P(valores[1], id="titulo"),
            html.Div([
                html.Div([html.H3("Ortofoto · detección en rojo"),
                          dcc.Graph(id="ortofoto", figure=valores[2], style={"height": "72vh"})]),
                html.Div([html.H3("Nube de puntos"),
                          dcc.Graph(id="nube", figure=valores[3], style={"height": "72vh"}),
                          html.P(
                              "Rojo: puntos dentro de la huella XY de la detección. "
                              "La línea discontinua muestra esa huella en la base de referencia. "
                              "Esto localiza la zona; no confirma que esos puntos sean del panel.",
                              style={"fontSize": "12px", "color": "#555"},
                          )]),
                html.Div([html.H3("Formulario"), *formulario]),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 330px", "gap": "18px"}),
        ], style={"fontFamily": "Arial, sans-serif", "padding": "16px"})

    app.layout = crear_pantalla

    @app.callback(
        Output("indice", "data"), Output("titulo", "children"),
        Output("ortofoto", "figure"), Output("nube", "figure"),
        Output("coincidencia", "value"), Output("visibilidad", "value"),
        Output("montaje", "value"), Output("confianza", "value"),
        Output("notas", "value"), Output("estado", "children"),
        Output("atras", "disabled"), Output("siguiente", "disabled"),
        Input("atras", "n_clicks"), Input("siguiente", "n_clicks"),
        Input("guardar", "n_clicks"),
        Input("coincidencia", "value"), Input("visibilidad", "value"),
        Input("montaje", "value"), Input("confianza", "value"), Input("notas", "value"),
        State("indice", "data"), prevent_initial_call=True,
    )
    def actualizar(atras, siguiente, guardar, coincidencia, visibilidad,
                   montaje, confianza, notas, indice):
        """Guarda el formulario y responde al control que cambió en la pantalla."""

        formulario = dict(zip(CAMPOS, [coincidencia, visibilidad, montaje, confianza, notas]))
        casos = guardar_formulario(carpeta, indice, formulario)

        if ctx.triggered_id == "siguiente":
            indice = min(indice + 1, len(casos) - 1)
        elif ctx.triggered_id == "atras":
            indice = max(indice - 1, 0)
        else:
            respuesta = [no_update] * 12
            respondidos = casos.formulario.ne("").sum()
            respuesta[9] = f"Guardado. Con respuestas: {respondidos}/{len(casos)}."
            return tuple(respuesta)

        return mostrar_caso(casos, indice, carpeta)

    return app


def main() -> None:
    """Arranca el visor local sobre una muestra ya preparada."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=CARPETA)
    parser.add_argument("--puerto", type=int, default=8050)
    args = parser.parse_args()
    carpeta = ruta_proyecto(args.carpeta)

    if not (carpeta / "casos.csv").exists():
        parser.error("Primero ejecuta python -m paneles_solares.lidar.preparar_revision.")

    app = crear_app(carpeta)
    app.run(host="127.0.0.1", port=args.puerto, debug=False, threaded=False)


if __name__ == "__main__":
    main()
