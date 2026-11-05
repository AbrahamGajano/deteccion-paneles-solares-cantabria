"""Prototipo de ajuste de un plano dominante en un caso revisado como coplanar."""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely import intersects_xy, wkt

from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/lidar/prueba_pueblos_20261105/revision")
SALIDA = ruta_proyecto("runs/lidar/planos")


def elegir_caso(casos: pd.DataFrame, panel_id: str | None = None) -> pd.Series:
    """Elige un panel indicado o el primer coplanar con correspondencia clara.

    Params:
        casos: Tabla casos.csv del visor.
        panel_id: Identificador concreto; None para buscar uno revisado.

    Returns:
        pd.Series: Fila del caso elegido.
    """

    if panel_id is not None:
        seleccion = casos[casos.id_panel == panel_id]

        if seleccion.empty:
            raise ValueError(f"El panel {panel_id} no aparece en casos.csv.")

        return seleccion.iloc[0]

    for _, caso in casos.iterrows():
        formulario = json.loads(caso.formulario) if caso.formulario else {}

        if (
            formulario.get("montaje") == "Aparentemente coplanar con cubierta"
            and formulario.get("coincidencia") == "Aparentemente alineado"
            and formulario.get("confianza") == "Alta"
        ):
            return caso

    raise ValueError("No hay un coplanar revisado con alineación clara y confianza alta.")


def seleccionar_puntos(caso: pd.Series, carpeta: Path) -> tuple:
    """Lee el contexto y localiza los puntos dentro de la detección original.

    Params:
        caso: Fila con la ruta LAZ y el polígono del panel en WKT.
        carpeta: Directorio de la muestra del visor.

    Returns:
        tuple: Matriz XYZ de contexto y máscara booleana de puntos del polígono.
    """

    nube = laspy.read(carpeta / caso.laz)
    puntos = np.column_stack([nube.x, nube.y, nube.z])
    dentro = intersects_xy(wkt.loads(caso.poligono), puntos[:, 0], puntos[:, 1])

    return puntos, dentro


def ajustar_plano(puntos: np.ndarray) -> tuple:
    """Obtiene el plano de menor error perpendicular mediante SVD.

    Params:
        puntos: Matriz de al menos tres puntos XYZ que no estén alineados.

    Returns:
        tuple: Centro del conjunto y normal unitaria orientada hacia arriba.
    """

    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos tres puntos para ajustar un plano.")

    centro = puntos.mean(axis=0)
    _, valores, vectores = np.linalg.svd(puntos - centro, full_matrices=False)

    if valores[1] < 1e-8:
        raise ValueError("Los puntos están alineados; no definen un plano único.")

    normal = vectores[-1]

    if normal[2] < 0:
        normal = -normal

    return centro, normal


def buscar_plano(
    puntos: np.ndarray, umbral: float = 0.10,
    iteraciones: int = 500, semilla: int = 42,
) -> dict:
    """Busca un plano por consenso RANSAC y lo reajusta con sus puntos compatibles.

    Params:
        puntos: Matriz XYZ de los puntos dentro de la detección.
        umbral: Distancia perpendicular máxima al plano, en metros.
        iteraciones: Número de combinaciones de tres puntos que se probarán.
        semilla: Valor para repetir la misma selección aleatoria.

    Returns:
        dict: Centro, normal, distancias y máscara de puntos compatibles.
    """

    if len(puntos) < 3:
        raise ValueError("Hay menos de tres puntos dentro de la detección.")

    if umbral <= 0 or iteraciones < 1:
        raise ValueError("El umbral y las iteraciones deben ser positivos.")

    origen = puntos.mean(axis=0)
    locales = puntos - origen
    generador = np.random.default_rng(semilla)
    mejor_mascara = None
    mejor_cantidad = 0
    mejor_error = float("inf")

    for _ in range(iteraciones):
        indices = generador.choice(len(locales), size=3, replace=False)
        p1, p2, p3 = locales[indices]
        vector_1 = p2 - p1
        vector_2 = p3 - p1
        normal = np.cross(vector_1, vector_2)
        longitud = np.linalg.norm(normal)
        escala = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

        if escala == 0 or longitud <= 1e-6 * escala:
            continue

        normal = normal / longitud
        distancias = np.abs((locales - p1) @ normal)
        compatibles = distancias <= umbral
        cantidad = int(compatibles.sum())
        error = float(distancias[compatibles].mean())

        if cantidad > mejor_cantidad or (
            cantidad == mejor_cantidad and error < mejor_error
        ):
            mejor_mascara = compatibles
            mejor_cantidad = cantidad
            mejor_error = error

    if mejor_mascara is None:
        raise ValueError("No se encontró una combinación que defina un plano.")

    centro, normal = ajustar_plano(locales[mejor_mascara])
    distancias = np.abs((locales - centro) @ normal)
    compatibles = distancias <= umbral

    if compatibles.sum() < 3:
        raise ValueError("El plano reajustado no conserva tres puntos compatibles.")

    return {
        "centro": centro + origen,
        "normal": normal,
        "distancias": distancias,
        "compatibles": compatibles,
    }


def calcular_inclinacion(normal: np.ndarray) -> float:
    """Calcula el ángulo del plano respecto a la horizontal.

    Params:
        normal: Vector perpendicular al plano y orientado hacia arriba.

    Returns:
        float: Inclinación en grados; cero corresponde a un plano horizontal.
    """

    return float(np.degrees(np.arctan2(np.hypot(normal[0], normal[1]), abs(normal[2]))))


def dibujar_ajuste(contexto: np.ndarray, dentro: np.ndarray, ajuste: dict) -> go.Figure:
    """Dibuja contexto, puntos compatibles, descartados y superficie ajustada.

    Params:
        contexto: Puntos XYZ del recorte completo.
        dentro: Máscara que identifica la detección dentro del contexto.
        ajuste: Resultado de buscar_plano aplicado a contexto[dentro].

    Returns:
        go.Figure: Vista 3D con coordenadas locales y proporciones reales.
    """

    origen = contexto.min(axis=0)
    locales = contexto - origen
    puntos = locales[dentro]
    compatibles = ajuste["compatibles"]
    centro = ajuste["centro"] - origen
    normal = ajuste["normal"]
    referencia = np.array([1., 0., 0.]) if abs(normal[2]) > 0.8 else np.array([0., 0., 1.])
    eje_u = np.cross(normal, referencia)
    eje_u = eje_u / np.linalg.norm(eje_u)
    eje_v = np.cross(normal, eje_u)
    seleccionados = puntos[compatibles] - centro
    u = seleccionados @ eje_u
    v = seleccionados @ eje_v
    esquinas = np.array([
        centro + a * eje_u + b * eje_v
        for a, b in [(u.min(), v.min()), (u.max(), v.min()),
                     (u.max(), v.max()), (u.min(), v.max())]
    ])
    figura = go.Figure()

    for datos, nombre, color, tamano in [
        (locales[~dentro], "Contexto", "#b0b0b0", 2),
        (puntos[compatibles], "Compatibles con el plano", "#159447", 4),
        (puntos[~compatibles], "Fuera del plano", "#e53935", 4),
    ]:
        figura.add_trace(go.Scatter3d(
            x=datos[:, 0], y=datos[:, 1], z=datos[:, 2], mode="markers",
            name=nombre, marker={"size": tamano, "color": color},
        ))

    figura.add_trace(go.Mesh3d(
        x=esquinas[:, 0], y=esquinas[:, 1], z=esquinas[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3], opacity=0.4,
        color="#ffad33", name="Plano ajustado", showlegend=True,
    ))
    inferior, superior = np.percentile(locales[:, 2], [1, 99])
    margen = max(1., (superior - inferior) * 0.1)
    dimensiones = np.array([
        max(float(np.ptp(locales[:, 0])), 1.),
        max(float(np.ptp(locales[:, 1])), 1.),
        superior - inferior + 2 * margen,
    ])
    proporciones = dimensiones / dimensiones.max()
    figura.update_layout(
        title="Plano dominante en la detección · prototipo para revisión",
        scene={
            "xaxis_title": "Este local (m)", "yaxis_title": "Norte local (m)",
            "zaxis": {"title": "Altura relativa (m)",
                      "range": [inferior - margen, superior + margen]},
            "aspectmode": "manual",
            "aspectratio": dict(zip(["x", "y", "z"], proporciones)),
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    return figura


def main() -> None:
    """Ejecuta un único caso y guarda métricas y una figura HTML independiente."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=CARPETA)
    parser.add_argument("--panel", help="ID del panel; por defecto elige un coplanar revisado.")
    parser.add_argument("--umbral", type=float, default=0.10)
    parser.add_argument("--iteraciones", type=int, default=500)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()
    carpeta = ruta_proyecto(args.carpeta)
    casos = pd.read_csv(carpeta / "casos.csv", keep_default_na=False)
    caso = elegir_caso(casos, args.panel)
    contexto, dentro = seleccionar_puntos(caso, carpeta)
    ajuste = buscar_plano(contexto[dentro], args.umbral, args.iteraciones, args.semilla)
    compatibles = ajuste["compatibles"]
    residuos = ajuste["distancias"][compatibles]
    resumen = {
        "id_panel": caso.id_panel,
        "laz": str(carpeta / caso.laz),
        "umbral_m": args.umbral, "iteraciones": args.iteraciones, "semilla": args.semilla,
        "n_puntos": int(dentro.sum()), "n_compatibles": int(compatibles.sum()),
        "porcentaje_compatible": float(compatibles.mean() * 100),
        "rmse_perpendicular_m": float(np.sqrt(np.mean(residuos**2))),
        "inclinacion_grados": calcular_inclinacion(ajuste["normal"]),
        "centro": ajuste["centro"].tolist(), "normal": ajuste["normal"].tolist(),
        "estado": "prototipo_pendiente_de_validacion",
    }
    salida = ruta_proyecto(args.salida) / caso.id_panel
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([{
        "x": punto[0], "y": punto[1], "z": punto[2],
        "compatible": bool(compatible), "distancia_m": float(distancia),
    } for punto, compatible, distancia in zip(
        contexto[dentro], compatibles, ajuste["distancias"]
    )]).to_csv(salida / "puntos.csv", index=False)
    dibujar_ajuste(contexto, dentro, ajuste).write_html(salida / "ajuste.html")
    print(f"Panel: {caso.id_panel}")
    print(f"Inclinación: {resumen['inclinacion_grados']:.2f} grados")
    print(f"Compatibles: {resumen['n_compatibles']}/{resumen['n_puntos']}")
    print(f"RMSE perpendicular: {resumen['rmse_perpendicular_m']:.4f} m")
    print(f"Resultado para revisar: {salida / 'ajuste.html'}")


if __name__ == "__main__":
    main()
