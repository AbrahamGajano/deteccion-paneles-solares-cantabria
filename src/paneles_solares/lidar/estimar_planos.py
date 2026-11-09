"""Estima inclinaciones compatibles con cubierta y registra los casos asumidos."""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely import intersects_xy, wkt
from shapely.geometry import box

from paneles_solares.lidar.ajuste_planos import (
    CRITERIOS, calcular_inclinacion, decidir_inclinacion, distancias_al_plano,
)
from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/lidar/prueba_pueblos_20261105/revision")
SALIDA = ruta_proyecto("runs/lidar/planos")


def elegir_caso(casos: pd.DataFrame, panel_id: str | None = None) -> pd.Series:
    """Elige un panel indicado o el primero de la tabla, sin usar sus etiquetas.

    Params:
        casos: Tabla casos.csv del visor.
        panel_id: Identificador concreto; None para elegir la primera fila.

    Returns:
        pd.Series: Fila del caso elegido.
    """

    if panel_id is not None:
        seleccion = casos[casos.id_panel == panel_id]

        if seleccion.empty:
            raise ValueError(f"El panel {panel_id} no aparece en casos.csv.")

        return seleccion.iloc[0]

    if casos.empty:
        raise ValueError("No hay casos para procesar.")

    return casos.iloc[0]


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


def evaluar_apoyo(distancias: np.ndarray, umbral: float) -> dict:
    """Resume el error y el apoyo manteniendo todos los puntos en el denominador.

    Params:
        distancias: Distancias con signo al plano, en metros.
        umbral: Distancia absoluta máxima para considerar un punto compatible.

    Returns:
        dict: Recuentos, porcentaje de apoyo y errores de todos los puntos y compatibles.
    """

    absolutas = np.abs(distancias)
    compatibles = absolutas <= umbral
    errores = absolutas[compatibles]
    cantidad = len(distancias)

    return {
        "n_puntos": cantidad,
        "n_compatibles": int(compatibles.sum()),
        "n_por_debajo": int((distancias < -umbral).sum()),
        "n_por_encima": int((distancias > umbral).sum()),
        "porcentaje_compatible": float(compatibles.mean() * 100) if cantidad else None,
        "suma_distancias_m": float(absolutas.sum()),
        "error_medio_absoluto_m": float(absolutas.mean()) if cantidad else None,
        "rmse_todos_m": float(np.sqrt(np.mean(distancias**2))) if cantidad else None,
        "mediana_absoluta_m": float(np.median(absolutas)) if cantidad else None,
        "p90_absoluto_m": float(np.percentile(absolutas, 90)) if cantidad else None,
        "mediana_con_signo_m": float(np.median(distancias)) if cantidad else None,
        "rmse_compatibles_m": float(np.sqrt(np.mean(errores**2))) if len(errores) else None,
    }


def evaluar_entorno(
    contexto: np.ndarray, poligono, dentro: np.ndarray,
    ajuste: dict, ancho: float = 2.0, umbral: float = 0.10,
) -> dict:
    """Comprueba la prolongación del plano en una franja exterior a la detección.

    Params:
        contexto: Matriz XYZ del recorte completo.
        poligono: Geometría original de la detección, en coordenadas métricas.
        dentro: Máscara de los puntos interiores usados para buscar el plano.
        ajuste: Plano obtenido exclusivamente con puntos interiores.
        ancho: Anchura de la franja exterior, en metros.
        umbral: Distancia perpendicular máxima para el apoyo exterior.

    Returns:
        dict: Máscara de la franja, distancias de todo el contexto y apoyo por cuadrante.
    """

    if ancho <= 0 or umbral <= 0:
        raise ValueError("El ancho de la franja y el umbral deben ser positivos.")

    exterior = intersects_xy(poligono.buffer(ancho), contexto[:, 0], contexto[:, 1]) & ~dentro
    distancias = distancias_al_plano(contexto, ajuste["centro"], ajuste["normal"])
    puntos = contexto[exterior]
    centro = poligono.centroid
    cuadrantes = np.where(
        puntos[:, 0] >= centro.x,
        np.where(puntos[:, 1] >= centro.y, "NE", "SE"),
        np.where(puntos[:, 1] >= centro.y, "NO", "SO"),
    )
    apoyo = {
        nombre: evaluar_apoyo(distancias[exterior][cuadrantes == nombre], umbral)
        for nombre in ["NE", "NO", "SE", "SO"]
    }

    return {
        "exterior": exterior,
        "distancias": distancias,
        "resumen": evaluar_apoyo(distancias[exterior], umbral),
        "cuadrantes": apoyo,
    }


def agregar_plano(figura: go.Figure, puntos: np.ndarray, centro: np.ndarray,
                  normal: np.ndarray, color: str, nombre: str) -> None:
    """Añade un rectángulo que representa la extensión matemática de un plano.

    Params:
        figura: Figura que recibirá la superficie.
        puntos: Puntos de apoyo en coordenadas locales.
        centro: Centro del plano en las mismas coordenadas.
        normal: Normal unitaria.
        color: Color de la superficie.
        nombre: Etiqueta de la leyenda.

    Returns:
        None.
    """

    referencia = np.array([1., 0., 0.]) if abs(normal[2]) > 0.8 else np.array([0., 0., 1.])
    eje_u = np.cross(normal, referencia)
    eje_u = eje_u / np.linalg.norm(eje_u)
    eje_v = np.cross(normal, eje_u)
    u = (puntos - centro) @ eje_u
    v = (puntos - centro) @ eje_v
    esquinas = np.array([
        centro + a * eje_u + b * eje_v
        for a, b in [(u.min(), v.min()), (u.max(), v.min()),
                     (u.max(), v.max()), (u.min(), v.max())]
    ])
    figura.add_trace(go.Mesh3d(
        x=esquinas[:, 0], y=esquinas[:, 1], z=esquinas[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3], opacity=0.35,
        color=color, name=nombre, showlegend=True,
    ))


def dibujar_ajuste(
    contexto: np.ndarray, dentro: np.ndarray, ajuste: dict,
    entorno: dict, umbral_exterior: float, cubierta: dict | None = None,
) -> go.Figure:
    """Dibuja el plano, su ajuste interior y el apoyo exterior, sin ocultar descartes.

    Params:
        contexto: Puntos XYZ del recorte completo.
        dentro: Máscara que identifica la detección dentro del contexto.
        ajuste: Resultado de buscar_plano aplicado a contexto[dentro].
        entorno: Resultado de evaluar_entorno, con la franja y sus distancias.
        umbral_exterior: Distancia máxima para colorear un punto como apoyo exterior.
        cubierta: Fragmento exterior independiente, si se ha encontrado uno.

    Returns:
        go.Figure: Vista 3D con coordenadas locales y proporciones reales.
    """

    origen = contexto.min(axis=0)
    locales = contexto - origen
    puntos = locales[dentro]
    compatibles = ajuste["compatibles"]
    exterior = entorno["exterior"]
    distancias = entorno["distancias"]
    apoyo = exterior & (np.abs(distancias) <= umbral_exterior)
    centro = ajuste["centro"] - origen
    normal = ajuste["normal"]
    figura = go.Figure()
    interior_compatible = np.zeros(len(contexto), dtype=bool)
    interior_compatible[dentro] = compatibles

    for mascara, nombre, color, tamano in [
        (~(dentro | exterior), "Contexto", "#b0b0b0", 2),
        (interior_compatible, "Interior compatible", "#159447", 4),
        (dentro & ~interior_compatible, "Interior fuera del plano", "#e53935", 4),
        (apoyo, "Exterior compatible", "#168be0", 3),
        (exterior & (distancias > umbral_exterior),
         "Exterior por encima", "#f28c28", 3),
        (exterior & (distancias < -umbral_exterior),
         "Exterior por debajo", "#9855b5", 3),
    ]:
        datos = locales[mascara]
        figura.add_trace(go.Scatter3d(
            x=datos[:, 0], y=datos[:, 1], z=datos[:, 2], mode="markers",
            name=nombre, marker={"size": tamano, "color": color},
            customdata=distancias[mascara],
            hovertemplate=(
                "Este: %{x:.2f} m<br>Norte: %{y:.2f} m<br>"
                "Altura relativa: %{z:.2f} m<br>"
                "Distancia con signo: %{customdata:.3f} m<extra>%{fullData.name}</extra>"
            ),
        ))

    agregar_plano(figura, puntos[compatibles], centro, normal, "#ffad33", "Plano interior")
    if cubierta is not None:
        datos = cubierta["puntos"] - origen
        plano = cubierta["ajuste"]
        figura.add_trace(go.Scatter3d(
            x=datos[:, 0], y=datos[:, 1], z=datos[:, 2], mode="markers",
            name="Fragmento exterior elegido", marker={"size": 4, "color": "#1645ab", "symbol": "diamond"},
        ))
        agregar_plano(figura, datos, plano["centro"] - origen, plano["normal"],
                       "#168be0", "Plano exterior independiente")
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


def procesar_caso(
    caso: pd.Series, carpeta: Path, salida: Path,
    umbral: float = 0.10, umbral_exterior: float = 0.10,
    anillo: float = 2.0, iteraciones: int = 500,
    semilla: int = 42, guardar_html: bool = True, criterios: dict | None = None,
) -> dict:
    """Guarda el ajuste candidato, la decisión y su procedencia sin usar etiquetas.

    Params:
        caso: Fila del CSV, incluidas sus anotaciones manuales si existen.
        carpeta: Directorio de la muestra con los LAZ.
        salida: Directorio raíz de resultados.
        umbral: Distancia perpendicular admitida en el ajuste interior, en metros.
        umbral_exterior: Distancia admitida para el apoyo exterior, en metros.
        anillo: Anchura de la franja exterior, en metros.
        iteraciones: Número de pruebas de RANSAC.
        semilla: Valor reproducible del generador aleatorio.
        guardar_html: Si se debe crear la figura 3D independiente.
        criterios: Cambios sobre los valores iniciales de CRITERIOS.

    Returns:
        dict: Medidas y parámetros; las etiquetas manuales solo se guardan como referencia.
    """

    criterios = CRITERIOS | (criterios or {})
    contexto, dentro = seleccionar_puntos(caso, carpeta)
    poligono = wkt.loads(caso.poligono)
    formulario = json.loads(caso.formulario) if caso.formulario else {}
    ventana = None
    if all(campo in caso.index for campo in ["xmin", "ymin", "xmax", "ymax"]):
        ventana = box(caso.xmin, caso.ymin, caso.xmax, caso.ymax)

    resultado = decidir_inclinacion(contexto, poligono, dentro, umbral, umbral_exterior,
                                    anillo, iteraciones, semilla, criterios, ventana)
    ajuste = resultado["ajuste"]
    decision = resultado["decision"]
    intentos = decision["intentos_exteriores"]
    ancho_evaluado = intentos[-1]["anillo_m"] if intentos else anillo
    franja_completa = bool(ventana.covers(poligono.buffer(ancho_evaluado))) if ventana is not None else None
    entorno = None
    interior = evaluar_apoyo(np.array([]), umbral)
    exterior = evaluar_apoyo(np.array([]), umbral_exterior)

    if ajuste is not None:
        entorno = evaluar_entorno(contexto, poligono, dentro, ajuste, ancho_evaluado, umbral_exterior)
        interior = evaluar_apoyo(entorno["distancias"][dentro], umbral)
        exterior = entorno["resumen"]
    else:
        franja = intersects_xy(poligono.buffer(ancho_evaluado), contexto[:, 0], contexto[:, 1]) & ~dentro
        for medidas, cantidad in [(interior, int(dentro.sum())), (exterior, int(franja.sum()))]:
            medidas.update(n_puntos=cantidad, n_compatibles=None,
                           n_por_debajo=None, n_por_encima=None, suma_distancias_m=None)

    resumen = {
        "id_panel": caso.id_panel,
        "laz": str(carpeta / caso.laz),
        "umbral_m": umbral, "umbral_exterior_m": umbral_exterior, "anillo_m": anillo,
        "iteraciones": iteraciones, "semilla": semilla,
        "criterios": criterios, "anillo_evaluado_m": ancho_evaluado,
        "franja_completa_en_ventana": franja_completa,
        "n_puntos": interior["n_puntos"], "n_compatibles": interior["n_compatibles"],
        "porcentaje_compatible": interior["porcentaje_compatible"],
        "rmse_perpendicular_m": interior["rmse_compatibles_m"],
        "n_exterior": exterior["n_puntos"],
        "n_exterior_compatibles": exterior["n_compatibles"],
        "porcentaje_exterior_compatible": exterior["porcentaje_compatible"],
        "n_cuadrantes_con_apoyo": sum(
            medidas["n_compatibles"] > 0 for medidas in entorno["cuadrantes"].values()
        ) if entorno else None,
        "inclinacion_candidata": calcular_inclinacion(ajuste["normal"]) if ajuste else None,
        "centro": ajuste["centro"].tolist() if ajuste else None,
        "normal": ajuste["normal"].tolist() if ajuste else None,
        "interior": interior, "exterior": exterior,
        "apoyo_por_cuadrante": entorno["cuadrantes"] if entorno else {},
        "revision_manual": formulario,
        "estado": "prototipo_pendiente_de_validacion", **decision,
    }
    salida = salida / caso.id_panel
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    tabla = pd.DataFrame(contexto, columns=["x", "y", "z"])
    tabla["zona"] = "contexto"
    tabla.loc[dentro, "zona"] = "interior"
    tabla["distancia_con_signo_m"] = None
    tabla["distancia_m"] = None
    tabla["compatible"] = None
    if entorno is not None:
        tabla.loc[entorno["exterior"], "zona"] = "franja_exterior"
        tabla["distancia_con_signo_m"] = entorno["distancias"]
        tabla["distancia_m"] = np.abs(entorno["distancias"])
        tabla["compatible"] = tabla.distancia_m <= np.where(dentro, umbral, umbral_exterior)
    tabla.to_csv(salida / "puntos.csv", index=False)

    cubierta = resultado["cubierta"]
    puntos_cubierta = cubierta["puntos"] if cubierta else np.empty((0, 3))
    pd.DataFrame(puntos_cubierta, columns=["x", "y", "z"]).to_csv(salida / "cubierta.csv", index=False)

    if guardar_html:
        if ajuste is not None:
            figura = dibujar_ajuste(contexto, dentro, ajuste, entorno, umbral_exterior, cubierta)
        else:
            locales = contexto - contexto.min(axis=0) if len(contexto) else contexto
            figura = go.Figure(go.Scatter3d(
                x=locales[:, 0], y=locales[:, 1], z=locales[:, 2], mode="markers",
                marker={"size": 2}, name="Sin plano interior",
            ))
            figura.update_layout(scene={"aspectmode": "data"})
        figura.update_layout(title=(f"{decision['inclinacion_utilizada']:.2f}° · "
                                    f"{decision['origen']} · {decision['motivo']}"))
        figura.write_html(salida / "ajuste.html")

    print(f"Panel: {caso.id_panel}")
    print(f"Inclinación utilizada: {decision['inclinacion_utilizada']:.2f} grados ({decision['origen']})")
    print(f"Motivo: {decision['motivo']}")
    if ajuste is not None:
        print(f"Compatibles: {resumen['n_compatibles']}/{resumen['n_puntos']}")
        print(f"Apoyo exterior: {resumen['n_exterior_compatibles']}/{resumen['n_exterior']}")

    return resumen


def main() -> None:
    """Procesa uno o varios casos y guarda una tabla común para comparar resultados."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=CARPETA)
    seleccion = parser.add_mutually_exclusive_group()
    seleccion.add_argument("--panel", help="ID concreto; por defecto procesa la primera fila.")
    seleccion.add_argument("--todos", action="store_true", help="Procesa todos los casos elegidos.")
    parser.add_argument("--revisados", action="store_true", help="Selecciona solo formularios no vacíos.")
    parser.add_argument("--umbral", type=float, default=0.10, help="Distancia interior en metros.")
    parser.add_argument("--umbral-exterior", type=float, help="Por defecto utiliza el umbral interior.")
    parser.add_argument("--anillo", type=float, default=2.0, help="Anchura exterior en metros.")
    parser.add_argument("--iteraciones", type=int, default=500)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--porcentaje-minimo", type=float, default=CRITERIOS["porcentaje_minimo"])
    parser.add_argument("--min-puntos", type=int, default=CRITERIOS["min_puntos"])
    parser.add_argument("--dispersion-minima", type=float, default=CRITERIOS["dispersion_minima"])
    parser.add_argument("--inclinacion-maxima", type=float, default=CRITERIOS["inclinacion_maxima"])
    parser.add_argument("--angulo-maximo", type=float, default=CRITERIOS["angulo_maximo"])
    parser.add_argument("--separacion-maxima", type=float, default=CRITERIOS["separacion_maxima"])
    parser.add_argument("--estabilidad-maxima", type=float, default=CRITERIOS["estabilidad_maxima"])
    parser.add_argument("--anillo-maximo", type=float, default=CRITERIOS["anillo_maximo"])
    parser.add_argument("--distancia-vecinos", type=float, default=CRITERIOS["distancia_vecinos"])
    parser.add_argument("--inclinacion-predeterminada", type=float, default=CRITERIOS["inclinacion_predeterminada"])
    parser.add_argument("--sin-html", action="store_true", help="Guarda medidas y puntos sin la figura 3D.")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()
    umbral_exterior = args.umbral if args.umbral_exterior is None else args.umbral_exterior

    if min(args.umbral, umbral_exterior, args.anillo) <= 0 or args.iteraciones < 1:
        parser.error("Los umbrales, el anillo y las iteraciones deben ser positivos.")
    criterios = {nombre: getattr(args, nombre) for nombre in CRITERIOS}
    if not all(np.isfinite(valor) for valor in [args.umbral, umbral_exterior, args.anillo, *criterios.values()]):
        parser.error("Los parámetros deben ser números finitos.")
    if not 0 < args.porcentaje_minimo <= 100 or args.min_puntos < 3:
        parser.error("El porcentaje debe estar entre 0 y 100, y min-puntos debe ser al menos 3.")
    if min(args.dispersion_minima, args.separacion_maxima, args.distancia_vecinos) <= 0:
        parser.error("Los límites espaciales deben ser positivos.")
    if not all(0 < valor < 90 for valor in [args.inclinacion_maxima, args.angulo_maximo, args.estabilidad_maxima]):
        parser.error("Los límites angulares deben estar entre 0 y 90 grados.")
    if not 0 <= args.inclinacion_predeterminada < 90 or args.anillo_maximo < args.anillo:
        parser.error("Revisa la inclinación predeterminada (0 a menos de 90) y anillo-maximo >= anillo.")

    carpeta = ruta_proyecto(args.carpeta)
    salida = ruta_proyecto(args.salida)
    casos = pd.read_csv(carpeta / "casos.csv", keep_default_na=False)

    if args.revisados:
        casos = casos[casos.formulario.ne("")]

    if not args.todos:
        try:
            casos = pd.DataFrame([elegir_caso(casos, args.panel)])
        except ValueError as error:
            parser.error(str(error))

    if casos.empty:
        parser.error("No hay casos para procesar.")

    salida.mkdir(parents=True, exist_ok=True)
    resultados = []
    errores = []

    for _, caso in casos.iterrows():
        try:
            resumen = procesar_caso(
                caso, carpeta, salida, args.umbral, umbral_exterior,
                args.anillo, args.iteraciones, args.semilla, not args.sin_html, criterios,
            )
        except (ValueError, OSError, laspy.errors.LaspyException) as error:
            errores.append({"id_panel": caso.id_panel, "error": str(error)})
            print(f"Sin ajuste: {caso.id_panel}: {error}")
            continue

        anidados = {"centro", "normal", "calidad_interior", "estabilidad_interior", "cubierta"}
        fila = {clave: valor for clave, valor in resumen.items()
                if clave not in anidados and not isinstance(valor, (dict, list))}

        for zona in ["interior", "exterior"]:
            fila.update({f"{zona}_{clave}": valor for clave, valor in resumen[zona].items()})

        fila.update({f"criterio_{clave}": valor for clave, valor in criterios.items()})
        for nombre in ["calidad_interior", "estabilidad_interior"]:
            fila.update({f"{nombre}_{clave}": valor for clave, valor in (resumen[nombre] or {}).items()})
        if resumen["cubierta"]:
            fila.update({clave: resumen["cubierta"][clave]
                         for clave in ["angulo_entre_planos_grados", "separacion_p90_m"]})

        fila.update({
            f"manual_{campo}": resumen["revision_manual"].get(campo, "")
            for campo in ["montaje", "coincidencia", "confianza"]
        })
        resultados.append(fila)

    pd.DataFrame(resultados).to_csv(salida / "resumen.csv", index=False)
    pd.DataFrame(errores, columns=["id_panel", "error"]).to_csv(salida / "errores.csv", index=False)
    estimados = sum(fila["origen"] == "estimada" for fila in resultados)
    print(f"Casos procesados: {len(resultados)}/{len(casos)}; estimados: {estimados}. Resultados: {salida}")

    if errores:
        raise SystemExit("Algunos casos no tienen ajuste. Consulta errores.csv.")


if __name__ == "__main__":
    main()
