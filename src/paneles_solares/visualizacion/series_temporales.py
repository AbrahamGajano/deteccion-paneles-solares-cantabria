"""Calcula producción horaria por instalación desde los factores del inventario."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

from paneles_solares.rutas import (
    CACHE_PVGIS_HORARIO,
    VISUALIZACION,
    guardar_json_atomico,
    huella_archivo,
    ruta_proyecto,
    ruta_registrada,
)
from paneles_solares.visualizacion.estimar_energia import (
    BASE_RADIACION,
    ESCENARIOS_HISTORICOS,
    MONTAJE,
    TECNOLOGIA,
    URL_PVGIS,
    USAR_HORIZONTE,
    solicitar_pvgis,
)


def parametros_horarios(fila, anio_meteo: int) -> dict:
    """Construye una consulta de un año para una referencia de 1 kWp sin pérdidas.

    Params:
        fila: Posición municipal, inclinación de consulta y orientación PVGIS.
        anio_meteo: Año meteorológico, independiente de la edad de los equipos.

    Returns:
        dict: Parámetros del servicio seriescalc.
    """

    return {
        "lat": round(float(fila.latitud), 6),
        "lon": round(float(fila.longitud), 6),
        "startyear": anio_meteo,
        "endyear": anio_meteo,
        "raddatabase": BASE_RADIACION,
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": 0,
        "angle": int(fila.inclinacion_pvgis_grados),
        "aspect": int(fila.orientacion_grados),
        "trackingtype": 0,
        "pvtechchoice": TECNOLOGIA,
        "mountingplace": MONTAJE,
        "usehorizon": USAR_HORIZONTE,
        "outputformat": "json",
    }


def normalizar_serie_horaria(respuesta: dict, anio_meteo: int) -> pd.Series:
    """Convierte la potencia de referencia a kW, etiquetando cada hora en UTC.

    Params:
        respuesta: JSON de seriescalc con outputs.hourly, time y P en W.
        anio_meteo: Año completo esperado, incluidos los años bisiestos.

    Returns:
        pd.Series: Potencia de 1 kWp en kW, indexada por hora UTC. Se asigna la
            muestra al comienzo de su hora para alinear municipios; la marca
            original se conserva en el JSON de la caché. No se interpolan huecos.
    """

    datos = pd.DataFrame(respuesta["outputs"]["hourly"])
    fechas = pd.to_datetime(datos.time, format="%Y%m%d:%H%M", utc=True).dt.floor("h")
    serie = pd.Series(datos.P.to_numpy(dtype=float) / 1000, index=pd.DatetimeIndex(fechas))
    serie = serie.sort_index()
    serie.index.name = "fecha_utc"
    serie.name = "potencia_referencia_kw"
    esperadas = pd.date_range(
        f"{anio_meteo}-01-01", f"{anio_meteo + 1}-01-01",
        freq="h", inclusive="left", tz="UTC", name="fecha_utc",
    )
    if not serie.index.equals(esperadas):
        raise ValueError("PVGIS no devolvió un año horario completo y sin duplicados")
    if not np.isfinite(serie).all() or serie.lt(0).any():
        raise ValueError("PVGIS devolvió potencias ausentes, negativas o no finitas")

    return serie


def obtener_serie_horaria(sesion, parametros: dict, carpeta: Path) -> tuple[Path, pd.Series]:
    """Descarga o reutiliza una referencia y conserva su respuesta comprimida.

    Params:
        sesion: Sesión HTTP compartida.
        parametros: Consulta completa, que identifica la referencia en la caché.
        carpeta: Directorio compartido de consultas.json y respuestas JSON.gz.

    Returns:
        tuple: Ruta de la respuesta y potencia horaria normalizada. Cada consulta
            terminada se registra antes de continuar; los errores se propagan.
    """

    carpeta.mkdir(parents=True, exist_ok=True)
    indice = carpeta / "consultas.json"
    consultas = json.loads(indice.read_text(encoding="utf-8")) if indice.exists() else {}
    clave = json.dumps([URL_PVGIS, "seriescalc", parametros], sort_keys=True)

    if clave in consultas:
        archivo = carpeta / consultas[clave]
        with gzip.open(archivo, "rt", encoding="utf-8") as entrada:
            respuesta = json.load(entrada)
        return archivo, normalizar_serie_horaria(respuesta, parametros["startyear"])

    respuesta = solicitar_pvgis(sesion, "seriescalc", parametros)
    serie = normalizar_serie_horaria(respuesta, parametros["startyear"])
    archivo = carpeta / f"referencia_{len(consultas) + 1:05d}.json.gz"
    with gzip.open(archivo, "wt", encoding="utf-8") as salida:
        json.dump(respuesta, salida, ensure_ascii=False)
    consultas[clave] = archivo.name
    guardar_json_atomico(indice, consultas, indent=2)

    return archivo, serie


def preparar_relaciones(energia: pd.DataFrame, anio_inventario: int, escenarios) -> pd.DataFrame:
    """Relaciona las piezas del inventario con sus curvas y factores de escala.

    Params:
        energia: Detalle de energia_paneles.csv, con superficies ya repartidas.
        anio_inventario: Campaña cuyos equipos y edades se representan.
        escenarios: Nombres de los escenarios históricos que se calcularán.

    Returns:
        pd.DataFrame: Una fila por pieza y escenario. escala_referencia incluye
            superficie, eficiencia inicial, inclinación, pérdidas y degradación;
            no equivale a la potencia nominal de la instalación.
    """

    inventario = energia[energia.anio.eq(anio_inventario)].copy()
    if inventario.empty:
        raise ValueError(f"No hay detecciones con superficie en el inventario de {anio_inventario}")
    columnas = [
        "id_panel", "id_instalacion", "anio", "codigo", "latitud", "longitud",
        "area_m2", "origen_inclinacion", "inclinacion_estimada", "antiguedad_supuesta",
    ]
    columnas += [c for c in ("azimut_lidar_grados", "origen_orientacion") if c in inventario]
    filas = []
    for nombre in escenarios:
        parte = inventario[columnas].copy()
        parte["escenario"] = nombre
        parte["inclinacion_grados"] = inventario[f"inclinacion_{nombre}_grados"]
        parte["inclinacion_pvgis_grados"] = inventario[f"inclinacion_pvgis_{nombre}_grados"]
        supuesto = ESCENARIOS_HISTORICOS[nombre]["orientacion_grados"]
        parte["orientacion_exacta_grados"] = inventario.get(f"orientacion_{nombre}_grados", supuesto)
        parte["orientacion_grados"] = inventario.get(f"orientacion_pvgis_{nombre}_grados", supuesto)
        parte["escala_referencia"] = inventario.area_m2 * inventario[f"factor_{nombre}"]
        filas.append(parte)

    return pd.concat(filas, ignore_index=True)


def preparar_referencias(relaciones, anio_meteo: int, cache: Path) -> pd.DataFrame:
    """Obtiene solo las curvas necesarias, compartidas entre instalaciones.

    Params:
        relaciones: Piezas y escenarios generados por preparar_relaciones.
        anio_meteo: Año de la meteorología consultada.
        cache: Carpeta de referencias horarias compartidas.

    Returns:
        pd.DataFrame: Consultas únicas con su ruta y doce rendimientos mensuales
            aproximados, en kWh por kWp de referencia y sin pérdidas adicionales.
    """

    claves = ["codigo", "latitud", "longitud", "inclinacion_pvgis_grados", "orientacion_grados"]
    consultas = relaciones[claves].drop_duplicates().sort_values(claves)
    filas = []
    with requests.Session() as sesion:
        for i, fila in enumerate(consultas.itertuples(index=False), 1):
            print(f"PVGIS horario: {i}/{len(consultas)} · {fila.codigo} · "
                  f"{fila.inclinacion_pvgis_grados}° · orientación {fila.orientacion_grados}°", flush=True)
            parametros = parametros_horarios(fila, anio_meteo)
            archivo, serie = obtener_serie_horaria(sesion, parametros, cache)
            mensual = serie.groupby(serie.index.month).sum()
            registro = fila._asdict()
            registro["referencia"] = ruta_registrada(archivo)
            registro["anio_meteo"] = anio_meteo
            registro["n_horas"] = len(serie)
            registro.update({f"rendimiento_mes_{m:02d}": float(mensual[m]) for m in range(1, 13)})
            filas.append(registro)

    return pd.DataFrame(filas)


def resumir_instalaciones(relaciones, referencias) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Suma energía mensual y anual sin materializar todas las horas de cada equipo.

    Params:
        relaciones: Piezas con referencia asignada y escala_referencia.
        referencias: Tabla de rendimientos mensuales de las referencias.

    Returns:
        tuple: Tablas mensual y anual por instalación y escenario, en kWh.
            No se suman instalaciones como un total territorial sin revisar
            antes los posibles solapes entre agrupaciones distintas.
    """

    rendimientos = referencias.drop_duplicates("referencia").set_index("referencia")
    columnas = ["id_instalacion", "anio", "escenario"]
    filas = []
    for mes in range(1, 13):
        parte = relaciones[columnas].copy()
        parte["mes"] = mes
        parte["energia_kwh"] = relaciones.escala_referencia * relaciones.referencia.map(
            rendimientos[f"rendimiento_mes_{mes:02d}"]
        )
        filas.append(parte)
    mensual = pd.concat(filas).groupby(columnas + ["mes"], as_index=False).energia_kwh.sum()
    anual = mensual.groupby(columnas, as_index=False).energia_kwh.sum()

    return mensual, anual


def calcular_serie_instalacion(relaciones: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye las curvas de una instalación sumando sus piezas y pendientes.

    Params:
        relaciones: Filas de una única instalación y campaña con sus referencias.

    Returns:
        pd.DataFrame: Potencia en kW y energía horaria aproximada en kWh para cada
            escenario. Las fechas son horas UTC; cada muestra representa una hora.
    """

    curvas = pd.DataFrame()
    grupos = relaciones.groupby(["referencia", "escenario"]).escala_referencia.sum()
    for (referencia, escenario), escala in grupos.items():
        with gzip.open(ruta_proyecto(referencia), "rt", encoding="utf-8") as entrada:
            respuesta = json.load(entrada)
        anio = int(relaciones.anio_meteo.iloc[0])
        serie = normalizar_serie_horaria(respuesta, anio) * escala
        columna = f"potencia_{escenario}_kw"
        if columna in curvas:
            curvas[columna] = curvas[columna] + serie
        else:
            curvas[columna] = serie
    for nombre in relaciones.escenario.unique():
        curvas[f"energia_{nombre}_kwh"] = curvas[f"potencia_{nombre}_kw"] * 1.

    return curvas.sort_index()


def crear_figura_produccion(serie: pd.DataFrame) -> go.Figure:
    """Dibuja las potencias en horario español sin modificar las fechas de cálculo.

    Params:
        serie: Curvas horarias UTC de una instalación.

    Returns:
        go.Figure: Gráfico interactivo de potencia por escenario.
    """

    figura = go.Figure()
    fechas = serie.index.tz_convert("Europe/Madrid")
    for columna in serie.filter(like="potencia_"):
        figura.add_scatter(x=fechas, y=serie[columna], name=columna, mode="lines")
    figura.update_layout(
        title="Producción fotovoltaica modelada", xaxis_title="Fecha · Europe/Madrid",
        yaxis_title="Potencia (kW)", xaxis_rangeslider_visible=True,
    )

    return figura


def preparar_series_temporales(
    energia, salida: Path, anio_inventario=2023, anio_meteo=2023,
    escenarios=("min", "central", "max"), cache=CACHE_PVGIS_HORARIO,
    exportar_horarias=False,
) -> pd.DataFrame:
    """Genera relaciones, referencias y resúmenes reutilizables del inventario.

    Params:
        energia: Detalle energético por detección y municipio.
        salida: Carpeta de esta ejecución, separada del cálculo anual medio.
        anio_inventario: Año que determina geometrías, eficiencia y edad.
        anio_meteo: Año del recurso meteorológico; no cambia la edad del inventario.
        escenarios: Escenarios históricos que se evaluarán.
        cache: Referencias compartidas, identificadas por todos los parámetros.
        exportar_horarias: Guarda un CSV.gz por instalación, una cada vez.

    Returns:
        pd.DataFrame: Relaciones necesarias para reconstruir cualquier instalación.
    """

    salida.mkdir(parents=True, exist_ok=True)
    relaciones = preparar_relaciones(energia, anio_inventario, escenarios)
    referencias = preparar_referencias(relaciones, anio_meteo, cache)
    claves = ["codigo", "latitud", "longitud", "inclinacion_pvgis_grados", "orientacion_grados"]
    relaciones = relaciones.merge(
        referencias[claves + ["referencia", "anio_meteo"]], on=claves, validate="many_to_one",
    )
    mensual, anual = resumir_instalaciones(relaciones, referencias)
    relaciones.to_csv(salida / "relaciones.csv", index=False)
    referencias.to_csv(salida / "referencias.csv", index=False)
    mensual.to_csv(salida / "mensual_instalaciones.csv", index=False)
    anual.to_csv(salida / "anual_instalaciones.csv", index=False)
    configuracion = {
        "anio_inventario": anio_inventario, "anio_meteo": anio_meteo,
        "escenarios": list(escenarios), "url_pvgis": URL_PVGIS,
        "base_radiacion": BASE_RADIACION, "potencia_referencia_kwp": 1, "loss": 0,
        "n_instalaciones": int(relaciones.id_instalacion.nunique()),
        "n_referencias": int(referencias.referencia.nunique()),
        "tiempo": "UTC. La muestra se etiqueta al inicio de su hora; la marca original queda en la caché.",
        "energia": "Potencia modelada por 1 hora. Energía aproximada; no lectura de contador.",
        "factores": "Se reutilizan area_m2 y factor del inventario: eficiencia, inclinación, pérdidas y degradación se aplican una sola vez. Cambiar meteorología no envejece los equipos.",
        "alcance": "Equipos del inventario funcionando un año completo; no reconstruye altas, averías ni producción real. Resúmenes por instalación, sin extrapolación territorial.",
        "meses": "Meses en UTC. La visualización convierte a Europe/Madrid.",
    }
    (salida / "configuracion.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if exportar_horarias:
        carpeta = salida / "instalaciones"
        carpeta.mkdir(exist_ok=True)
        grupos = relaciones.groupby("id_instalacion")
        for i, (identificador, partes) in enumerate(grupos, 1):
            print(f"Exportando horas: {i}/{len(grupos)} · {identificador}", flush=True)
            calcular_serie_instalacion(partes).to_csv(carpeta / f"{identificador}.csv.gz")

    return relaciones


def registrar_procedencia(carpeta: Path, salida: Path):
    """Registra el detalle energético de entrada y sus hipótesis de cálculo.

    Params:
        carpeta: Directorio de energia_paneles.csv e hipotesis_energia.json.
        salida: Directorio de las series temporales.

    Returns:
        None.
    """

    archivo = carpeta / "energia_paneles.csv"
    procedencia = {"entrada": ruta_registrada(archivo), "sha256": huella_archivo(archivo)}
    hipotesis = carpeta / "hipotesis_energia.json"
    if hipotesis.exists():
        procedencia["hipotesis_inventario"] = json.loads(hipotesis.read_text(encoding="utf-8"))
    (salida / "procedencia.json").write_text(
        json.dumps(procedencia, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main():
    """Calcula o exporta las series desde el detalle energético ya preparado."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=ruta_proyecto, default=VISUALIZACION)
    parser.add_argument("--salida", type=ruta_proyecto)
    parser.add_argument("--cache", type=ruta_proyecto, default=CACHE_PVGIS_HORARIO)
    parser.add_argument("--anio-inventario", type=int, default=2023)
    parser.add_argument("--anio-meteo", type=int, default=2023)
    parser.add_argument("--escenarios", nargs="+", choices=list(ESCENARIOS_HISTORICOS),
                        default=list(ESCENARIOS_HISTORICOS))
    parser.add_argument("--instalacion", help="Calcula una instalación y guarda su curva y gráfico")
    parser.add_argument("--exportar-horarias", action="store_true")
    args = parser.parse_args()
    if args.instalacion and args.salida is None:
        parser.error("Una instalación necesita --salida para conservar el resultado completo")
    archivo = args.carpeta / "energia_paneles.csv"
    energia = pd.read_csv(archivo, dtype={"codigo": str})
    if args.instalacion:
        energia = energia[energia.id_instalacion.eq(args.instalacion)]
    salida = args.salida or args.carpeta / "series_temporales" / (
        f"inventario_{args.anio_inventario}_meteo_{args.anio_meteo}"
    )
    relaciones = preparar_series_temporales(
        energia, salida, args.anio_inventario, args.anio_meteo, args.escenarios,
        args.cache, args.exportar_horarias or bool(args.instalacion),
    )
    registrar_procedencia(args.carpeta, salida)
    if args.instalacion:
        crear_figura_produccion(calcular_serie_instalacion(relaciones)).write_html(
            salida / "produccion.html", include_plotlyjs=True,
        )
    print(salida)


if __name__ == "__main__":
    main()
