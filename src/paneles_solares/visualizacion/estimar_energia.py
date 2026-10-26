"""Estima potencia pico y producción desde los indicadores U-Net de Cantabria."""

import json
import os
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import requests

from paneles_solares.rutas import (
    INDICADORES_UNET,
    PRODUCCION_CONFIG,
    PRODUCCION_MENSUAL,
    PRODUCCION_MUNICIPAL,
    version_indicadores,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARPETA = INDICADORES_UNET
EDIFICIOS = CARPETA / "edificios.gpkg"
MUNICIPIOS = CARPETA / "municipios.gpkg"
CACHE = CARPETA / "pvgis_municipios.jsonl"
SALIDA_GRAFICO = CARPETA / "resumen_produccion.png"

URL_PVGIS = "https://re.jrc.ec.europa.eu/api/v5_3"
VERSION_PVGIS = "5.3"
BASE_RADIACION = "PVGIS-SARAH3"
INCLINACION = 30
ORIENTACION = 0  # En PVGIS, 0 grados significa sur.
PERDIDAS = 14
POTENCIA_REFERENCIA_KWP = 1
TECNOLOGIA = "crystSi"
MONTAJE = "free"
USAR_HORIZONTE = 1
ESCENARIOS = {
    "conservador": ("superficie_fv_detectada_total_m2", 0.17),
    "central": ("superficie_fv_detectada_total_m2", 0.20),
    "superior": ("superficie_fv_detectada_total_m2", 0.23),
}

# Escenarios del inventario histórico; no son intervalos de confianza.
ESCENARIOS_HISTORICOS = {
    "min": {
        "inclinacion_grados": 15,
        "orientacion_grados": 90,
        "perdidas_operativas_pct": 18,
        "degradacion_anual": 0.01,
        "factor_eficiencia": 0.85,
    },
    "central": {
        "inclinacion_grados": 30,
        "orientacion_grados": 0,
        "perdidas_operativas_pct": 10,
        "degradacion_anual": 0.005,
        "factor_eficiencia": 1.0,
    },
    "max": {
        "inclinacion_grados": 40,
        "orientacion_grados": 0,
        "perdidas_operativas_pct": 5,
        "degradacion_anual": 0.002,
        "factor_eficiencia": 1.15,
    },
}
EFICIENCIA_POR_ANIO = {
    1989: 0.10,
    2000: 0.12,
    2010: 0.15,
    2014: 0.163,
    2015: 0.170,
    2016: 0.175,
    2017: 0.177,
    2018: 0.184,
    2019: 0.192,
    2020: 0.200,
    2021: 0.209,
    2022: 0.212,
    2023: 0.218,
}
FUENTES_HISTORICAS = {
    "PVGIS": "https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en",
    "Eficiencia 2014–2023 (JRC, tabla 2)": "https://publications.jrc.ec.europa.eu/repository/bitstream/JRC144141/JRC144141_01.pdf",
    "Degradación (NREL)": "https://docs.nrel.gov/docs/fy12osti/53712.pdf",
}
ANIOS_ANTIGUEDAD_SUPUESTA = 25
TOTALES_ENERGIA = [
    "produccion_min_mwh",
    "produccion_central_mwh",
    "produccion_max_mwh",
    "potencia_central_kwp",
    "superficie_antiguedad_supuesta_m2",
]


def estimar_cohorte(
    superficie_modulos_m2: float,
    densidad_inicial_kwp_m2: float,
    rendimiento_sin_perdidas_kwh_kwp: float,
    anio_calculo: int,
    instalacion_hasta: int,
    instalacion_desde: int | None = None,
    degradacion_anual: float = 0.005,
    perdidas_operativas_pct: float = 10,
) -> dict:
    """Estima producción de módulos de una misma época, conservando su incertidumbre.

    Args:
        superficie_modulos_m2: Superficie inclinada de módulos, no de tejado o parcela.
        densidad_inicial_kwp_m2: Potencia por m² al instalarse; depende de la tecnología.
        rendimiento_sin_perdidas_kwh_kwp: Rendimiento PVGIS consultado con loss=0.
        anio_calculo: Año para el que se estima la producción anual esperada.
        instalacion_hasta: Primera presencia, límite superior del periodo de instalación.
        instalacion_desde: Última ausencia atribuida a estos módulos. None significa
            que ya estaban en la primera campaña, sin antigüedad máxima conocida.
        degradacion_anual: Pérdida fraccional por año; 0,5 % es una hipótesis de partida.
        perdidas_operativas_pct: Inversor, cableado y otras pérdidas, sin envejecimiento.
            El 10 % es un escenario provisional, no una medición de la instalación.

    Returns:
        dict: Potencia inicial y producción mínima, central y máxima. Sin límite
            inferior de instalación no se inventan antigüedad máxima ni valor central.

    Raises:
        ValueError: El periodo es posterior al cálculo o las hipótesis son imposibles.
    """

    if anio_calculo < instalacion_hasta or (
        instalacion_desde is not None and instalacion_desde > instalacion_hasta
    ):
        raise ValueError("Revisa el periodo de instalación y el año de cálculo")
    if min(
        superficie_modulos_m2, densidad_inicial_kwp_m2, rendimiento_sin_perdidas_kwh_kwp
    ) < 0 or not (0 <= degradacion_anual < 1 and 0 <= perdidas_operativas_pct < 100):
        raise ValueError("Superficies, potencias, rendimientos y pérdidas no válidos")
    potencia = superficie_modulos_m2 * densidad_inicial_kwp_m2
    base = (
        potencia
        * rendimiento_sin_perdidas_kwh_kwp
        * (1 - perdidas_operativas_pct / 100)
    )
    edad_minima = anio_calculo - instalacion_hasta
    edad_maxima = (
        None if instalacion_desde is None else anio_calculo - instalacion_desde
    )
    return {
        "potencia_inicial_kwp": potencia,
        "edad_minima_anios": edad_minima,
        "edad_maxima_anios": edad_maxima,
        "produccion_max_kwh": base * (1 - degradacion_anual) ** edad_minima,
        "produccion_min_kwh": None
        if edad_maxima is None
        else base * (1 - degradacion_anual) ** edad_maxima,
        "produccion_central_kwh": None
        if edad_maxima is None
        else base * (1 - degradacion_anual) ** ((edad_minima + edad_maxima) / 2),
    }


def hipotesis_historicas() -> dict:
    """Documenta los supuestos comunes a las tablas y al panel histórico."""

    return {
        "alcance": "Producción anual representativa de los paneles observados; no consumo ni total de Cantabria.",
        "radiacion": f"PVGIS {VERSION_PVGIS}, {BASE_RADIACION}; clima medio, silicio cristalino, montaje ventilado, 1 kWp y loss=0. Un punto municipal ponderado por superficie; horizonte de terreno incluido.",
        "escenarios": {
            nombre: parametros.copy()
            for nombre, parametros in ESCENARIOS_HISTORICOS.items()
        },
        "eficiencia_por_anio": EFICIENCIA_POR_ANIO,
        "eficiencia": "Tabla JRC 2014–2023 e interpolación lineal. Antes de 2014: hipótesis de 10 % en 1989, 12 % en 2000 y 15 % en 2010. No identifica la tecnología de cada placa.",
        "superficie": "Área de módulos = proyección horizontal / cos(inclinación). Los solapes se disuelven y su reparto energético se pondera por área.",
        "antiguedad": "Central: punto medio del intervalo de aparición del emplazamiento, como aproximación a la edad de los módulos. Sin límite inferior compatible: desde 25 años antes de la primera geometría, como hipótesis explícita.",
        "ampliaciones": "No se fechan módulos ni se interpretan diferencias del modelo como ampliaciones. El escenario favorable permite que toda el área sea de módulos nuevos del año de la imagen; el conservador usa la fecha más antigua supuesta. El central puede infravalorar renovaciones y ampliaciones recientes.",
        "perdidas": "Operativas y degradación se aplican una sola vez, fuera de PVGIS. No se suma el 14 % del escenario regional uniforme.",
        "limites": "Escenarios de sensibilidad, no intervalos de confianza. No cubren falsos positivos, omisiones, pendientes reales, sombras locales, averías ni indisponibilidad. Las áreas del modelo de 2023 siguen siendo estimadas.",
        "periodo": "Un año completo representativo con los equipos de cada imagen; no la energía realmente generada durante el año del vuelo ni una predicción meteorológica.",
        "fuentes": FUENTES_HISTORICAS,
    }


def factores_historicos(candidatos, paneles) -> pd.DataFrame:
    """Calcula potencia y envejecimiento por m² horizontal, sin consultar radiación.

    La edad central aproxima la época del emplazamiento. Las renovaciones no
    observadas quedan en los escenarios; no se asignan fechas medidas a módulos.
    """

    primeras = paneles.groupby("id_instalacion").anio.min()
    periodos = candidatos.set_index("id_instalacion")
    filas = []
    for identificador, anio in (
        paneles[["id_instalacion", "anio"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        periodo = periodos.loc[identificador]
        hasta = min(float(primeras[identificador]), anio)
        if pd.notna(periodo.instalacion_hasta):
            hasta = min(hasta, float(periodo.instalacion_hasta))
        desde = periodo.instalacion_desde
        supuesta = (
            pd.isna(desde)
            or desde >= hasta
            or periodo.estado_periodo == "revisar_cambio_o_contradiccion"
        )
        if supuesta:
            desde = hasta - ANIOS_ANTIGUEDAD_SUPUESTA
        fila = {
            "id_instalacion": identificador,
            "anio": anio,
            "fecha_desde_modelada": desde,
            "fecha_hasta_modelada": hasta,
            "antiguedad_supuesta": bool(supuesta),
        }
        fechas = {"min": desde, "central": (desde + hasta) / 2, "max": anio}
        for nombre, escenario in ESCENARIOS_HISTORICOS.items():
            fecha = fechas[nombre]
            densidad = (
                float(
                    np.interp(
                        fecha,
                        list(EFICIENCIA_POR_ANIO),
                        list(EFICIENCIA_POR_ANIO.values()),
                    )
                )
                * escenario["factor_eficiencia"]
            )
            estimacion = estimar_cohorte(
                1 / np.cos(np.radians(escenario["inclinacion_grados"])),
                densidad,
                1,
                anio,
                fecha,
                fecha,
                degradacion_anual=escenario["degradacion_anual"],
                perdidas_operativas_pct=escenario["perdidas_operativas_pct"],
            )
            fila[f"factor_{nombre}"] = estimacion["produccion_central_kwh"]
            if nombre == "central":
                fila["potencia_por_m2_kwp"] = estimacion["potencia_inicial_kwp"]
        filas.append(fila)
    return pd.DataFrame(filas)


def rendimientos_historicos(recortes, carpeta: Path) -> pd.DataFrame:
    """Consulta tres geometrías de montaje por municipio y conserva su respuesta."""

    cache = carpeta / "pvgis_anual.json"
    consultas = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    filas = []
    with requests.Session() as sesion:
        grupos = list(recortes.groupby("codigo"))
        for i, (codigo, grupo) in enumerate(grupos, 1):
            puntos = grupo.geometry.centroid
            pesos = grupo.geometry.area
            punto = (
                gpd.GeoSeries(
                    gpd.points_from_xy(
                        [np.average(puntos.x, weights=pesos)],
                        [np.average(puntos.y, weights=pesos)],
                    ),
                    crs=recortes.crs,
                )
                .to_crs(4326)
                .iloc[0]
            )
            fila = {"codigo": codigo, "latitud": punto.y, "longitud": punto.x}
            print(f"PVGIS anual: {codigo} ({i}/{len(grupos)})", flush=True)
            for nombre, escenario in ESCENARIOS_HISTORICOS.items():
                parametros = {
                    "lat": round(punto.y, 6),
                    "lon": round(punto.x, 6),
                    "raddatabase": BASE_RADIACION,
                    "peakpower": 1,
                    "loss": 0,
                    "angle": escenario["inclinacion_grados"],
                    "aspect": escenario["orientacion_grados"],
                    "pvtechchoice": TECNOLOGIA,
                    "mountingplace": MONTAJE,
                    "usehorizon": 1,
                    "outputformat": "json",
                }
                clave = json.dumps([URL_PVGIS, parametros], sort_keys=True)
                if clave not in consultas:
                    consultas[clave] = solicitar_pvgis(sesion, "PVcalc", parametros)
                    carpeta.mkdir(parents=True, exist_ok=True)
                    temporal = cache.with_suffix(".tmp")
                    temporal.write_text(
                        json.dumps(consultas, ensure_ascii=False), encoding="utf-8"
                    )
                    temporal.replace(cache)
                fila[f"rendimiento_{nombre}"] = consultas[clave]["outputs"]["totals"][
                    "fixed"
                ]["E_y"]
            filas.append(fila)
    return pd.DataFrame(filas)


def preparar_energia_historica(candidatos, paneles, recortes, carpeta: Path):
    """Prepara áreas por emplazamiento y municipio para reutilizar los desgloses."""

    carpeta.mkdir(parents=True, exist_ok=True)
    factores = factores_historicos(candidatos, paneles)
    factores.to_csv(carpeta / "factores_energia.csv", index=False)
    rendimiento = rendimientos_historicos(recortes, carpeta)
    areas = recortes.dissolve(by=["id_instalacion", "anio", "codigo"])
    areas["area_m2"] = areas.geometry.area
    tabla = (
        areas.drop(columns="geometry")
        .reset_index()
        .merge(factores, on=["id_instalacion", "anio"])
        .merge(rendimiento, on="codigo")
    )
    for nombre in ESCENARIOS_HISTORICOS:
        tabla[f"produccion_{nombre}_mwh"] = (
            tabla.area_m2
            * tabla[f"factor_{nombre}"]
            * tabla[f"rendimiento_{nombre}"]
            / 1000
        )
    tabla["potencia_central_kwp"] = tabla.area_m2 * tabla.potencia_por_m2_kwp
    tabla["superficie_antiguedad_supuesta_m2"] = tabla.area_m2.where(
        tabla.antiguedad_supuesta, 0
    )
    return tabla


def energia_union(superficie_m2, partes: pd.DataFrame) -> dict:
    """Escala la mezcla de épocas al área disuelta, sin duplicar energía por solapes."""

    if superficie_m2 is None:
        return dict.fromkeys(TOTALES_ENERGIA)
    total = partes.area_m2.sum()
    factor = superficie_m2 / total if total else 0
    return {
        columna: float(partes[columna].sum() * factor) for columna in TOTALES_ENERGIA
    }


def configuracion() -> dict:
    """Reúne todas las hipótesis y la versión de los indicadores de entrada.

    Returns:
        dict: Hipótesis energéticas y versión de los GeoPackage utilizados.
    """

    return {
        "url_pvgis": URL_PVGIS,
        "version_pvgis": VERSION_PVGIS,
        "base_radiacion": BASE_RADIACION,
        "inclinacion_grados": INCLINACION,
        "orientacion_pvgis_grados": ORIENTACION,
        "perdidas_pct": PERDIDAS,
        "tratamiento_antiguedad": "Escenario uniforme del 14 %, con envejecimiento medio implícito; todavía sin periodos individuales. No añadir otra degradación a estos resultados.",
        "potencia_referencia_kwp": POTENCIA_REFERENCIA_KWP,
        "tecnologia": TECNOLOGIA,
        "montaje": MONTAJE,
        "usar_horizonte_pvgis": bool(USAR_HORIZONTE),
        "calibracion_test_superficie": "No aplicada: pendiente validar cubiertas y exteriores en el marco territorial de 2023",
        "densidades_kwp_m2": {
            clave: densidad for clave, (_, densidad) in ESCENARIOS.items()
        },
        "superficies_escenarios": {
            clave: campo for clave, (campo, _) in ESCENARIOS.items()
        },
        "indicadores_entrada": version_indicadores(),
        "descripcion": (
            "Potencia pico estimada sobre superficie planimétrica; orientación, "
            "inclinación y pérdidas homogéneas; medias mensuales y anuales de PVGIS."
        ),
    }


def cargar_entradas() -> tuple[gpd.GeoDataFrame, dict[str, tuple[float, float]]]:
    """Comprueba recuentos y calcula posiciones usando todas las máscaras.

    Returns:
        tuple[gpd.GeoDataFrame, dict[str, tuple[float, float]]]: Municipios y posiciones
            de consulta ponderadas por superficie detectada.

    Raises:
        ValueError: Los GeoPackage son incompatibles o sus superficies y recuentos no
            coinciden.
    """

    version_indicadores()
    municipios = gpd.read_file(MUNICIPIOS)
    edificios = gpd.read_file(
        EDIFICIOS,
        where="presencia_fotovoltaica = 1",
        columns=["codigo_municipal", "superficie_fv_detectada_m2", "geometry"],
    )
    if (
        municipios.crs is None
        or edificios.crs is None
        or municipios.crs != edificios.crs
    ):
        raise ValueError("Los dos GeoPackage necesitan el mismo CRS proyectado")
    if municipios.crs.to_epsg() != 25830:
        raise ValueError(
            "Se espera EPSG:25830 para las superficies y la posición ponderada"
        )
    municipios["codigo"] = municipios.codigo.astype(str)
    edificios["codigo_municipal"] = edificios.codigo_municipal.astype(str)
    if municipios.codigo.duplicated().any():
        raise ValueError("Los municipios tienen códigos duplicados")
    detecciones = pd.read_csv(
        CARPETA / "detecciones_municipios.csv", dtype={"codigo": str}
    )
    superficies = detecciones.groupby("codigo").area_m2.sum()
    recuentos = edificios.groupby("codigo_municipal").size()
    if not np.allclose(
        municipios.superficie_fv_detectada_total_m2.to_numpy(dtype=float),
        municipios.codigo.map(superficies).fillna(0).to_numpy(dtype=float),
        atol=0.01,
    ) or not np.array_equal(
        municipios.edificios_con_deteccion.to_numpy(dtype=int),
        municipios.codigo.map(recuentos).fillna(0).to_numpy(dtype=int),
    ):
        raise ValueError(
            "Las detecciones o recuentos de cubiertas no coinciden con el resumen municipal"
        )
    posiciones = {}
    for codigo, grupo in detecciones.groupby("codigo"):
        pesos = grupo.area_m2.to_numpy(dtype=float)
        if np.any(pesos < 0) or pesos.sum() <= 0:
            raise ValueError(f"Superficies detectadas inválidas en {codigo}")
        x = np.average(grupo.x, weights=pesos)
        y = np.average(grupo.y, weights=pesos)
        punto = (
            gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=edificios.crs)
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        posiciones[codigo] = (float(punto.y), float(punto.x))
    return municipios, posiciones


def preparar_cache(config: dict) -> dict[str, dict]:
    """Abre una única caché fija y descarta resultados de otra configuración.

    Args:
        config: Hipótesis energéticas y versión de los indicadores utilizados.

    Returns:
        dict[str, dict]: Consultas completas recuperadas por código municipal.

    Raises:
        ValueError: Falta la caché de resultados existentes o contiene una línea dañada.
    """

    anterior = (
        json.loads(PRODUCCION_CONFIG.read_text(encoding="utf-8"))
        if PRODUCCION_CONFIG.is_file()
        else None
    )
    if anterior != config:
        for ruta in (
            CACHE,
            PRODUCCION_MUNICIPAL,
            PRODUCCION_MENSUAL,
            (CARPETA / "produccion_horaria_representativa.csv"),
            SALIDA_GRAFICO,
        ):
            ruta.unlink(missing_ok=True)
        PRODUCCION_CONFIG.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    consultas = {}
    if CACHE.is_file():
        with CACHE.open("rb+") as archivo:
            while True:
                posicion = archivo.tell()
                linea = archivo.readline()
                if not linea:
                    break
                try:
                    registro = json.loads(linea)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if archivo.read():
                        raise ValueError(
                            "La caché PVGIS contiene una línea dañada"
                        ) from None
                    archivo.truncate(posicion)
                    break
                consultas[registro["codigo"]] = registro
    return consultas


def guardar_consulta(registro: dict) -> None:
    """Guarda los rendimientos mensuales consultados para un municipio.

    Args:
        registro: Consulta PVGIS completa de un municipio.
    """

    with CACHE.open("a", encoding="utf-8") as archivo:
        archivo.write(
            json.dumps(registro, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        archivo.flush()
        os.fsync(archivo.fileno())


def solicitar_pvgis(sesion: requests.Session, servicio: str, parametros: dict) -> dict:
    """Consulta PVGIS secuencialmente y reintenta solo fallos temporales.

    Args:
        sesion: Sesión HTTP reutilizada para consultar PVGIS.
        servicio: Endpoint de PVGIS: PVcalc o seriescalc.
        parametros: Valores que se pasan a la consulta.

    Returns:
        dict: Respuesta JSON del servicio solicitado.

    """

    for intento in range(4):
        try:
            respuesta = sesion.get(
                f"{URL_PVGIS}/{servicio}", params=parametros, timeout=120
            )
            if respuesta.status_code in (429, 529) and intento < 3:
                time.sleep(2**intento)
                continue
            respuesta.raise_for_status()
            return respuesta.json()
        except (requests.ConnectionError, requests.Timeout):
            if intento == 3:
                raise
            time.sleep(2**intento)


def consultar_referencia(
    sesion: requests.Session,
    codigo: str,
    latitud: float,
    longitud: float,
) -> dict:
    """Consulta el rendimiento mensual de un kWp para el escenario regional.

    Args:
        sesion: Sesión HTTP reutilizada.
        codigo: Código del municipio.
        latitud: Latitud WGS84 del punto de consulta.
        longitud: Longitud WGS84 del punto de consulta.

    Returns:
        dict: Posición y doce medias mensuales de PVGIS, en kWh/kWp.
    """

    parametros = {
        "lat": latitud,
        "lon": longitud,
        "raddatabase": BASE_RADIACION,
        "peakpower": POTENCIA_REFERENCIA_KWP,
        "loss": PERDIDAS,
        "angle": INCLINACION,
        "aspect": ORIENTACION,
        "pvtechchoice": TECNOLOGIA,
        "mountingplace": MONTAJE,
        "usehorizon": USAR_HORIZONTE,
        "outputformat": "json",
    }

    datos = solicitar_pvgis(sesion, "PVcalc", parametros)
    mensual = datos["outputs"]["monthly"]["fixed"]
    rendimientos = [0.0] * 12

    for fila in mensual:
        rendimientos[int(fila["month"]) - 1] = float(fila["E_m"])

    if len(mensual) != 12 or not np.isfinite(rendimientos).all():
        raise ValueError(f"PVGIS no devolvió doce meses válidos para {codigo}")

    return {
        "codigo": codigo,
        "latitud": latitud,
        "longitud": longitud,
        "rendimientos_mensuales": rendimientos,
    }


def calcular_resultados(
    municipios: gpd.GeoDataFrame,
    consultas: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula potencia y producción municipal y mensual de los tres escenarios.

    Args:
        municipios: Superficie detectada por municipio.
        consultas: Rendimientos mensuales de PVGIS por código municipal.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Resumen municipal y detalle mensual.
    """

    filas_municipales = []
    filas_mensuales = []

    for municipio in municipios.itertuples():
        codigo = str(municipio.codigo)
        registro = consultas.get(codigo)

        if municipio.superficie_fv_detectada_total_m2 > 0 and registro is None:
            raise ValueError(f"Falta la consulta PVGIS del municipio {codigo}")

        rendimientos = registro["rendimientos_mensuales"] if registro else [0.0] * 12
        fila = {
            "codigo": codigo,
            "municipio": municipio.municipio,
            "superficie_fv_detectada_total_m2": municipio.superficie_fv_detectada_total_m2,
            "rendimiento_anual_kwh_kwp": sum(rendimientos),
        }

        potencias = {}
        for escenario, (campo, densidad) in ESCENARIOS.items():
            potencia = getattr(municipio, campo) * densidad
            potencias[escenario] = potencia
            fila[f"potencia_{escenario}_kwp"] = potencia
            fila[f"produccion_anual_{escenario}_kwh"] = potencia * sum(rendimientos)

        filas_municipales.append(fila)

        for mes, rendimiento in enumerate(rendimientos, 1):
            mensual = {
                "codigo": codigo,
                "municipio": municipio.municipio,
                "mes": mes,
                "rendimiento_mensual_kwh_kwp": rendimiento,
            }
            for escenario, potencia in potencias.items():
                mensual[f"produccion_{escenario}_kwh"] = potencia * rendimiento

            filas_mensuales.append(mensual)

    return pd.DataFrame(filas_municipales), pd.DataFrame(filas_mensuales)


def guardar_grafico(municipal: pd.DataFrame, mensual: pd.DataFrame) -> None:
    """Guarda el escenario regional por mes y los municipios con mayor producción."""

    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    por_mes = mensual.groupby("mes").produccion_central_kwh.sum() / 1_000_000

    ejes[0].bar(por_mes.index, por_mes.values, color="#28796f")
    ejes[0].set_xticks(range(1, 13))
    ejes[0].set_xlabel("Mes")
    ejes[0].set_ylabel("Producción central (GWh/mes)")

    mayores = municipal.nlargest(10, "produccion_anual_central_kwh")
    ejes[1].barh(
        mayores.municipio,
        mayores.produccion_anual_central_kwh / 1_000_000,
        color="#28796f",
    )
    ejes[1].invert_yaxis()
    ejes[1].set_xlabel("Producción central (GWh/año)")
    figura.suptitle("Escenario regional uniforme · PVGIS")
    figura.savefig(SALIDA_GRAFICO, dpi=170)
    plt.close(figura)


def main() -> None:
    """Calcula escenarios anuales y mensuales desde la inferencia regional terminada."""

    municipios, posiciones = cargar_entradas()
    consultas = preparar_cache(configuracion())

    with requests.Session() as sesion:
        for codigo, (latitud, longitud) in posiciones.items():
            anterior = consultas.get(codigo)
            if anterior and np.allclose(
                [anterior["latitud"], anterior["longitud"]],
                [latitud, longitud],
                rtol=0,
                atol=1e-9,
            ):
                continue

            print(f"PVGIS: {codigo}", flush=True)
            registro = consultar_referencia(sesion, codigo, latitud, longitud)
            guardar_consulta(registro)
            consultas[codigo] = registro

    municipal, mensual = calcular_resultados(municipios, consultas)
    municipal.to_csv(PRODUCCION_MUNICIPAL, index=False)
    mensual.to_csv(PRODUCCION_MENSUAL, index=False)
    guardar_grafico(municipal, mensual)
    print(f"Resultados de producción: {CARPETA}")
    print("Escenario uniforme; no asigna antigüedades a instalaciones.")


if __name__ == "__main__":
    main()
