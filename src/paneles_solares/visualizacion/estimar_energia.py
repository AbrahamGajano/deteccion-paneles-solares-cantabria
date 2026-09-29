"""Estima potencia pico y producción desde los indicadores U-Net de Cantabria."""

import json
import os
import time

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import requests

from paneles_solares.rutas import (
    INDICADORES_UNET,
    PRODUCCION_CONFIG,
    PRODUCCION_HORARIA,
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
ANIO_PERFIL = 2019  # Año no bisiesto dentro de la cobertura SARAH3 (2005-2023).
INCLINACION = 30
ORIENTACION = 0  # En PVGIS, 0 grados significa sur.
PERDIDAS = 14
POTENCIA_REFERENCIA_KWP = 1
TECNOLOGIA = "crystSi"
MONTAJE = "free"
USAR_HORIZONTE = 1
ESCENARIOS = {
    "conservador": ("superficie_fv_detectada_total_m2", 0.17),
    "central": ("superficie_fv_corregida_test_total_m2", 0.20),
    "superior": ("superficie_fv_corregida_test_total_m2", 0.23),
}


def configuracion() -> dict:
    """Reúne todas las hipótesis y la versión de los indicadores de entrada."""
    return {
        "url_pvgis": URL_PVGIS,
        "version_pvgis": VERSION_PVGIS,
        "base_radiacion": BASE_RADIACION,
        "anio_perfil_horario": ANIO_PERFIL,
        "inclinacion_grados": INCLINACION,
        "orientacion_pvgis_grados": ORIENTACION,
        "perdidas_pct": PERDIDAS,
        "potencia_referencia_kwp": POTENCIA_REFERENCIA_KWP,
        "tecnologia": TECNOLOGIA,
        "montaje": MONTAJE,
        "usar_horizonte_pvgis": bool(USAR_HORIZONTE),
        "calibracion_test_superficie": "superficie detectada / 0.9246 (calculada en indicadores_unet.py)",
        "densidades_kwp_m2": {clave: densidad for clave, (_, densidad) in ESCENARIOS.items()},
        "superficies_escenarios": {clave: campo for clave, (campo, _) in ESCENARIOS.items()},
        "indicadores_entrada": version_indicadores(),
        "descripcion": "Potencia pico estimada sobre superficie planimétrica; orientación, inclinación y pérdidas homogéneas. Perfil horario representativo normalizado a medias mensuales PVGIS; no es producción observada.",
    }


def calendario_horario() -> pd.DatetimeIndex:
    """Devuelve 8.760 horas UTC sin crear una fecha ficticia en las salidas."""
    horas = pd.date_range(
        f"{ANIO_PERFIL}-01-01", f"{ANIO_PERFIL + 1}-01-01",
        freq="h", inclusive="left", tz="UTC",
    )
    horas = horas[~((horas.month == 2) & (horas.day == 29))]
    if len(horas) != 8760:
        raise ValueError("El año elegido debe aportar exactamente 8.760 horas")
    return horas


def cargar_entradas() -> tuple[gpd.GeoDataFrame, dict[str, tuple[float, float]]]:
    """Lee municipios y edificios positivos; calcula sus posiciones ponderadas."""
    version_indicadores()
    municipios = gpd.read_file(MUNICIPIOS)
    edificios = gpd.read_file(
        EDIFICIOS, where="presencia_fotovoltaica = 1",
        columns=["codigo_municipal", "superficie_fv_detectada_m2", "geometry"],
    )
    if municipios.crs is None or edificios.crs is None or municipios.crs != edificios.crs:
        raise ValueError("Los dos GeoPackage necesitan el mismo CRS proyectado")
    if municipios.crs.to_epsg() != 25830:
        raise ValueError("Se espera EPSG:25830 para las superficies y la posición ponderada")
    municipios["codigo"] = municipios.codigo.astype(str)
    edificios["codigo_municipal"] = edificios.codigo_municipal.astype(str)
    if municipios.codigo.duplicated().any():
        raise ValueError("Los municipios tienen códigos duplicados")
    superficies = edificios.groupby("codigo_municipal").superficie_fv_detectada_m2.sum()
    recuentos = edificios.groupby("codigo_municipal").size()
    if not np.allclose(
        municipios.superficie_fv_detectada_total_m2.to_numpy(dtype=float),
        municipios.codigo.map(superficies).fillna(0).to_numpy(dtype=float), atol=0.01,
    ) or not np.array_equal(
        municipios.edificios_con_deteccion.to_numpy(dtype=int),
        municipios.codigo.map(recuentos).fillna(0).to_numpy(dtype=int),
    ):
        raise ValueError("Los edificios positivos y el resumen municipal no coinciden")
    puntos = edificios.geometry.representative_point()
    edificios["x"] = puntos.x
    edificios["y"] = puntos.y
    posiciones = {}
    for codigo, grupo in edificios.groupby("codigo_municipal"):
        pesos = grupo.superficie_fv_detectada_m2.to_numpy(dtype=float)
        if np.any(pesos < 0) or pesos.sum() <= 0:
            raise ValueError(f"Superficies detectadas inválidas en {codigo}")
        x = np.average(grupo.x, weights=pesos)
        y = np.average(grupo.y, weights=pesos)
        punto = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=edificios.crs).to_crs("EPSG:4326").iloc[0]
        posiciones[codigo] = (float(punto.y), float(punto.x))
    return municipios, posiciones


def preparar_cache(config: dict) -> dict[str, dict]:
    """Abre una única caché fija y descarta resultados de otra configuración."""
    anterior = json.loads(PRODUCCION_CONFIG.read_text(encoding="utf-8")) if PRODUCCION_CONFIG.is_file() else None
    if anterior != config:
        for ruta in (CACHE, PRODUCCION_MUNICIPAL, PRODUCCION_MENSUAL, PRODUCCION_HORARIA, SALIDA_GRAFICO):
            ruta.unlink(missing_ok=True)
        PRODUCCION_CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    elif not CACHE.is_file() and any(ruta.is_file() for ruta in (PRODUCCION_MUNICIPAL, PRODUCCION_MENSUAL, PRODUCCION_HORARIA)):
        raise ValueError("Falta la caché PVGIS de los resultados existentes; no se pueden reanudar")
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
                        raise ValueError("La caché PVGIS contiene una línea dañada") from None
                    archivo.truncate(posicion)
                    break
                consultas[registro["codigo"]] = registro
    return consultas


def guardar_consulta(registro: dict) -> None:
    """Persiste un municipio completo tras terminar ambas llamadas a PVGIS."""
    with CACHE.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False, separators=(",", ":")) + "\n")
        archivo.flush()
        os.fsync(archivo.fileno())


def solicitar_pvgis(sesion: requests.Session, servicio: str, parametros: dict) -> dict:
    """Consulta PVGIS secuencialmente y reintenta solo fallos temporales."""
    for intento in range(4):
        try:
            respuesta = sesion.get(f"{URL_PVGIS}/{servicio}", params=parametros, timeout=120)
            if respuesta.status_code in (429, 529) and intento < 3:
                time.sleep(2 ** intento)
                continue
            respuesta.raise_for_status()
            return respuesta.json()
        except (requests.ConnectionError, requests.Timeout):
            if intento == 3:
                raise
            time.sleep(2 ** intento)
    raise RuntimeError("No se pudo completar la consulta a PVGIS")


def consultar_referencia(
    sesion: requests.Session, codigo: str, latitud: float, longitud: float,
    horas: pd.DatetimeIndex,
) -> dict:
    """Obtiene medias mensuales y forma horaria de un único kWp."""
    parametros = {
        "lat": latitud, "lon": longitud, "raddatabase": BASE_RADIACION,
        "peakpower": POTENCIA_REFERENCIA_KWP, "loss": PERDIDAS,
        "angle": INCLINACION, "aspect": ORIENTACION,
        "pvtechchoice": TECNOLOGIA, "mountingplace": MONTAJE,
        "usehorizon": USAR_HORIZONTE,
        "outputformat": "json",
    }
    mensual = solicitar_pvgis(sesion, "PVcalc", parametros)["outputs"]["monthly"]["fixed"]
    rendimientos = np.zeros(12, dtype=float)
    for fila in mensual:
        rendimientos[int(fila["month"]) - 1] = float(fila["E_m"])
    if (len(mensual) != 12 or {int(fila["month"]) for fila in mensual} != set(range(1, 13))
        or not np.all(np.isfinite(rendimientos)) or np.any(rendimientos < 0)):
        raise ValueError(f"PVcalc devolvió meses inválidos para {codigo}")
    horario = solicitar_pvgis(
        sesion, "seriescalc", {
            **parametros, "pvcalculation": 1,
            "startyear": ANIO_PERFIL, "endyear": ANIO_PERFIL,
        },
    )["outputs"]["hourly"]
    tiempos = pd.to_datetime([fila["time"] for fila in horario], format="%Y%m%d:%H%M", utc=True).floor("h")
    potencias_w = np.array([float(fila["P"]) for fila in horario])
    vigentes = ~((tiempos.month == 2) & (tiempos.day == 29))
    tiempos = tiempos[vigentes]
    potencias_w = potencias_w[vigentes]
    if not tiempos.equals(horas) or not np.all(np.isfinite(potencias_w)) or np.any(potencias_w < 0):
        raise ValueError(f"seriescalc devolvió horas o potencias inválidas para {codigo}")
    return {
        "codigo": codigo, "latitud": latitud, "longitud": longitud,
        "rendimientos_mensuales": rendimientos.tolist(),
        "potencias_horarias_w": potencias_w.tolist(),
    }


def normalizar_perfil(registro: dict, horas: pd.DatetimeIndex) -> np.ndarray:
    """Ajusta la distribución horaria a cada media mensual PVcalc en kWh/kWp."""
    forma = np.asarray(registro["potencias_horarias_w"], dtype=float) / 1000  # W medios durante una hora -> kWh.
    meses = horas.month.to_numpy()
    rendimiento = np.asarray(registro["rendimientos_mensuales"], dtype=float)
    if len(forma) != 8760 or len(rendimiento) != 12 or np.any(forma < 0) or np.any(rendimiento < 0):
        raise ValueError(f"Caché PVGIS incompleta para {registro['codigo']}")
    normalizado = np.zeros(8760, dtype=float)
    for mes in range(1, 13):
        mascara = meses == mes
        total = forma[mascara].sum()
        if total == 0 and rendimiento[mes - 1] > 0:
            raise ValueError(f"Perfil horario nulo con producción mensual en {registro['codigo']}, mes {mes}")
        if total > 0:
            normalizado[mascara] = forma[mascara] * rendimiento[mes - 1] / total
    return normalizado


def calcular_resultados(
    municipios: gpd.GeoDataFrame, consultas: dict[str, dict],
    horas: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Escala los rendimientos de referencia a los tres escenarios."""
    filas_municipales = []
    filas_mensuales = []
    filas_horarias = []
    meses_horas = horas.month.to_numpy()
    base_horas = pd.DataFrame({
        "mes": meses_horas, "dia": horas.day, "hora_utc": horas.hour,
    })
    for municipio in municipios.itertuples():
        codigo = str(municipio.codigo)
        registro = consultas.get(codigo)
        if municipio.edificios_con_deteccion > 0 and registro is None:
            raise ValueError(f"Falta la consulta PVGIS del municipio positivo {codigo}")
        rendimiento_mensual = (
            np.asarray(registro["rendimientos_mensuales"], dtype=float)
            if registro else np.zeros(12)
        )
        rendimiento_horario = normalizar_perfil(registro, horas) if registro else np.zeros(8760)
        superficie_detectada = float(municipio.superficie_fv_detectada_total_m2)
        superficie_corregida = float(municipio.superficie_fv_corregida_test_total_m2)
        resumen = {
            "codigo": codigo, "municipio": municipio.municipio,
            "latitud": registro["latitud"] if registro else np.nan,
            "longitud": registro["longitud"] if registro else np.nan,
            "superficie_fv_detectada_m2": superficie_detectada,
            "superficie_fv_corregida_test_m2": superficie_corregida,
            "rendimiento_anual_kwh_kwp": rendimiento_mensual.sum(),
            "inclinacion_grados": INCLINACION,
            "orientacion_pvgis_grados": ORIENTACION,
            "perdidas_pct": PERDIDAS,
            "consultado_pvgis": registro is not None,
        }
        meses = pd.DataFrame({
            "codigo": codigo, "municipio": municipio.municipio,
            "mes": np.arange(1, 13),
            "rendimiento_mensual_kwh_kwp": rendimiento_mensual,
        })
        serie = base_horas.copy()
        serie.insert(0, "municipio", municipio.municipio)
        serie.insert(0, "codigo", codigo)
        serie["rendimiento_horario_kwh_kwp"] = rendimiento_horario
        for escenario, (campo, densidad) in ESCENARIOS.items():
            superficie = float(getattr(municipio, campo))
            potencia = superficie * densidad
            resumen[f"potencia_{escenario}_kwp"] = potencia
            resumen[f"produccion_anual_{escenario}_kwh"] = rendimiento_mensual.sum() * potencia
            meses[f"produccion_{escenario}_kwh"] = rendimiento_mensual * potencia
            serie[f"produccion_{escenario}_kwh"] = rendimiento_horario * potencia
        filas_municipales.append(resumen)
        filas_mensuales.append(meses)
        filas_horarias.append(serie)
    return (
        pd.DataFrame(filas_municipales),
        pd.concat(filas_mensuales, ignore_index=True),
        pd.concat(filas_horarias, ignore_index=True),
    )


def comprobar_resultados(
    municipal: pd.DataFrame, mensual: pd.DataFrame, horaria: pd.DataFrame,
) -> None:
    """Comprueba orden de escenarios, calendario y conservación de energía."""
    for datos in (municipal, mensual, horaria):
        columnas = [
            nombre for nombre in datos.columns
            if nombre.startswith(("superficie_", "potencia_", "produccion_", "rendimiento_"))
        ]
        valores = datos[columnas].to_numpy(dtype=float)
        if not np.all(np.isfinite(valores)) or np.any(valores < 0):
            raise ValueError("Hay superficies, potencias o energías negativas o no finitas")
        magnitudes = (
            [("potencia", "kwp"), ("produccion_anual", "kwh")]
            if datos is municipal else [("produccion", "kwh")]
        )
        for magnitud, unidad in magnitudes:
            conservador = datos[f"{magnitud}_conservador_{unidad}"]
            central = datos[f"{magnitud}_central_{unidad}"]
            superior = datos[f"{magnitud}_superior_{unidad}"]
            if np.any(conservador > central + 1e-8):
                raise ValueError("El escenario conservador supera al central")
            if np.any(central > superior + 1e-8):
                raise ValueError("El escenario central supera al superior")
    codigos = set(municipal.codigo)
    if (set(mensual.codigo) != codigos or set(horaria.codigo) != codigos or
        not mensual.groupby("codigo").mes.nunique().eq(12).all() or
        not mensual.groupby("codigo").size().eq(12).all() or
        not horaria.groupby("codigo").size().eq(8760).all() or
        horaria.duplicated(["codigo", "mes", "dia", "hora_utc"]).any()):
        raise ValueError("Faltan meses u horas en uno o varios municipios")
    for escenario in ESCENARIOS:
        columna = f"produccion_{escenario}_kwh"
        suma_horaria = horaria.groupby(["codigo", "mes"])[columna].sum().sort_index()
        suma_mensual = mensual.set_index(["codigo", "mes"])[columna].sort_index()
        if not np.allclose(suma_horaria, suma_mensual, rtol=0, atol=1e-6):
            raise ValueError(f"El perfil horario no coincide con los meses: {escenario}")
        anuales = mensual.groupby("codigo")[columna].sum().sort_index()
        resumen = municipal.set_index("codigo")[f"produccion_anual_{escenario}_kwh"].sort_index()
        if not np.allclose(anuales, resumen, rtol=0, atol=1e-6):
            raise ValueError(f"Los meses no coinciden con la producción anual: {escenario}")
        if not np.isclose(mensual[columna].sum(), resumen.sum(), rtol=0, atol=1e-6):
            raise ValueError(f"El total de Cantabria no coincide: {escenario}")
        if not np.isclose(horaria[columna].sum(), resumen.sum(), rtol=0, atol=1e-6):
            raise ValueError(f"El perfil horario de Cantabria no coincide: {escenario}")


def guardar_grafico(municipal: pd.DataFrame, mensual: pd.DataFrame, horaria: pd.DataFrame) -> None:
    """Dibuja cuatro vistas de potencia y producción para la memoria."""
    figura, ejes = plt.subplots(2, 2, figsize=(16, 12))
    figura.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.09, hspace=0.30, wspace=0.28)
    totales = mensual.groupby("mes")[[f"produccion_{e}_kwh" for e in ESCENARIOS]].sum() / 1e6
    meses = totales.index.to_numpy()
    ejes[0, 0].fill_between(meses, totales.produccion_conservador_kwh.to_numpy(),
                            totales.produccion_superior_kwh.to_numpy(), alpha=0.25, color="#2e78a6")
    ejes[0, 0].plot(meses, totales.produccion_central_kwh, color="#155e75", lw=2)
    ejes[0, 0].set(title="Producción mensual estimada · Cantabria", xlabel="Mes", ylabel="GWh", xticks=meses)

    mayores = municipal.nlargest(15, "produccion_anual_central_kwh").iloc[::-1]
    ejes[0, 1].barh(mayores.municipio, mayores.produccion_anual_central_kwh / 1e6, color="#20705c")
    ejes[0, 1].set(title="15 municipios con mayor producción central", xlabel="GWh/año")

    perfil = horaria.groupby("hora_utc").produccion_central_kwh.sum() / 365 / 1000
    ejes[1, 0].plot(perfil.index, perfil, color="#cc4c02", lw=2)
    ejes[1, 0].set(title="Perfil medio diario representativo · Cantabria", xlabel="Hora UTC", ylabel="MWh por hora", xticks=np.arange(0, 24, 3))

    ejes[1, 1].scatter(municipal.potencia_central_kwp / 1000,
                       municipal.produccion_anual_central_kwh / 1000,
                       color="#8c4b9f", alpha=0.7, s=35)
    ejes[1, 1].set(title="Potencia pico y producción anual centrales", xlabel="Potencia estimada (MWp)", ylabel="Producción estimada (MWh/año)")
    for eje in ejes.flat:
        eje.grid(alpha=0.2)
        eje.set_axisbelow(True)
    figura.suptitle("Potencia pico y producción fotovoltaica estimadas · Cantabria", fontsize=17)
    figura.text(0.5, 0.025,
                f"Superficie planimétrica U-Net; corrección experimental del test. PVGIS {VERSION_PVGIS}: "
                f"{INCLINACION}° de inclinación, {ORIENTACION}° de orientación, "
                f"pérdidas {PERDIDAS} %. Perfil horario representativo.",
                ha="center", fontsize=9)
    figura.savefig(SALIDA_GRAFICO, dpi=180, facecolor="white")
    plt.close(figura)


def prueba() -> None:
    """Consulta un municipio positivo y valida el cálculo sin escribir salidas."""
    municipios, posiciones = cargar_entradas()
    horas = calendario_horario()
    municipio = municipios.loc[municipios.codigo.isin(posiciones)].iloc[[0]]
    codigo = str(municipio.codigo.iloc[0])
    with requests.Session() as sesion:
        registro = consultar_referencia(sesion, codigo, *posiciones[codigo], horas)
    resumen, mensual, horaria = calcular_resultados(municipio, {codigo: registro}, horas)
    comprobar_resultados(resumen, mensual, horaria)
    print(f"Prueba correcta: {municipio.municipio.iloc[0]} ({codigo}); 12 meses, 8.760 horas; 2 consultas PVGIS")
    print(f"Potencia central: {resumen.potencia_central_kwp.iloc[0]:.2f} kWp")
    print(f"Producción anual central: {resumen.produccion_anual_central_kwh.iloc[0]:.2f} kWh")


def main() -> None:
    """Calcula Cantabria usando una única caché PVGIS reanudable."""
    municipios, posiciones = cargar_entradas()
    horas = calendario_horario()
    consultas = preparar_cache(configuracion())
    with requests.Session() as sesion:
        for codigo, (latitud, longitud) in posiciones.items():
            if codigo in consultas:
                if not np.allclose([consultas[codigo]["latitud"], consultas[codigo]["longitud"]],
                                   [latitud, longitud], rtol=0, atol=1e-9):
                    raise ValueError(f"La posición en caché ha cambiado: {codigo}")
                continue
            print(f"PVGIS: {codigo} ({len(consultas) + 1}/{len(posiciones)})", flush=True)
            registro = consultar_referencia(sesion, codigo, latitud, longitud, horas)
            guardar_consulta(registro)
            consultas[codigo] = registro
    municipal, mensual, horaria = calcular_resultados(municipios, consultas, horas)
    comprobar_resultados(municipal, mensual, horaria)
    municipal.to_csv(PRODUCCION_MUNICIPAL, index=False)
    mensual.to_csv(PRODUCCION_MENSUAL, index=False)
    horaria.to_csv(PRODUCCION_HORARIA, index=False)
    guardar_grafico(municipal, mensual, horaria)
    print(f"Resultados de producción: {CARPETA}")
    print("Actualiza el panel: python -m paneles_solares.visualizacion.panel_indicadores")


if __name__ == "__main__":
    main()
