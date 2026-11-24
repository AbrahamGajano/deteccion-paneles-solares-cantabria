"""Relaciona la inclinación y el azimut LiDAR con las detecciones de 2023."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj


ANIO_LIDAR = 2023
INCLINACION_MINIMA_AZIMUT = 5
COMPONENTES_NORMAL = ["normal_x", "normal_y", "normal_z"]


def normales_guardadas(resultados: Path, identificadores) -> np.ndarray:
    """Recupera las normales de los JSON de un lote anterior al CSV vectorial."""

    normales = []
    for identificador in identificadores:
        archivo = resultados.parent / identificador / "resumen.json"
        try:
            normal = json.loads(archivo.read_text(encoding="utf-8")).get("normal")
        except FileNotFoundError:
            normal = None
        normales.append(normal if normal is not None else [np.nan] * 3)
    return np.asarray(normales, dtype=float)


def azimut_geografico(paneles, normales: np.ndarray) -> np.ndarray:
    """Calcula el azimut horario desde el norte geográfico a partir del vector UTM.

    Params:
        paneles: Detecciones con geometría georreferenciada.
        normales: Vectores en el sistema LiDAR EPSG:25830, uno por detección.

    Returns:
        np.ndarray: Azimut en grados de 0 a 360. NaN si falta una normal utilizable.
    """

    este, norte, arriba = normales.T
    validas = np.isfinite(normales).all(axis=1) & (np.hypot(este, norte) > 0) & (arriba != 0)
    signo = np.where(arriba < 0, -1, 1)
    azimut_utm = np.degrees(np.arctan2(signo * este, signo * norte))
    centros = paneles.to_crs(25830).geometry.centroid.to_crs(4258)
    convergencia = Proj("EPSG:25830").get_factors(
        centros.x.to_numpy(), centros.y.to_numpy(),
    ).meridian_convergence
    return np.where(validas, (azimut_utm + convergencia) % 360, np.nan)


def incorporar_inclinaciones(paneles, resultados: Path | None):
    """Añade inclinación aceptada, azimut y procedencia a las detecciones.

    Params:
        paneles: GeoDataFrame con id_panel y anio. Los identificadores originales
            vinculan la detección y su geometría con el procesamiento LiDAR.
        resultados: Resumen CSV del lote LiDAR. None desactiva su aplicación.

    Returns:
        GeoDataFrame: Paneles con inclinación y azimut geográfico de 2023.
            El azimut queda vacío para pendientes menores de 5° o sin normal.
    """

    datos = paneles.copy()
    datos["inclinacion_estimada"] = np.nan
    datos["origen_inclinacion"] = "sin_lidar"
    datos["motivo_inclinacion"] = "lidar_desactivado"

    datos["azimut_lidar_grados"] = np.nan
    datos["origen_orientacion"] = "supuesta"
    if resultados is None:
        return datos

    columnas = ["id_panel", "inclinacion_estimada", "origen", "motivo"]
    estimaciones = pd.read_csv(resultados, usecols=lambda nombre: nombre in columnas + COMPONENTES_NORMAL)
    aceptadas = estimaciones.origen.eq("estimada")
    angulos = estimaciones.loc[aceptadas, "inclinacion_estimada"]
    if not (np.isfinite(angulos) & angulos.ge(0) & angulos.lt(90)).all():
        raise ValueError("El resumen LiDAR contiene inclinaciones aceptadas no válidas")
    estimaciones.loc[~aceptadas, "inclinacion_estimada"] = np.nan
    estimaciones["anio"] = ANIO_LIDAR
    estimaciones = estimaciones.rename(columns={
        "origen": "origen_inclinacion", "motivo": "motivo_inclinacion",
    })
    datos = datos.drop(columns=[
        "inclinacion_estimada", "origen_inclinacion", "motivo_inclinacion",
    ]).merge(estimaciones, on=["id_panel", "anio"], how="left", validate="many_to_one")
    sin_resultado = datos.anio.eq(ANIO_LIDAR)
    datos["origen_inclinacion"] = datos.origen_inclinacion.fillna(
        pd.Series(np.where(sin_resultado, "sin_resultado", "otra_campania"), index=datos.index)
    )
    datos["motivo_inclinacion"] = datos.motivo_inclinacion.fillna(
        "sin_estimacion_lidar_para_esta_deteccion_y_campania"
    )

    casi_horizontal = datos.inclinacion_estimada.lt(INCLINACION_MINIMA_AZIMUT)
    datos.loc[casi_horizontal, "origen_orientacion"] = "pendiente_baja"
    orientables = datos.inclinacion_estimada.ge(INCLINACION_MINIMA_AZIMUT)
    if orientables.any():
        parte = datos.loc[orientables]
        if set(COMPONENTES_NORMAL).issubset(parte.columns):
            normales = parte[COMPONENTES_NORMAL].to_numpy(float)
        else:
            normales = normales_guardadas(resultados, parte.id_panel)
        datos.loc[orientables, "azimut_lidar_grados"] = azimut_geografico(parte, normales)
        datos.loc[orientables, "origen_orientacion"] = "sin_normal"
        datos.loc[datos.azimut_lidar_grados.notna(), "origen_orientacion"] = "estimada"

    return datos.drop(columns=COMPONENTES_NORMAL, errors="ignore")
