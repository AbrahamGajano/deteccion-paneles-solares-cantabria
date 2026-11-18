"""Relaciona las estimaciones LiDAR de 2023 con sus detecciones originales."""

from pathlib import Path

import numpy as np
import pandas as pd


ANIO_LIDAR = 2023


def incorporar_inclinaciones(paneles, resultados: Path | None):
    """Añade la inclinación aceptada y su procedencia sin cambiar las geometrías.

    Params:
        paneles: GeoDataFrame con id_panel y anio. Los identificadores originales
            vinculan la detección y su geometría con el procesamiento LiDAR.
        resultados: Resumen CSV del lote LiDAR. None desactiva su aplicación.

    Returns:
        GeoDataFrame: Paneles con inclinacion_estimada, origen_inclinacion y
            motivo_inclinacion. Las campañas distintas de 2023 no heredan ángulos.
    """

    datos = paneles.copy()
    datos["inclinacion_estimada"] = np.nan
    datos["origen_inclinacion"] = "sin_lidar"
    datos["motivo_inclinacion"] = "lidar_desactivado"

    if resultados is None:
        return datos

    columnas = ["id_panel", "inclinacion_estimada", "origen", "motivo"]
    estimaciones = pd.read_csv(resultados, usecols=columnas)
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

    return datos
