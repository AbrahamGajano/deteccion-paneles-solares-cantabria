"""Lectura de las ventanas RGB descritas en los manifiestos PNOA."""

from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio
from rasterio.windows import Window


def leer_tesela(registro: Mapping, carpeta_pnoa: Path) -> np.ndarray:
    """Lee solamente el recorte indicado por una fila del índice de teselas.

    Args:
        registro: Diccionario o fila del manifest con tif, fila, columna, ancho y alto.
        carpeta_pnoa: Carpeta que contiene las ortofotos originales.

    Returns:
        Imagen RGB con dimensiones alto × ancho × canales.
    """

    ruta = carpeta_pnoa / str(registro["tif"])

    ventana = Window(
        col_off=int(registro["columna"]),
        row_off=int(registro["fila"]),
        width=int(registro["ancho"]),
        height=int(registro["alto"]),
    )

    with rasterio.open(ruta) as ortofoto:
        imagen = ortofoto.read([1, 2, 3], window=ventana)

    # Rasterio devuelve canales × alto × ancho; las imágenes usan los canales al final.
    imagen_rgb = np.moveaxis(imagen, 0, -1)

    return imagen_rgb
