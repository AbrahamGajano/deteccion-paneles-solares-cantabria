"""Configuración y rutas de las validaciones de resolución."""

import json
from copy import deepcopy
from pathlib import Path

from paneles_solares.rutas import ruta_proyecto

CAMPANIAS = {
    "20cm": {
        "anio": 2024,
        "resolucion_m": 0.2,
        "procedencia": "Gobierno de Cantabria: Ortofoto 2024 True Ortho, 20 cm "
        "nativos. "
        "https://geoservicios.cantabria.es/inspire/rest/services/Ortofoto_2024/MapServer",
        "fuentes": [
            {
                "tipo": "wcs",
                "url": "https://geoservicios.cantabria.es/inspire/services/Ortofoto_Series_WCS/MapServer/WCSServer",
                "cobertura": "16",
                "etiqueta": "Ortofoto_de_2024_True_Ortho",
                "bandas": [1, 2, 3],
                "divisor_rgb": 257.0,
            }
        ],
        "aleatorias": 219,
        "positivas": 31,
        "semilla": 42,
        "bootstrap": 1000,
        "ejemplos_por_categoria": 2,
    }
}

EXTENSION_M = 76.8
TAM_MODELO = 512
GRUPOS = ("positivos", "aleatorias")
CAMPOS_MUESTRA = (
    "anio",
    "resolucion_m",
    "procedencia",
    "fuentes",
    "semilla",
    "aleatorias",
    "positivas",
)


def cargar_campania(nombre: str, archivo: str | Path | None = None) -> dict:
    """Lee una campaña; sus datos siempre quedan dentro de datasets.

    Args:
        nombre: Nombre de la campaña configurada.
        archivo: JSON opcional; None utiliza CAMPANIAS de este módulo.

    Returns:
        dict: Configuración de la campaña solicitada.

    Raises:
        ValueError: El nombre, la resolución o las cantidades de la campaña son
            inválidos.
    """

    if not nombre or any(
        letra not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for letra in nombre
    ):
        raise ValueError(
            "Usa un nombre de campaña con letras minúsculas, números, _ o -"
        )
    if archivo is None:
        campania = deepcopy(CAMPANIAS[nombre])
    else:
        campania = json.loads(ruta_proyecto(archivo).read_text(encoding="utf-8"))[
            nombre
        ]
    resolucion = campania["resolucion_m"]
    if resolucion <= 0 or campania["aleatorias"] < 1 or campania["positivas"] < 0:
        raise ValueError("Revisa la resolución y las cantidades de la campaña")
    if campania["bootstrap"] < 0 or campania["ejemplos_por_categoria"] < 0:
        raise ValueError("Bootstrap y ejemplos deben ser cantidades no negativas")
    return campania


def carpeta_campania(nombre: str) -> Path:
    """Obtiene la carpeta de datos de una campaña de validación.

    Args:
        nombre: Nombre de la campaña configurada.

    Returns:
        Path: Carpeta de la campaña dentro de data/datasets.
    """

    return ruta_proyecto(f"data/datasets/validacion_{nombre}")


def rutas_tesela(carpeta: Path, fila) -> tuple[Path, Path]:
    """Obtiene el PNG y el JSON de una tesela en su grupo de selección.

    Args:
        carpeta: Carpeta de datos de la campaña.
        fila: Metadatos de la tesela o ventana.

    Returns:
        tuple[Path, Path]: Rutas de la imagen PNG y de la anotación JSON.
    """

    nombre = f"{fila['tile_id']}.png"
    imagen = carpeta / "images" / fila["grupo"] / nombre
    anotacion = carpeta / "annotations" / fila["grupo"] / f"{fila['tile_id']}.json"
    return imagen, anotacion
