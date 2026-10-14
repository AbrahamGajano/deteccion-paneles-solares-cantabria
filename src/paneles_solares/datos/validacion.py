"""Configuración y rutas de las validaciones de resolución."""

import json
from pathlib import Path

from paneles_solares.rutas import ruta_proyecto

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


def cargar_campania(nombre: str, archivo: str = "validaciones.json") -> dict:
    """Lee una campaña; sus datos siempre quedan dentro de datasets.

    Args:
        nombre: Nombre de la campaña configurada.
        archivo: Ruta relativa al proyecto o absoluta.

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
    campania = json.loads(ruta_proyecto(archivo).read_text(encoding="utf-8"))[nombre]
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
