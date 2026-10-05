"""Configuración y rutas de las validaciones de resolución."""

import json
from pathlib import Path

from paneles_solares.rutas import ruta_proyecto

EXTENSION_M = 76.8
TAM_MODELO = 512
GRUPOS = ("positivos", "aleatorias")


def cargar_campania(nombre: str, archivo: str = "validaciones.json") -> dict:
    """Lee una campaña; sus datos siempre quedan dentro de datasets."""
    if not nombre or any(letra not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for letra in nombre):
        raise ValueError("Usa un nombre de campaña con letras minúsculas, números, _ o -")
    campania = json.loads(ruta_proyecto(archivo).read_text(encoding="utf-8"))[nombre]
    resolucion = campania["resolucion_m"]
    if resolucion <= 0 or campania["aleatorias"] < 1 or campania["positivas"] < 0:
        raise ValueError("Revisa la resolución y las cantidades de la campaña")
    if campania["bootstrap"] < 0 or campania["ejemplos_por_categoria"] < 0:
        raise ValueError("Bootstrap y ejemplos deben ser cantidades no negativas")
    return campania


def carpeta_campania(nombre: str) -> Path:
    return ruta_proyecto(f"data/datasets/validacion_{nombre}")


def rutas_tesela(carpeta: Path, fila) -> tuple[Path, Path]:
    nombre = f"{fila['tile_id']}.png"
    imagen = carpeta / "images" / fila["grupo"] / nombre
    anotacion = carpeta / "annotations" / fila["grupo"] / f"{fila['tile_id']}.json"
    return imagen, anotacion
