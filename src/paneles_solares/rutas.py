"""Rutas independientes del directorio desde el que se ejecute Python."""

import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def ruta_proyecto(ruta: str | Path) -> Path:
    """Completa una ruta relativa; si ya es absoluta, la devuelve sin cambiarla."""
    ruta = Path(ruta)
    if ruta.is_absolute():
        return ruta
    return RAIZ / ruta


def reemplazar_carpeta(temporal: Path, destino: Path) -> None:
    """Publica un resultado terminado y retira únicamente su versión anterior.

    Se usa para el dataset y las runs. Los originales del pool nunca pasan aquí.
    Si falla el cambio de nombre, se restaura la carpeta anterior.
    """
    temporal = temporal.resolve()
    destino = destino.resolve()
    permitidas = (RAIZ / "data/datasets", RAIZ / "runs")
    if not any(destino.is_relative_to(base) and destino != base for base in permitidas):
        raise ValueError(f"Destino fuera de datasets/runs: {destino}")
    if temporal != destino.with_name(f".{destino.name}_preparando"):
        raise ValueError("La carpeta temporal debe ser hermana del destino")
    anterior = destino.with_name(f".{destino.name}_anterior")
    if anterior.exists():
        raise FileExistsError(f"Hay una operación anterior pendiente de revisar: {anterior}")
    if destino.exists():
        destino.rename(anterior)
    try:
        temporal.rename(destino)
    except OSError:
        if anterior.exists():
            anterior.rename(destino)
        raise
    if anterior.exists():
        shutil.rmtree(anterior)
