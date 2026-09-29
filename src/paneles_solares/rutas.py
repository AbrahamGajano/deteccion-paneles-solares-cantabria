"""Rutas independientes del directorio desde el que se ejecute Python."""

import shutil
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def ruta_proyecto(ruta: str | Path) -> Path:
    """Completa una ruta relativa; si ya es absoluta, la devuelve sin cambiarla."""
    ruta = Path(ruta)
    if ruta.is_absolute():
        return ruta
    return RAIZ / ruta


PESOS_UNET = ruta_proyecto("weights/unet_resnet34.pt")
EVALUACION_UNET = ruta_proyecto("runs/evaluacion/unet_actual")
INDICADORES_UNET = ruta_proyecto("runs/indicadores/unet_actual")
PRODUCCION_CONFIG = INDICADORES_UNET / "produccion_config.json"
PRODUCCION_MUNICIPAL = INDICADORES_UNET / "produccion_municipal.csv"
PRODUCCION_MENSUAL = INDICADORES_UNET / "produccion_mensual.csv"
PRODUCCION_HORARIA = INDICADORES_UNET / "produccion_horaria_representativa.csv"


def version_indicadores() -> dict[str, dict[str, int | str]]:
    """Identifica los GeoPackage usados para estimar producción y dibujar el panel."""
    version = {}
    for nombre in ("edificios", "municipios"):
        ruta = INDICADORES_UNET / f"{nombre}.gpkg"
        if not ruta.is_file():
            raise FileNotFoundError(ruta)
        estado = ruta.stat()
        version[nombre] = {
            "tamano_bytes": estado.st_size,
            "modificado_ns": estado.st_mtime_ns,
            "modificado_utc": datetime.fromtimestamp(estado.st_mtime, timezone.utc).isoformat(),
        }
    return version


def reemplazar_carpeta(temporal: Path, destino: Path) -> None:
    """Publica un resultado terminado y retira únicamente su versión anterior.

    Se usa para entrenamientos y evaluaciones dentro de runs/.
    Si falla el cambio de nombre, se restaura la carpeta anterior.
    """
    temporal = temporal.resolve()
    destino = destino.resolve()
    base = RAIZ / "runs"
    if not destino.is_relative_to(base) or destino == base:
        raise ValueError(f"Destino fuera de runs: {destino}")
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
