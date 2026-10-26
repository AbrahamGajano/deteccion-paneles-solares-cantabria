"""Rutas independientes del directorio desde el que se ejecute Python."""

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def ruta_proyecto(ruta: str | Path) -> Path:
    """Completa una ruta relativa; si ya es absoluta, la devuelve sin cambiarla.

    Args:
        ruta: Ruta relativa a la raíz del proyecto o ruta absoluta.

    Returns:
        Path: Ruta absoluta, respetando las entradas que ya son absolutas.
    """

    ruta = Path(ruta)
    if ruta.is_absolute():
        return ruta
    return RAIZ / ruta


def ruta_registrada(ruta: str | Path) -> str:
    """Guarda rutas del proyecto relativas y conserva absolutas las externas.

    Args:
        ruta: Archivo cuya ubicación se registra en un manifiesto o metadato.

    Returns:
        str: Ruta relativa al proyecto cuando el archivo está dentro de él.
    """

    ruta = ruta_proyecto(ruta).resolve()

    if ruta.is_relative_to(RAIZ):
        return ruta.relative_to(RAIZ).as_posix()

    return str(ruta)


PNOA = ruta_proyecto("data/pnoa")
INDICE_TESELAS = ruta_proyecto("data/manifests/teselas.csv")
EDIFICIOS_CATASTRO = ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg")
MUNICIPIOS_CANTABRIA = ruta_proyecto("data/geografia/municipios_cantabria.gpkg")
DATASETS = ruta_proyecto("data/datasets")
DATASET_UNET = DATASETS / "unet"
DATASET_YOLO = DATASETS / "yolo"

PESOS_UNET = ruta_proyecto("weights/unet_resnet34.pt")
PESOS_YOLO = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")
EVALUACION_UNET = ruta_proyecto("runs/evaluacion/unet_actual")
INDICADORES_UNET = ruta_proyecto("runs/indicadores/unet_actual")
EVALUACION_SEGUIMIENTO = ruta_proyecto("runs/evaluacion/seguimiento")
INDICADORES_SEGUIMIENTO = ruta_proyecto("runs/indicadores/seguimiento_historico")
VISUALIZACION = ruta_proyecto("runs/visualizacion")
PRODUCCION_CONFIG = INDICADORES_UNET / "produccion_config.json"
PRODUCCION_MUNICIPAL = INDICADORES_UNET / "produccion_municipal.csv"
PRODUCCION_MENSUAL = INDICADORES_UNET / "produccion_mensual.csv"


def retirar_panel() -> None:
    """Retira las salidas del panel cuando cambian sus datos de entrada.

    La caché de PVGIS se conserva: cada consulta identifica sus parámetros.
    El mapa del test de los modelos es una salida independiente.
    """

    archivos = [
        "panel_indicadores.html",
        "resumen_indicadores.png",
        "resumen_municipal.csv",
        "procedencia_panel.json",
        "produccion_estimada.csv",
        "factores_energia.csv",
        "hipotesis_energia.json",
    ]

    for nombre in archivos:
        (VISUALIZACION / nombre).unlink(missing_ok=True)


def huella_archivo(ruta: Path) -> str:
    """Calcula SHA256 sin cargar el archivo completo en memoria.

    Args:
        ruta: Archivo cuyo contenido se quiere identificar.

    Returns:
        str: Huella hexadecimal del contenido.
    """

    huella = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            huella.update(bloque)
    return huella.hexdigest()


def version_indicadores() -> dict[str, dict[str, int | str]]:
    """Identifica los GeoPackage usados para estimar la producción regional.

    Returns:
        dict[str, dict[str, int | str]]: Tamaño y fechas de los GeoPackage de edificios
            y municipios.

    Raises:
        FileNotFoundError: Falta el GeoPackage de edificios o municipios.
    """

    version = {}
    for nombre in ("edificios", "municipios"):
        ruta = INDICADORES_UNET / f"{nombre}.gpkg"
        if not ruta.is_file():
            raise FileNotFoundError(ruta)
        estado = ruta.stat()
        version[nombre] = {
            "tamano_bytes": estado.st_size,
            "modificado_ns": estado.st_mtime_ns,
            "modificado_utc": datetime.fromtimestamp(
                estado.st_mtime, timezone.utc
            ).isoformat(),
        }
    return version


def reemplazar_carpeta(temporal: Path, destino: Path) -> None:
    """Publica un resultado terminado y retira únicamente su versión anterior.

    Se usa para entrenamientos y evaluaciones dentro de runs/. Si falla el cambio de
    nombre, se restaura la carpeta anterior.

    Args:
        temporal: Archivo o carpeta completa pendiente de publicar.
        destino: Ruta definitiva de la salida.

    Raises:
        ValueError: El destino queda fuera de runs o el temporal no es su carpeta
            hermana prevista.
        FileExistsError: Queda una sustitución anterior pendiente de revisar.
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
        raise FileExistsError(
            f"Hay una operación anterior pendiente de revisar: {anterior}"
        )
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
