"""Incorpora al pool las imágenes revisadas y deja en labeling lo pendiente."""

import json
from pathlib import Path

import pandas as pd

from paneles_solares.datos.anotaciones import excluida, revisar_par
from paneles_solares.datos.coleccion import (
    LABELING,
    POOL,
    cargar_manifest,
    guardar_manifest,
    nueva_fila,
    rutas_muestra,
)


def cargar_pendientes(labeling: Path) -> pd.DataFrame:
    """Carga labeling y registra los PNG que se hayan añadido manualmente.

    Args:
        labeling (Path): Carpeta temporal de etiquetado.

    Returns:
        pd.DataFrame: Manifest de las imágenes pendientes.
    """

    pendientes = cargar_manifest(labeling / "manifest.csv")
    registrados = set(pendientes["tile_id"])

    for imagen in sorted((labeling / "images").glob("*.png")):
        if imagen.stem in registrados:
            continue

        fila = nueva_fila({"tile_id": imagen.stem}, "manual")
        pendientes.loc[len(pendientes)] = fila

    # Registramos los archivos manuales antes de moverlos para poder reanudar.
    guardar_manifest(pendientes, labeling / "manifest.csv")

    return pendientes


def localizar_par(
    tile_id: str,
    labeling: Path,
    pool: Path,
) -> tuple[Path, Path]:
    """Localiza el par aunque una incorporación anterior quedara interrumpida.

    Args:
        tile_id (str): Identificador de la imagen.
        labeling (Path): Carpeta de origen.
        pool (Path): Colección global.

    Returns:
        tuple[Path, Path]: Ubicación actual del PNG y del JSON.
    """

    imagen, anotacion = rutas_muestra(labeling, tile_id)
    imagen_pool, anotacion_pool = rutas_muestra(pool, tile_id)

    if not imagen.exists() and imagen_pool.exists():
        imagen = imagen_pool

    if not anotacion.exists() and anotacion_pool.exists():
        anotacion = anotacion_pool

    return imagen, anotacion


def actualizar_estado(fila: pd.Series, datos: dict) -> pd.Series:
    """Marca la muestra como válida o descartada según LabelMe.

    Args:
        fila (pd.Series): Registro de la imagen.
        datos (dict): Anotación revisada de LabelMe.

    Returns:
        pd.Series: Registro preparado para incorporarlo al pool.
    """

    # La fila combina textos del CSV con números, como el total de instancias.
    fila = fila.astype(object)

    if excluida(datos):
        fila["estado"] = "descartada"
        fila["uso"] = "no_usar"
        fila["tipo"] = ""
        fila["instancias"] = ""
        fila["motivo"] = "marcada en LabelMe"

    else:
        fila["estado"] = "valida"
        fila["uso"] = "pendiente"
        fila["tipo"] = "positiva" if datos["shapes"] else "negativa"
        fila["instancias"] = len(datos["shapes"])
        fila["motivo"] = ""

    return fila


def mover_al_pool(
    tile_id: str,
    datos: dict,
    labeling: Path,
    pool: Path,
) -> None:
    """Mueve el par al pool y actualiza la ruta de la imagen dentro del JSON.

    Args:
        tile_id (str): Identificador de la muestra.
        datos (dict): Anotación revisada.
        labeling (Path): Carpeta de origen.
        pool (Path): Carpeta de destino.
    """

    imagen, anotacion = rutas_muestra(labeling, tile_id)
    imagen_pool, anotacion_pool = rutas_muestra(pool, tile_id)

    # Comprobamos ambos destinos antes de mover nada para no pisar originales.
    if imagen.exists() and imagen_pool.exists():
        raise FileExistsError(imagen_pool)

    if anotacion.exists() and anotacion_pool.exists():
        raise FileExistsError(anotacion_pool)

    imagen_pool.parent.mkdir(parents=True, exist_ok=True)
    anotacion_pool.parent.mkdir(parents=True, exist_ok=True)

    if imagen.exists():
        imagen.rename(imagen_pool)

    if anotacion.exists():
        anotacion.rename(anotacion_pool)

    # LabelMe debe poder abrir la imagen desde su nueva ubicación.
    datos["imagePath"] = "../images/" + imagen_pool.name

    temporal = anotacion_pool.with_suffix(".tmp")
    temporal.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporal.replace(anotacion_pool)


def incorporar(
    labeling: Path = LABELING,
    pool: Path = POOL,
) -> tuple[int, int]:
    """Incorpora las revisadas y mantiene las incompletas o incorrectas en labeling.

    Args:
        labeling (Path): Carpeta con las imágenes que se están etiquetando.
        pool (Path): Colección global de imágenes revisadas.

    Returns:
        tuple[int, int]: Cantidad incorporada y cantidad que sigue pendiente.
    """

    pendientes = cargar_pendientes(labeling)
    manifest = cargar_manifest(pool / "manifest.csv")
    registrados = set(manifest["tile_id"])

    incorporadas = 0

    for indice, fila in pendientes.iterrows():
        tile_id = fila["tile_id"]

        # Si el pool ya se guardó antes de una interrupción, falta limpiar su registro.
        if tile_id in registrados:
            imagen, anotacion = rutas_muestra(labeling, tile_id)

            if imagen.exists() or anotacion.exists():
                raise ValueError(
                    f"ID ya incorporado con archivos duplicados: {tile_id}"
                )

            revisar_par(*rutas_muestra(pool, tile_id))
            pendientes = pendientes.drop(indice)
            continue

        imagen, anotacion = localizar_par(tile_id, labeling, pool)

        try:
            datos = revisar_par(imagen, anotacion)
        except (OSError, ValueError, TypeError, KeyError) as error:
            pendientes.loc[indice, "motivo"] = str(error)
            print(f"Pendiente {tile_id}: {error}")
            continue

        fila = actualizar_estado(fila, datos)
        mover_al_pool(tile_id, datos, labeling, pool)

        manifest.loc[len(manifest)] = fila
        guardar_manifest(manifest, pool / "manifest.csv")

        registrados.add(tile_id)
        pendientes = pendientes.drop(indice)
        incorporadas += 1

    # Las descartadas también salen de labeling, pero quedan en el pool como no_usar.
    guardar_manifest(pendientes, labeling / "manifest.csv")

    return incorporadas, len(pendientes)


def main() -> None:
    """Incorpora lo revisado y muestra lo que todavía necesita atención."""

    incorporadas, pendientes = incorporar()

    print(f"Incorporadas al pool: {incorporadas}")
    print(f"Pendientes en labeling: {pendientes}")


if __name__ == "__main__":
    main()
