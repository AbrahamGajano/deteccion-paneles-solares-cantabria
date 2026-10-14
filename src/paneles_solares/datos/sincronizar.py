"""Sincroniza los datasets con las decisiones guardadas en el manifest."""

from pathlib import Path

import pandas as pd

from paneles_solares.datos.anotaciones import (
    crear_mascara_unet,
    leer_anotacion,
    texto_yolo,
)
from paneles_solares.datos.coleccion import POOL, SPLITS, cargar_manifest
from paneles_solares.rutas import (
    DATASETS,
)


def obtener_seleccion() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Obtiene las imágenes asignadas a entrenamiento, validación y test.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: Identificadores para train, val y test,
            en ese orden.
    """

    manifest = cargar_manifest()

    manifest = manifest[manifest["estado"] == "valida"]

    entreno = manifest.loc[manifest["uso"] == "train", "tile_id"]
    validacion = manifest.loc[manifest["uso"] == "val", "tile_id"]
    test = manifest.loc[manifest["uso"] == "test", "tile_id"]

    return entreno, validacion, test


def buscar_cambios(
    ids: pd.Series,
    carpeta: Path,
    extension: str,
) -> tuple[set[str], set[str]]:
    """Busca los archivos que faltan y los que sobran en una carpeta.

    Args:
        ids: Identificadores de las imágenes deseadas.
        carpeta: Carpeta cuyos archivos se comparan con los identificadores deseados.
        extension: Extensión de los archivos, incluido el punto.

    Returns:
        tuple[set[str], set[str]]: Identificadores que faltan y que sobran, en ese
            orden.
    """

    deseados = set(ids)
    existentes = {archivo.stem for archivo in carpeta.glob(f"*{extension}")}

    faltan = deseados - existentes
    sobran = existentes - deseados

    return faltan, sobran


def sincronizar_imagenes(
    dataset: Path,
    seleccion: tuple[pd.Series, pd.Series, pd.Series],
) -> None:
    """Añade las imágenes que faltan y elimina las que sobran.

    Args:
        dataset: Carpeta del dataset de destino.
        seleccion: Identificadores de train, val y test, en ese orden.

    Raises:
        FileNotFoundError: Falta una imagen original del pool.
    """

    for split, ids in zip(SPLITS, seleccion):
        carpeta = dataset / "images" / split
        carpeta.mkdir(parents=True, exist_ok=True)

        faltan, sobran = buscar_cambios(ids, carpeta, ".png")

        for tile_id in sobran:
            (carpeta / f"{tile_id}.png").unlink()

        # Enlazamos al pool sin duplicar las imágenes en disco.
        for tile_id in faltan:
            origen = POOL / "images" / f"{tile_id}.png"
            destino = carpeta / f"{tile_id}.png"

            if not origen.is_file():
                raise FileNotFoundError(f"No existe la imagen: {origen}")

            destino.hardlink_to(origen)


def sincronizar_anotaciones(
    dataset: Path,
    formato: str,
    seleccion: tuple[pd.Series, pd.Series, pd.Series],
) -> None:
    """Regenera TXT de YOLO o máscaras U-Net desde los JSON originales.

    Args:
        dataset: Carpeta del dataset de destino.
        formato: Formato de destino: yolo o unet.
        seleccion: Identificadores de train, val y test, en ese orden.
    """

    ruta = dataset / ("labels" if formato == "yolo" else "masks")
    extension = ".txt" if formato == "yolo" else ".png"

    for split, ids in zip(SPLITS, seleccion):
        carpeta = ruta / split
        carpeta.mkdir(parents=True, exist_ok=True)

        _, sobrantes = buscar_cambios(ids, carpeta, extension)

        for tile_id in sobrantes:
            (carpeta / f"{tile_id}{extension}").unlink()

        # Las regeneramos para recoger posibles cambios en los JSON.
        for tile_id in ids:
            anotacion = POOL / "annotations" / f"{tile_id}.json"
            destino = carpeta / f"{tile_id}{extension}"

            datos = leer_anotacion(anotacion)
            if formato == "yolo":
                destino.write_text(texto_yolo(datos), encoding="utf-8")
            else:
                crear_mascara_unet(datos).save(destino)


def crear_yaml_yolo(dataset: Path) -> None:
    """Crea el archivo de configuración que necesita YOLO.

    Args:
        dataset: Carpeta del dataset YOLO.
    """

    contenido = [
        f'path: "{dataset.resolve().as_posix()}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: panel_solar",
        "",
    ]

    (dataset / "dataset.yaml").write_text(
        "\n".join(contenido),
        encoding="utf-8",
    )


def sincronizar() -> None:
    """Sincroniza todos los datasets conocidos."""

    seleccion = obtener_seleccion()

    for formato in ("yolo", "unet"):
        dataset = DATASETS / formato
        dataset.mkdir(parents=True, exist_ok=True)

        sincronizar_imagenes(dataset, seleccion)
        sincronizar_anotaciones(dataset, formato, seleccion)

        if formato == "yolo":
            crear_yaml_yolo(dataset)

        print(f"Dataset sincronizado: {formato}")


if __name__ == "__main__":
    sincronizar()
