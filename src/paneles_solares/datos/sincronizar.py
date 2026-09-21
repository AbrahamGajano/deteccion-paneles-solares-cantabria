"""Sincroniza los datasets con las decisiones guardadas en el manifest."""

from pathlib import Path

import pandas as pd

from paneles_solares.datos.anotaciones import (
    crear_mascara_unet,
    leer_anotacion,
    texto_yolo,
)
from paneles_solares.datos.coleccion import POOL, SPLITS, cargar_manifest
from paneles_solares.rutas import ruta_proyecto

DATASETS = ruta_proyecto("data/datasets")


def exportar_yolo(datos: dict, destino: Path) -> None:
    """Pasa una anotación de LabelMe al formato de segmentación de YOLO.

    Args:
        datos (dict): Datos de la anotación.
        destino (Path): Ruta del archivo TXT de destino.
    """
    destino.write_text(
        texto_yolo(datos),
        encoding="utf-8",
    )


def exportar_unet(datos: dict, destino: Path) -> None:
    """Pasa una anotación de LabelMe a una máscara para U-Net.

    Args:
        datos (dict): Datos de la anotación.
        destino (Path): Ruta de la máscara PNG de destino.
    """
    mascara = crear_mascara_unet(datos)
    mascara.save(destino)


def obtener_seleccion() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Obtiene las imágenes asignadas a entrenamiento, validación y test.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: IDs de train, val y test.
    """
    # Cargamos el manifest que guarda el estado de cada imagen.
    manifest = cargar_manifest()

    # Nos quedamos solo con las imágenes válidas.
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
        ids (pd.Series): IDs que deberían existir.
        carpeta (Path): Carpeta que se quiere comprobar.
        extension (str): Extensión de los archivos.

    Returns:
        tuple[set[str], set[str]]: IDs que faltan y que sobran.
    """
    # Con conjuntos podemos calcular fácilmente las diferencias.
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
        dataset (Path): Ruta del dataset que se quiere sincronizar.
        seleccion (tuple[pd.Series, pd.Series, pd.Series]):
            IDs de train, val y test.
    """
    # Repetimos el proceso para cada división del dataset.
    for split, ids in zip(SPLITS, seleccion):
        carpeta = dataset / "images" / split
        carpeta.mkdir(parents=True, exist_ok=True)

        faltan, sobran = buscar_cambios(ids, carpeta, ".png")

        # Quitamos las imágenes que ya no pertenecen a este split.
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
    modelo: dict,
    seleccion: tuple[pd.Series, pd.Series, pd.Series],
) -> None:
    """Sincroniza las etiquetas o máscaras de un formato.

    Args:
        dataset (Path): Ruta principal del dataset.
        modelo (dict): Configuración del formato que se quiere generar.
        seleccion (tuple[pd.Series, pd.Series, pd.Series]):
            IDs de train, val y test.
    """
    ruta = dataset / modelo["carpeta"]
    extension = modelo["extension"]
    exportar = modelo["exportar"]

    for split, ids in zip(SPLITS, seleccion):
        carpeta = ruta / split
        carpeta.mkdir(parents=True, exist_ok=True)

        _, sobrantes = buscar_cambios(ids, carpeta, extension)

        # Eliminamos las anotaciones que ya no pertenecen al split.
        for tile_id in sobrantes:
            (carpeta / f"{tile_id}{extension}").unlink()

        # Las regeneramos para recoger posibles cambios en los JSON.
        for tile_id in ids:
            anotacion = POOL / "annotations" / f"{tile_id}.json"
            destino = carpeta / f"{tile_id}{extension}"

            datos = leer_anotacion(anotacion)
            exportar(datos, destino)


def crear_yaml_yolo(dataset: Path) -> None:
    """Crea el archivo de configuración que necesita YOLO.

    Args:
        dataset (Path): Ruta del dataset YOLO.
    """
    # Ultralytics utiliza este archivo para localizar los tres splits.
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


# Cada formato define únicamente lo que cambia respecto a los demás.
SINCRONIZADORES = [
    {
        "nombre": "yolo",
        "carpeta": "labels",
        "extension": ".txt",
        "exportar": exportar_yolo,
        "finalizar": crear_yaml_yolo,
    },
    {
        "nombre": "unet",
        "carpeta": "masks",
        "extension": ".png",
        "exportar": exportar_unet,
        "finalizar": None,
    },
]


def sincronizar() -> None:
    """Sincroniza todos los datasets conocidos."""
    seleccion = obtener_seleccion()

    for modelo in SINCRONIZADORES:
        dataset = DATASETS / modelo["nombre"]
        dataset.mkdir(parents=True, exist_ok=True)

        sincronizar_imagenes(dataset, seleccion)
        sincronizar_anotaciones(dataset, modelo, seleccion)

        # Algunos formatos necesitan archivos adicionales.
        finalizar = modelo["finalizar"]

        if finalizar is not None:
            finalizar(dataset)

        print(f"Dataset sincronizado: {modelo['nombre']}")


def main() -> None:
    """Ejecuta la sincronización de los datasets."""
    sincronizar()


if __name__ == "__main__":
    main()
