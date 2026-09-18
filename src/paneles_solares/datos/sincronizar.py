"""Sincroniza los datasets con las decisiones guardadas en el manifest."""

from pathlib import Path

import pandas as pd

from paneles_solares.datos.anotaciones import excluida, revisar_par, texto_yolo
from paneles_solares.datos.coleccion import POOL, SPLITS, cargar_manifest
from paneles_solares.rutas import ruta_proyecto

DATASETS = ruta_proyecto("data/datasets")


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

        # Quitamos las imágenes que ya no aparecen en este split.
        for tile_id in sobran:
            (carpeta / f"{tile_id}.png").unlink()

        # Enlazamos al pool sin duplicar los píxeles en disco.
        for tile_id in faltan:
            origen = POOL / "images" / f"{tile_id}.png"
            destino = carpeta / f"{tile_id}.png"

            destino.hardlink_to(origen)


def sincronizar_etiquetas_yolo(
    dataset: Path,
    seleccion: tuple[pd.Series, pd.Series, pd.Series],
) -> None:
    """Genera las etiquetas que necesita YOLO.

    Args:
        dataset (Path): Ruta del dataset YOLO.
        seleccion (tuple[pd.Series, pd.Series, pd.Series]):
            IDs de train, val y test.
    """

    for split, ids in zip(SPLITS, seleccion):
        carpeta = dataset / "labels" / split
        carpeta.mkdir(parents=True, exist_ok=True)

        # YOLO debe releer los TXT aunque una corrección no cambie su tamaño en bytes.
        carpeta.with_suffix(".cache").unlink(missing_ok=True)

        _, sobran = buscar_cambios(ids, carpeta, ".txt")

        # Eliminamos las etiquetas que ya no pertenecen a este split.
        for tile_id in sobran:
            (carpeta / f"{tile_id}.txt").unlink()

        # Las regeneramos todas por si se corrigió algún JSON.
        for tile_id in ids:
            imagen = POOL / "images" / f"{tile_id}.png"
            anotacion = POOL / "annotations" / f"{tile_id}.json"

            datos = revisar_par(imagen, anotacion)

            destino = carpeta / f"{tile_id}.txt"
            destino.write_text(
                texto_yolo(datos),
                encoding="utf-8",
            )


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


def sincronizar_yolo(
    dataset: Path,
    seleccion: tuple[pd.Series, pd.Series, pd.Series],
) -> None:
    """Sincroniza la parte específica del dataset YOLO.

    Args:
        dataset (Path): Ruta del dataset YOLO.
        seleccion (tuple[pd.Series, pd.Series, pd.Series]):
            IDs de train, val y test.
    """

    sincronizar_etiquetas_yolo(dataset, seleccion)
    crear_yaml_yolo(dataset)


def sincronizar() -> None:
    """Sincroniza todos los datasets conocidos."""

    seleccion = obtener_seleccion()

    # Comprobamos los originales antes de quitar o modificar archivos del dataset.
    for ids in seleccion:
        for tile_id in ids:
            imagen = POOL / "images" / f"{tile_id}.png"
            anotacion = POOL / "annotations" / f"{tile_id}.json"
            datos = revisar_par(imagen, anotacion)

            if excluida(datos):
                raise ValueError(f"Muestra descartada aún seleccionada: {tile_id}")

    # Cada dataset tiene una forma distinta de representar sus anotaciones.
    sincronizadores = {
        "yolo": sincronizar_yolo,
        # "unet": sincronizar_unet,
    }

    # Buscamos automáticamente los datasets que existan.
    for dataset in DATASETS.glob("*"):
        if not dataset.is_dir():
            continue

        sincronizador = sincronizadores.get(dataset.name)

        if sincronizador is None:
            print(f"Dataset ignorado porque su formato no está definido: {dataset.name}")
            continue

        # Las imágenes tienen la misma organización en todos los formatos.
        sincronizar_imagenes(dataset, seleccion)

        # Después generamos los archivos específicos del modelo.
        sincronizador(dataset, seleccion)

        print(f"Dataset sincronizado: {dataset}")


def main() -> None:
    """Ejecuta la sincronización de los datasets."""

    sincronizar()


if __name__ == "__main__":
    main()
