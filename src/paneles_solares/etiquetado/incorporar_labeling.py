"""Incorpora al pool las imágenes revisadas y deja en labeling lo pendiente."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from paneles_solares.datos.anotaciones import excluida, revisar_par
from paneles_solares.datos.coleccion import (
    LABELING,
    POOL,
    cargar_manifest,
    guardar_manifest,
    nueva_fila,
    rutas_muestra,
)
from paneles_solares.rutas import PNOA, huella_archivo, ruta_proyecto, ruta_registrada


def cargar_pendientes(labeling: Path) -> pd.DataFrame:
    """Carga labeling y registra los PNG que se hayan añadido manualmente.

    Args:
        labeling: Carpeta temporal de etiquetado.

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
        tile_id: Identificador de la imagen.
        labeling: Carpeta de origen.
        pool: Colección global.

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
        fila: Registro de la imagen.
        datos: Anotación revisada de LabelMe.

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
        tile_id: Identificador de la muestra.
        datos: Anotación revisada.
        labeling: Carpeta de origen.
        pool: Carpeta de destino.

    Raises:
        FileExistsError: El destino ya contiene una imagen o anotación original.
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
        labeling: Carpeta con las imágenes que se están etiquetando.
        pool: Colección global de imágenes revisadas.

    Returns:
        tuple[int, int]: Cantidad incorporada y cantidad que sigue pendiente.

    Raises:
        ValueError: Un identificador ya incorporado conserva archivos duplicados en
            labeling.
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


def zonas_reservadas(manifest: pd.DataFrame, pnoa: Path):
    """Recupera las huellas de validación y test para separar también otros años.

    Args:
        manifest: Colección con las asignaciones originales de los datasets.
        pnoa: Ortofotos de referencia que georreferencian fila y columna.

    Returns:
        gpd.GeoDataFrame: Recortes reservados en EPSG:25830.
    """

    filas = []
    reservadas = manifest[manifest.uso.isin(("val", "test"))]
    for nombre, grupo in reservadas.groupby("tif"):
        with rasterio.open(pnoa / nombre) as tif:
            for r in grupo.itertuples():
                ventana = Window(int(r.columna), int(r.fila), int(r.ancho), int(r.alto))
                filas.append(
                    {"tile_id": r.tile_id, "geometry": box(*tif.window_bounds(ventana))}
                )
    return gpd.GeoDataFrame(filas, columns=["tile_id", "geometry"], crs=25830)


def incorporar_negativos(
    manifest: Path, pool: Path = POOL, pnoa: Path = PNOA
) -> pd.DataFrame:
    """Incorpora negativos históricos confirmados al pool y separa zonas reservadas.

    Args:
        manifest: Revisión histórica finalizada, con estados y huellas de los JSON.
        pool: Colección habitual; no se crea otro dataset.
        pnoa: Ortofotos usadas para localizar las imágenes de validación y test.

    Returns:
        pd.DataFrame: Muestras incorporadas. Las que solapan zonas reservadas
            quedan en no_usar, sin alterar la composición de validación o test.
    """

    tabla = pd.read_csv(manifest, keep_default_na=False)
    coleccion = cargar_manifest(pool / "manifest.csv")
    reservadas = zonas_reservadas(coleccion, pnoa)
    registros = []
    for r in tabla[tabla.estado.eq("ausente")].itertuples():
        anotacion = manifest.parent / r.anotacion
        if (
            r.revision_completa not in (True, "True")
            or huella_archivo(anotacion) != r.anotacion_sha256
        ):
            raise ValueError(f"Revisión modificada o sin finalizar: {anotacion}")
        imagen = manifest.parent / r.imagen
        datos = revisar_par(imagen, anotacion)
        if datos["shapes"] or excluida(datos):
            raise ValueError(f"La imagen no es un negativo confirmado: {imagen}")
        tile_id = "historico_" + imagen.stem
        zona = box(r.xmin, r.ymin, r.xmax, r.ymax)
        posibles = reservadas.sindex.query(zona, predicate="intersects")
        solapes = [
            reservadas.iloc[i].tile_id
            for i in posibles
            if zona.intersection(reservadas.iloc[i].geometry).area > 1e-6
        ]
        fila = nueva_fila(
            {"tile_id": tile_id, "ancho": r.ancho, "alto": r.alto}, "historico_revisado"
        )
        fila = actualizar_estado(pd.Series(fila), datos).to_dict()
        fila.update(
            uso="no_usar" if solapes else "train",
            motivo="solape espacial con validacion/test"
            if solapes
            else "negativo historico confirmado",
            anio=r.anio,
            ventana=r.ventana,
            xmin=r.xmin,
            ymin=r.ymin,
            xmax=r.xmax,
            ymax=r.ymax,
            crs=r.crs,
            revision_origen=ruta_registrada(manifest),
            zonas_reservadas=";".join(sorted(solapes)),
        )
        destino_imagen, destino_json = rutas_muestra(pool, tile_id)
        destino_imagen.parent.mkdir(parents=True, exist_ok=True)
        destino_json.parent.mkdir(parents=True, exist_ok=True)
        if not destino_imagen.exists():
            shutil.copy2(imagen, destino_imagen)
        datos["imagePath"] = "../images/" + destino_imagen.name
        destino_json.write_text(
            json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        registros.append(fila)

    nuevas = pd.DataFrame(registros)
    if not nuevas.empty:
        coleccion = coleccion[~coleccion.tile_id.isin(nuevas.tile_id)]
        coleccion = pd.concat([coleccion, nuevas], ignore_index=True).fillna("")
        guardar_manifest(coleccion, pool / "manifest.csv")
    return nuevas


def main() -> None:
    """Incorpora lo revisado y muestra lo que todavía necesita atención."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historico", type=Path, help="Manifiesto finalizado: incorporar sus negativos"
    )
    args = parser.parse_args()
    if args.historico:
        nuevas = incorporar_negativos(ruta_proyecto(args.historico))
        print(
            nuevas.uso.value_counts().to_dict()
            if not nuevas.empty
            else "Sin negativos nuevos"
        )
        return

    incorporadas, pendientes = incorporar()

    print(f"Incorporadas al pool: {incorporadas}")
    print(f"Pendientes en labeling: {pendientes}")


if __name__ == "__main__":
    main()
