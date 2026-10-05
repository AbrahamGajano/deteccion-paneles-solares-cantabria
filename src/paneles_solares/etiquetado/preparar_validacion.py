"""Selecciona localizaciones del test y extrae únicamente sus ventanas RGB."""

import argparse
import json
import xml.etree.ElementTree as ET
from contextlib import ExitStack

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds

from paneles_solares.datos.coleccion import MANIFEST
from paneles_solares.datos.validacion import (
    EXTENSION_M, GRUPOS, cargar_campania, carpeta_campania, rutas_tesela,
)
from paneles_solares.rutas import ruta_proyecto


def seleccionar_localizaciones(config: dict) -> pd.DataFrame:
    """La muestra aleatoria se sortea antes de añadir positivos conocidos."""
    pool = pd.read_csv(MANIFEST, keep_default_na=False)
    test = pool[(pool["uso"] == "test") & (pool["estado"] == "valida")].copy()
    imagenes = ruta_proyecto("data/datasets/unet/images/test")
    test = test[test["tile_id"].isin(p.stem for p in imagenes.glob("*.png"))]
    test = test.sort_values("tile_id")
    aleatorias = test.sample(n=config["aleatorias"], random_state=config["semilla"]).copy()
    aleatorias["grupo"] = "aleatorias"
    positivas = test[test["tipo"] == "positiva"]
    positivas = positivas.sample(n=min(config["positivas"], len(positivas)), random_state=config["semilla"])
    positivas = positivas[~positivas["tile_id"].isin(aleatorias["tile_id"])].copy()
    positivas["grupo"] = "positivos"
    seleccion = pd.concat([positivas, aleatorias], ignore_index=True)
    print(f"Test disponible: {len(test)} localizaciones, {(test['tipo'] == 'positiva').sum()} positivas.")
    print(f"Selección: {len(aleatorias)} aleatorias + {len(positivas)} positivos adicionales.")
    return seleccion


def georreferenciar(seleccion: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Recupera de los TIF de 2023 la extensión exacta de cada PNG del test."""
    registros = []
    for nombre, grupo in seleccion.groupby("tif", sort=False):
        with rasterio.open(ruta_proyecto("data/pnoa") / nombre) as tif:
            if tif.crs != rasterio.crs.CRS.from_epsg(25830) or not np.allclose(tif.res, (0.15, 0.15)):
                raise ValueError(f"La referencia no es EPSG:25830 a 15 cm: {nombre}")
            for fila in grupo.to_dict("records"):
                ventana = Window(fila["columna"], fila["fila"], fila["ancho"], fila["alto"])
                xmin, ymin, xmax, ymax = tif.window_bounds(ventana)
                if not np.allclose((xmax - xmin, ymax - ymin), EXTENSION_M):
                    raise ValueError(f"Referencia incompleta: {fila['tile_id']}")
                registros.append({
                    "tile_id": fila["tile_id"], "grupo": fila["grupo"],
                    "tipo_2023": fila["tipo"], "anio_referencia": 2023,
                    "tif_referencia": nombre, "fila_referencia": fila["fila"],
                    "columna_referencia": fila["columna"], "num_edificios": fila["num_edificios"],
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                    "centro_x": (xmin + xmax) / 2, "centro_y": (ymin + ymax) / 2,
                    "crs": "EPSG:25830", "anio": config["anio"],
                    "resolucion_m": config["resolucion_m"], "extension_m": EXTENSION_M,
                    "ancho_nativo": round(EXTENSION_M / config["resolucion_m"]),
                    "alto_nativo": round(EXTENSION_M / config["resolucion_m"]),
                    "tam_modelo": 512, "procedencia": config["procedencia"],
                    "semilla": config["semilla"], "fuente": "",
                })
    return pd.DataFrame(registros).sort_values(["grupo", "tile_id"]).reset_index(drop=True)


def preparar_seleccion(nombre: str, config: dict) -> pd.DataFrame:
    """Una repetición mantiene la selección y las etiquetas existentes."""
    carpeta = carpeta_campania(nombre)
    manifest = carpeta / "manifest.csv"
    if manifest.exists():
        anterior = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
        campos = ("resolucion_m", "aleatorias", "positivas", "semilla")
        if any(anterior[campo] != config[campo] for campo in campos):
            raise ValueError("Esta campaña ya tiene otra selección u origen. Usa otro nombre de campaña para conservar sus etiquetas.")
        seleccion = pd.read_csv(manifest, keep_default_na=False)
        origen_cambia = any(anterior[campo] != config[campo] for campo in ("anio", "procedencia", "fuentes"))
        if origen_cambia:
            if any((carpeta / "images").rglob("*.png")) or any((carpeta / "annotations").rglob("*.json")):
                raise ValueError("No se puede cambiar el origen de imágenes ya extraídas. Usa otra campaña.")
            seleccion["anio"] = config["anio"]
            seleccion["procedencia"] = config["procedencia"]
            seleccion.to_csv(manifest, index=False)
            (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        return seleccion
    seleccion = georreferenciar(seleccionar_localizaciones(config), config)
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    seleccion.to_csv(manifest, index=False)
    return seleccion


def comprobar_acceso_parcial(url: str) -> None:
    """Impide que un servidor sin Range envíe una ortofoto completa."""
    with requests.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}, stream=True, timeout=30) as respuesta:
        if respuesta.status_code != 206 or not respuesta.headers.get("Content-Range", "").startswith("bytes 0-0/"):
            raise ValueError(f"La fuente no permite lecturas parciales HTTP: {url}")


def abrir_fuentes(config: dict, pila: ExitStack) -> list:
    fuentes = []
    for origen in config["fuentes"]:
        if isinstance(origen, dict):
            comprobar_wcs(origen, config["resolucion_m"])
            fuentes.append(origen)
            continue
        remoto = origen.startswith("https://") or origen.startswith("http://")
        if remoto:
            comprobar_acceso_parcial(origen)
        tif = pila.enter_context(rasterio.open(origen if remoto else ruta_proyecto(origen)))
        if tif.crs != rasterio.crs.CRS.from_epsg(25830) or not np.allclose(tif.res, config["resolucion_m"]):
            raise ValueError(f"CRS o resolución nativa incorrectos: {origen}")
        if tif.transform.b or tif.transform.d or tif.count < 3 or tif.dtypes[:3] != ("uint8",) * 3:
            raise ValueError(f"Se necesita un raster RGB de 8 bits sin rotación: {origen}")
        if remoto and (not tif.is_tiled or max(tif.block_shapes[0]) > 1024):
            raise ValueError(f"Usa un COG con bloques pequeños para evitar lecturas grandes: {origen}")
        fuentes.append(tif)
    return fuentes


def comprobar_wcs(fuente: dict, resolucion: float) -> None:
    """Comprueba la cobertura nativa, no solo el tamaño solicitado al servidor."""
    if fuente["tipo"] != "wcs":
        raise ValueError(f"Tipo de fuente desconocido: {fuente['tipo']}")
    parametros = {"SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "DescribeCoverage",
                  "COVERAGE": fuente["cobertura"]}
    respuesta = requests.get(fuente["url"], params=parametros, timeout=30)
    respuesta.raise_for_status()
    raiz = ET.fromstring(respuesta.content)
    ns = {"w": "http://www.opengis.net/wcs", "g": "http://www.opengis.net/gml"}
    etiqueta = raiz.findtext(".//w:CoverageOffering/w:label", namespaces=ns)
    rejilla = raiz.find(".//g:RectifiedGrid", ns)
    if etiqueta != fuente["etiqueta"] or rejilla is None or rejilla.attrib["srsName"] != "EPSG:25830":
        raise ValueError("La cobertura WCS no coincide con la campaña configurada")
    vectores = [np.fromstring(v.text, sep=" ") for v in rejilla.findall("g:offsetVector", ns)]
    if not np.allclose(vectores, [[resolucion, 0], [0, -resolucion]]):
        raise ValueError("La resolución nativa del WCS no coincide con la campaña")


def extraer_wcs(fila: dict, fuente: dict) -> tuple[Image.Image, str]:
    """Recibe solo una ventana GeoTIFF en memoria y convierte su RGB a 8 bits."""
    parametros = {
        "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "COVERAGE": fuente["cobertura"], "CRS": "EPSG:25830", "RESPONSE_CRS": "EPSG:25830",
        "BBOX": ",".join(str(fila[campo]) for campo in ("xmin", "ymin", "xmax", "ymax")),
        "WIDTH": fila["ancho_nativo"], "HEIGHT": fila["alto_nativo"],
        "FORMAT": "GeoTIFF", "Band": ",".join(str(banda) for banda in fuente["bandas"]),
        "INTERPOLATION": "bilinear",
    }
    with requests.get(fuente["url"], params=parametros, stream=True, timeout=60) as respuesta:
        respuesta.raise_for_status()
        contenido = bytearray()
        for bloque in respuesta.iter_content(64 * 1024):
            contenido.extend(bloque)
            if len(contenido) > 8 * 1024 * 1024:
                raise ValueError("El WCS devolvió más de 8 MB para una sola ventana")
        url = respuesta.url
    with MemoryFile(bytes(contenido)) as memoria, memoria.open() as tif:
        if (tif.width, tif.height) != (fila["ancho_nativo"], fila["alto_nativo"]):
            raise ValueError("El WCS no devolvió el tamaño de ventana solicitado")
        limites = [fila[campo] for campo in ("xmin", "ymin", "xmax", "ymax")]
        if tif.crs != rasterio.crs.CRS.from_epsg(25830) or not np.allclose(tif.bounds, limites, rtol=0, atol=0.001):
            raise ValueError("El WCS no devolvió el encuadre solicitado")
        datos = tif.read([1, 2, 3], masked=True)
        if np.ma.getmaskarray(datos).any():
            raise ValueError(f"Ventana WCS con nodata: {fila['tile_id']}")
        tipo_pixel = tif.dtypes[0]
        divisor = {"uint8": 1.0, "uint16": 257.0}.get(tipo_pixel)
        if divisor is None or tif.dtypes[:3] != (tipo_pixel,) * 3 or fuente["divisor_rgb"] != divisor:
            raise ValueError("Configura divisor_rgb=1 para RGB uint8 o 257 para RGB uint16")
        # 0..65535 -> 0..255, igual para todas las teselas; sin realce por imagen.
        rgb = np.rint(np.asarray(datos, dtype=np.float32) / fuente["divisor_rgb"]).astype(np.uint8)
    return Image.fromarray(np.moveaxis(rgb, 0, -1)), url


def extraer_ventana(fila: dict, fuentes: list) -> tuple[Image.Image, str]:
    """Lee 384 × 384 a 20 cm; mantiene el encuadre de 2023 aunque cambie la rejilla."""
    limites = (fila["xmin"], fila["ymin"], fila["xmax"], fila["ymax"])
    if len(fuentes) == 1 and isinstance(fuentes[0], dict):
        return extraer_wcs(fila, fuentes[0])
    if any(isinstance(fuente, dict) for fuente in fuentes):
        raise ValueError("Configura un único WCS por campaña; no lo mezcles con GeoTIFF")
    for tif in fuentes:
        if not (tif.bounds.left <= limites[0] and tif.bounds.bottom <= limites[1]
                and tif.bounds.right >= limites[2] and tif.bounds.top >= limites[3]):
            continue
        ventana = from_bounds(*limites, transform=tif.transform)
        datos = tif.read([1, 2, 3], window=ventana,
                         out_shape=(3, fila["alto_nativo"], fila["ancho_nativo"]),
                         resampling=Resampling.bilinear, masked=True)
        if np.ma.getmaskarray(datos).any():
            raise ValueError(f"Ventana con nodata: {fila['tile_id']}")
        return Image.fromarray(np.moveaxis(np.asarray(datos), 0, -1)), tif.name
    solapadas = [tif for tif in fuentes if tif.bounds.left < limites[2] and tif.bounds.right > limites[0]
                 and tif.bounds.bottom < limites[3] and tif.bounds.top > limites[1]]
    if solapadas:
        # En los bordes unimos solo esta ventana en memoria, nunca las hojas completas.
        resolucion = EXTENSION_M / fila["ancho_nativo"]
        datos, _ = merge(solapadas, bounds=limites, res=resolucion, indexes=[1, 2, 3],
                         dtype="uint16", nodata=256, resampling=Resampling.bilinear)
        if not np.any(datos == 256):
            rgb = np.moveaxis(np.asarray(datos).astype(np.uint8), 0, -1)
            return Image.fromarray(rgb), " | ".join(tif.name for tif in solapadas)
    raise ValueError(f"Cobertura incompleta o nodata en {fila['tile_id']}. Añade las fuentes que cubran su extensión.")


def extraer_muestra(nombre: str, config: dict, seleccion: pd.DataFrame) -> None:
    if not config["anio"] or not config["procedencia"] or not config["fuentes"]:
        raise ValueError("Completa anio, procedencia y fuentes en validaciones.json. La selección ya está guardada; no se ha descargado nada.")
    carpeta = carpeta_campania(nombre)
    for grupo in GRUPOS:
        (carpeta / "images" / grupo).mkdir(parents=True, exist_ok=True)
        (carpeta / "annotations" / grupo).mkdir(parents=True, exist_ok=True)
    with ExitStack() as pila:
        pila.enter_context(rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_CACHEMAX=64 * 1024 * 1024))
        fuentes = abrir_fuentes(config, pila)
        for indice, fila in seleccion.iterrows():
            imagen, _ = rutas_tesela(carpeta, fila)
            if imagen.exists():
                continue
            recorte, fuente = extraer_ventana(fila.to_dict(), fuentes)
            # El manifiesto se actualiza antes del PNG para poder reanudar tras una interrupción.
            seleccion.at[indice, "fuente"] = fuente
            seleccion.to_csv(carpeta / "manifest.csv", index=False)
            recorte.save(imagen)
            print(f"Guardada {imagen.name} ({fila['grupo']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campania")
    parser.add_argument("--config", default="validaciones.json")
    parser.add_argument("--solo-seleccion", action="store_true", help="Guarda coordenadas sin leer las ortofotos de destino")
    args = parser.parse_args()
    config = cargar_campania(args.campania, args.config)
    seleccion = preparar_seleccion(args.campania, config)
    if not args.solo_seleccion:
        extraer_muestra(args.campania, config, seleccion)
    print(f"Muestra de {len(seleccion)} localizaciones: {carpeta_campania(args.campania)}")


if __name__ == "__main__":
    main()
