"""Descarga municipal de Catastro y unión de edificios en un GeoPackage."""

import shutil
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from paneles_solares.rutas import ruta_proyecto

# Conserva la excepción SSL del acceso original a Catastro.
CONTEXTO_SSL = ssl._create_unverified_context()

URL_CATASTRO = "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/39/ES.SDGC.bu.atom_39.xml"

CARPETA_ZIPS = ruta_proyecto("data/raw/catastro/zips")
CARPETA_GMLS = ruta_proyecto("data/geografia/catastro/gml")
ARCHIVO_FINAL = ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg")

# None procesa todos los municipios.
LIMITE = None


def preparar_url(url: str) -> str:
    """Codifica el nombre del archivo para poder descargarlo.

    Args:
        url: Enlace original, que puede contener espacios, eñes o acentos.

    Returns:
        Enlace con la ruta codificada para una petición HTTP.
    """

    partes = urllib.parse.urlsplit(url.strip())
    ruta_codificada = urllib.parse.quote(urllib.parse.unquote(partes.path), safe="/")

    partes_codificadas = (
        partes.scheme,
        partes.netloc,
        ruta_codificada,
        partes.query,
        partes.fragment,
    )

    return urllib.parse.urlunsplit(partes_codificadas)


def obtener_urls_de_zips() -> list[str]:
    """Consulta el índice de Catastro y recoge los enlaces a los municipios.

    Returns:
        Enlaces a los ZIP, ordenados y sin duplicados.
    """

    with urllib.request.urlopen(URL_CATASTRO, context=CONTEXTO_SSL) as respuesta:
        contenido_xml = respuesta.read()

    raiz = ET.fromstring(contenido_xml)
    urls = []

    for elemento in raiz.iter():
        href = elemento.attrib.get("href")

        if href and href.lower().endswith(".zip"):
            urls.append(preparar_url(href))

    return sorted(set(urls))


def nombre_desde_url(url: str) -> str:
    """Obtiene el nombre del archivo situado al final de una URL.

    Args:
        url: Enlace de descarga.

    Returns:
        Nombre del archivo, con los espacios y acentos recuperados.
    """

    ruta = urllib.parse.urlsplit(url).path
    ruta_decodificada = urllib.parse.unquote(ruta)

    return Path(ruta_decodificada).name


def descargar_zip(url: str, destino: Path) -> str:
    """Descarga un municipio, salvo que ya exista un ZIP válido.

    Args:
        url: Enlace del municipio.
        destino: Ruta donde se guardará el ZIP.

    Returns:
        Texto que indica si se ha descargado o ya existía.
    """

    if destino.exists() and zipfile.is_zipfile(destino):
        return "ya existía"

    with urllib.request.urlopen(url, context=CONTEXTO_SSL) as respuesta:
        with destino.open("wb") as archivo:
            shutil.copyfileobj(respuesta, archivo)

    # No conservar una página de error del servidor como si fuera un ZIP.
    if not zipfile.is_zipfile(destino):
        destino.unlink()
        raise ValueError("El archivo descargado no es un ZIP")

    return "descargado"


def descargar_zips(urls: list[str]) -> list[Path]:
    """Descarga los municipios y continúa si alguno falla.

    Args:
        urls: Enlaces de los municipios que se quieren descargar.

    Returns:
        Rutas de los ZIP descargados o reutilizados.
    """

    CARPETA_ZIPS.mkdir(parents=True, exist_ok=True)
    zips_correctos = []

    for numero, url in enumerate(urls, start=1):
        destino = CARPETA_ZIPS / nombre_desde_url(url)

        try:
            estado = descargar_zip(url, destino)
            zips_correctos.append(destino)
            print(f"[{numero}/{len(urls)}] {destino.name}: {estado}")

        except Exception as error:
            print(f"[{numero}/{len(urls)}] ERROR: {error}")

    return zips_correctos


def extraer_edificios(zips: list[Path]) -> list[Path]:
    """Extrae las geometrías de edificios de cada municipio.

    Args:
        zips: Archivos ZIP descargados de Catastro.

    Returns:
        Rutas de los GML de edificios.
    """

    CARPETA_GMLS.mkdir(parents=True, exist_ok=True)
    gmls = []

    for archivo_zip in zips:
        with zipfile.ZipFile(archivo_zip) as comprimido:
            for nombre_interno in comprimido.namelist():
                # Catastro también incluye partes de edificios y otras construcciones.
                if not nombre_interno.lower().endswith(".building.gml"):
                    continue

                destino = CARPETA_GMLS / Path(nombre_interno).name

                if not destino.exists():
                    with comprimido.open(nombre_interno) as origen:
                        with destino.open("wb") as salida:
                            shutil.copyfileobj(origen, salida)

                gmls.append(destino)
                print(f"Extraído: {destino.name}")

    return gmls


def unir_edificios(gmls: list[Path], salida: Path = ARCHIVO_FINAL) -> None:
    """Une los municipios en un GeoPackage con las coordenadas de PNOA.

    Args:
        gmls: Archivos de edificios que se van a unir.
        salida: GeoPackage donde se guardará el conjunto.
    """

    municipios = []

    for numero, archivo_gml in enumerate(gmls, start=1):
        edificios = gpd.read_file(archivo_gml)

        # Para buscar teselas solo hacen falta las geometrías, no los datos catastrales.
        edificios = edificios.to_crs("EPSG:25830")
        edificios = edificios[["geometry"]]
        municipios.append(edificios)

        print(f"[{numero}/{len(gmls)}] {archivo_gml.name}: {len(edificios)} edificios leidos")

    if not municipios:
        raise RuntimeError("No se encontro ningun archivo de edificios")

    todos = pd.concat(municipios, ignore_index=True)
    todos = gpd.GeoDataFrame(todos, geometry="geometry", crs="EPSG:25830")

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.unlink(missing_ok=True)
    todos.to_file(salida, layer="edificios", driver="GPKG")

    print(f"\nTotal de edificios: {len(todos)}")
    print(f"Archivo creado: {salida}")


def main() -> None:
    """Descarga y prepara los edificios de Catastro de Cantabria."""

    if LIMITE is not None and LIMITE <= 0:
        raise ValueError("El límite de municipios debe ser positivo")

    print("1. Buscando los enlaces en el XML...")
    urls = obtener_urls_de_zips()

    if LIMITE is not None:
        urls = urls[:LIMITE]

    print(f"Se procesaran {len(urls)} municipios")

    print("\n2. Descargando los ZIP...")
    zips = descargar_zips(urls)

    print("\n3. Extrayendo los archivos de edificios...")
    gmls = extraer_edificios(zips)

    print("\n4. Uniendo los edificios...")
    unir_edificios(gmls, salida=ARCHIVO_FINAL)


if __name__ == "__main__":
    main()
