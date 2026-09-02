import shutil
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Ignoro la comprobacion del certificado porque daba error con Catastro
CONTEXTO_SSL = ssl._create_unverified_context()

# XML donde aparecen los enlaces de todos los municipios de Cantabria
URL_CATASTRO = (
    "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/39/ES.SDGC.bu.atom_39.xml"
)

# Defino las carpetas donde guardare los archivos
CARPETA_ZIPS = Path("data/raw/catastro/zips")
CARPETA_GMLS = Path("data/processed/catastro/gml")
ARCHIVO_FINAL = Path("data/processed/catastro/edificios_cantabria.gpkg")

# Si es None procesa todos los municipios. Para probar puedo poner 2, por ejemplo
LIMITE = None


def preparar_url(url: str) -> str:
    """Funcion que prepara una URL para que acepte espacios, enes y acentos.

    Args:
        url (str): La URL que quiero preparar.

    Returns:
        str: La URL correctamente codificada.
    """
    # Separo la URL en sus distintas partes
    partes = urllib.parse.urlsplit(url.strip())

    # Codifico solamente la ruta para no modificar https ni las barras
    ruta_codificada = urllib.parse.quote(urllib.parse.unquote(partes.path), safe="/")

    # Vuelvo a unir todas las partes de la URL
    return urllib.parse.urlunsplit(
        (
            partes.scheme,
            partes.netloc,
            ruta_codificada,
            partes.query,
            partes.fragment,
        )
    )


def obtener_urls_de_zips() -> list[str]:
    """Funcion que descarga el XML y obtiene los enlaces de los ZIP.

    Returns:
        list[str]: Lista con las URLs de los ZIP de los municipios.
    """
    # Descargo el contenido del XML de Catastro
    with urllib.request.urlopen(URL_CATASTRO, context=CONTEXTO_SSL) as respuesta:
        contenido_xml = respuesta.read()

    # Convierto el contenido XML en un arbol que pueda recorrer
    raiz = ET.fromstring(contenido_xml)
    urls = []

    # Recorro todos los elementos y guardo los enlaces que terminan en .zip
    for elemento in raiz.iter():
        href = elemento.attrib.get("href")

        if href and href.lower().endswith(".zip"):
            urls.append(preparar_url(href))

    # Elimino posibles repetidos y ordeno las URLs
    return sorted(set(urls))


def nombre_desde_url(url: str) -> str:
    """Funcion que obtiene el nombre del archivo situado al final de una URL.

    Args:
        url (str): URL del ZIP.

    Returns:
        str: Nombre del archivo ZIP.
    """
    ruta = urllib.parse.urlsplit(url).path
    return Path(urllib.parse.unquote(ruta)).name


def descargar_zips(urls: list[str]) -> list[Path]:
    """Funcion que descarga los ZIP de los municipios de Cantabria.

    Args:
        urls (list[str]): Lista con las URLs de los ZIP.

    Returns:
        list[Path]: Rutas de todos los ZIP descargados correctamente.
    """
    # Creo la carpeta de los ZIP en caso de que no exista
    CARPETA_ZIPS.mkdir(parents=True, exist_ok=True)
    zips_correctos = []

    for numero, url in enumerate(urls, start=1):
        destino = CARPETA_ZIPS / nombre_desde_url(url)

        try:
            # Si el ZIP ya existe y es correcto no lo vuelvo a descargar
            if destino.exists() and zipfile.is_zipfile(destino):
                estado = "ya existia"
            else:
                # Abro la descarga y copio los datos al archivo del ordenador
                with urllib.request.urlopen(url, context=CONTEXTO_SSL) as respuesta:
                    with destino.open("wb") as archivo:
                        shutil.copyfileobj(respuesta, archivo)

                # Compruebo que el archivo descargado sea realmente un ZIP
                if not zipfile.is_zipfile(destino):
                    destino.unlink(missing_ok=True)
                    raise ValueError("el archivo descargado no es un ZIP")

                estado = "descargado"

            zips_correctos.append(destino)
            print(f"[{numero}/{len(urls)}] {destino.name}: {estado}")

        # Si un municipio falla muestro el error y continuo con el siguiente
        except Exception as error:
            print(f"[{numero}/{len(urls)}] ERROR: {error}")

    return zips_correctos


def extraer_edificios(zips: list[Path]) -> list[Path]:
    """Funcion que extrae de cada ZIP solamente el archivo de edificios.

    Args:
        zips (list[Path]): Rutas de los ZIP descargados.

    Returns:
        list[Path]: Rutas de los archivos terminados en .building.gml.
    """
    # Creo la carpeta donde guardare los GML
    CARPETA_GMLS.mkdir(parents=True, exist_ok=True)
    gmls = []

    for archivo_zip in zips:
        with zipfile.ZipFile(archivo_zip) as comprimido:
            # Recorro todos los archivos que hay dentro del ZIP
            for nombre_interno in comprimido.namelist():
                # Solo necesito building, no buildingpart ni otherconstruction
                if nombre_interno.lower().endswith(".building.gml"):
                    destino = CARPETA_GMLS / Path(nombre_interno).name

                    # Si ya estaba extraido no lo vuelvo a copiar
                    if not destino.exists():
                        with comprimido.open(nombre_interno) as origen:
                            with destino.open("wb") as salida:
                                shutil.copyfileobj(origen, salida)

                    gmls.append(destino)
                    print(f"Extraido: {destino.name}")

    return gmls


def unir_edificios(gmls: list[Path]) -> None:
    """Funcion que une los edificios y los guarda en un unico GeoPackage.

    Args:
        gmls (list[Path]): Rutas de los archivos de edificios de los municipios.

    Raises:
        RuntimeError: En caso de no encontrar ningun archivo de edificios.
    """
    municipios = []

    for numero, archivo_gml in enumerate(gmls, start=1):
        # Leo los edificios del municipio
        edificios = gpd.read_file(archivo_gml)

        # Transformo las coordenadas para que coincidan con las del PNOA
        edificios = edificios.to_crs("EPSG:25830")

        # Solo necesito la geometria de cada edificio
        edificios = edificios[["geometry"]]
        municipios.append(edificios)

        print(
            f"[{numero}/{len(gmls)}] {archivo_gml.name}: "
            f"{len(edificios)} edificios leidos"
        )

    if not municipios:
        raise RuntimeError("No se encontro ningun archivo de edificios")

    # Uno todos los municipios en un solo GeoDataFrame
    todos = gpd.GeoDataFrame(
        pd.concat(municipios, ignore_index=True),
        geometry="geometry",
        crs="EPSG:25830",
    )

    # Creo la carpeta y guardo el resultado final
    ARCHIVO_FINAL.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_FINAL.unlink(missing_ok=True)
    todos.to_file(ARCHIVO_FINAL, layer="edificios", driver="GPKG")

    print(f"\nTotal de edificios: {len(todos)}")
    print(f"Archivo creado: {ARCHIVO_FINAL}")


def main() -> None:
    """Programa que descarga y prepara los edificios de Catastro de Cantabria."""
    print("1. Buscando los enlaces en el XML...")
    urls = obtener_urls_de_zips()

    # Para hacer pruebas puedo limitar el numero de municipios
    if LIMITE is not None:
        urls = urls[:LIMITE]

    print(f"Se procesaran {len(urls)} municipios")

    print("\n2. Descargando los ZIP...")
    zips = descargar_zips(urls)

    print("\n3. Extrayendo los archivos de edificios...")
    gmls = extraer_edificios(zips)

    print("\n4. Uniendo los edificios...")
    unir_edificios(gmls)


if __name__ == "__main__":
    main()
