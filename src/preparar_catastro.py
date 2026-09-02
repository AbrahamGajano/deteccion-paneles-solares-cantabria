"""Descarga y prepara los edificios de Catastro de toda Cantabria:

1. Lee el índice XML/ATOM oficial de Catastro.
2. Encuentra dentro del XML todos los enlaces que terminan en .zip.
3. Descarga los ZIP municipales, sin repetir los que ya existen.
4. Extrae los archivos GML que contienen la cartografía.
5. Conserva la capa Building y descarta BuildingPart.
6. Une todos los edificios en edificios_cantabria.gpkg.
"""

import argparse
import csv
import os
import re
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


# Url del XML del catastro de cantabria
FEED_CANTABRIA = (
    "https://www.catastro.hacienda.gob.es/INSPIRE/buildings/39/"
    "ES.SDGC.bu.atom_39.xml"
)


def preparar_url(url: str) -> str:
    """Codifica y prepara el str extraido del xml a una url valida.
    """
    partes = urllib.parse.urlsplit(url.strip())
    ruta_codificada = urllib.parse.quote(
        urllib.parse.unquote(partes.path), safe="/%"
    )
    return urllib.parse.urlunsplit(
        (partes.scheme, partes.netloc, ruta_codificada, partes.query, partes.fragment)
    )


def descargar_xml(url: str) -> bytes:
    """Descarga el XML y devuelve su contenido."""
    peticion = urllib.request.Request(
        preparar_url(url),
        headers={"User-Agent": "Proyecto-paneles-solares-Cantabria/1.0"},
    )
    with urllib.request.urlopen(peticion, timeout=120) as respuesta:
        return respuesta.read()


def encontrar_zips_en_xml(contenido_xml: bytes) -> list[str]:
    """Extrae del XML todos los atributos ``href`` que apuntan a un ZIP.
    """
    raiz = ET.fromstring(contenido_xml)
    enlaces = set()

    for elemento in raiz.iter():
        href = elemento.attrib.get("href")
        if href and href.lower().split("?")[0].endswith(".zip"):
            enlaces.add(preparar_url(href))

    return sorted(enlaces)


def codigo_municipio(texto: str) -> str:
    """Obtiene un código como 39061 a partir del nombre o URL de un archivo."""
    coincidencia = re.search(r"BU\.(\d{5})\.zip", texto, flags=re.IGNORECASE)
    return coincidencia.group(1) if coincidencia else "desconocido"


def nombre_desde_url(url: str) -> str:
    """Devuelve el nombre final del archivo indicado por una URL."""
    ruta = urllib.parse.urlsplit(url).path
    return Path(urllib.parse.unquote(ruta)).name


def descargar_zip(url: str, destino: Path) -> str:
    """Descarga un ZIP de forma segura y permite reanudar el programa.

    Primero se escribe un archivo terminado en ``.part``. Solo cuando la descarga
    acaba correctamente se renombra al nombre definitivo. Así, un corte de red no
    deja un ZIP incompleto que parezca válido.
    """
    if destino.exists() and zipfile.is_zipfile(destino):
        return "ya_existia"

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".part")

    peticion = urllib.request.Request(
        preparar_url(url),
        headers={"User-Agent": "Proyecto-paneles-solares-Cantabria/1.0"},
    )
    try:
        with urllib.request.urlopen(peticion, timeout=180) as respuesta:
            with temporal.open("wb") as archivo:
                shutil.copyfileobj(respuesta, archivo, length=1024 * 1024)

        if not zipfile.is_zipfile(temporal):
            raise ValueError("La respuesta descargada no es un ZIP válido")

        os.replace(temporal, destino)
        return "descargado"
    finally:
        # Si hubo un error, eliminamos únicamente el archivo temporal incompleto.
        if temporal.exists():
            temporal.unlink()


def guardar_manifest(registros: list[dict[str, object]], ruta: Path) -> None:
    """Guarda el estado de las descargas en un CSV fácil de inspeccionar."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = ["codigo", "archivo", "url", "estado", "tamano_bytes", "error"]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(registros)


def descargar_todos_los_zips(
    urls: list[str], directorio: Path, manifest: Path
) -> list[Path]:
    """Descarga los municipios y actualiza el manifest después de cada intento."""
    registros: list[dict[str, object]] = []
    zips_validos: list[Path] = []
    total = len(urls)

    for numero, url in enumerate(urls, start=1):
        nombre = nombre_desde_url(url)
        destino = directorio / nombre
        registro: dict[str, object] = {
            "codigo": codigo_municipio(nombre),
            "archivo": nombre,
            "url": url,
            "estado": "pendiente",
            "tamano_bytes": 0,
            "error": "",
        }

        try:
            estado = descargar_zip(url, destino)
            registro["estado"] = estado
            registro["tamano_bytes"] = destino.stat().st_size
            zips_validos.append(destino)
            print(f"[{numero}/{total}] {nombre}: {estado}")
        except Exception as error:  # Continuamos aunque falle un municipio.
            registro["estado"] = "error"
            registro["error"] = str(error)
            print(f"[{numero}/{total}] {nombre}: ERROR - {error}")

        registros.append(registro)
        guardar_manifest(registros, manifest)

    return zips_validos


def extraer_gmls(zip_path: Path, directorio_gml: Path) -> list[Path]:
    """Extrae únicamente ``*.building.gml`` del ZIP municipal.

    Catastro entrega tres geometrías diferentes: ``building`` (edificios),
    ``buildingpart`` (partes de edificios) y ``otherconstruction`` (otras
    construcciones). Para recortar tejados solo necesitamos la primera.
    """
    codigo = codigo_municipio(zip_path.name)
    destino_municipio = directorio_gml / codigo
    destino_municipio.mkdir(parents=True, exist_ok=True)
    extraidos: list[Path] = []

    with zipfile.ZipFile(zip_path) as comprimido:
        miembros = [
            miembro
            for miembro in comprimido.infolist()
            if not miembro.is_dir()
            and miembro.filename.lower().endswith(".building.gml")
        ]

        for miembro in miembros:
            # Usamos solo el nombre final para impedir que el ZIP escriba fuera
            # del directorio previsto mediante rutas como ../../archivo.
            destino = destino_municipio / Path(miembro.filename).name
            if not destino.exists():
                with comprimido.open(miembro) as origen, destino.open("wb") as salida:
                    shutil.copyfileobj(origen, salida)
            extraidos.append(destino)

    return extraidos


def capas_building(gml_path: Path) -> list[str | None]:
    """Localiza la capa Building y evita la capa BuildingPart.

    Un mismo GML puede contener varias capas. ``Building`` representa el edificio
    completo; ``BuildingPart`` son sus partes y podría provocar duplicados.
    """
    capas = gpd.list_layers(gml_path)
    nombres = capas["name"].astype(str).tolist()
    seleccionadas = [
        nombre
        for nombre in nombres
        if "building" in nombre.lower() and "buildingpart" not in nombre.lower()
    ]

    if seleccionadas:
        return seleccionadas

    # Algunos GML sencillos solo tienen una capa con un nombre no estándar.
    if len(nombres) == 1 and "buildingpart" not in nombres[0].lower():
        return [nombres[0]]

    return []


def convertir_a_multipoligono(geometria):
    """Homogeneiza Polygon y MultiPolygon para poder unir municipios."""
    if isinstance(geometria, Polygon):
        return MultiPolygon([geometria])
    if isinstance(geometria, MultiPolygon):
        return geometria
    return None


def unir_edificios(zips: list[Path], directorio_gml: Path, salida: Path) -> None:
    """Une secuencialmente los edificios en un único GeoPackage.

    Se escribe municipio a municipio para no guardar toda Cantabria a la vez en
    la memoria RAM. El resultado se crea primero como archivo temporal; el archivo
    definitivo solo se sustituye cuando el proceso ha terminado correctamente.
    """
    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = salida.with_name(f"{salida.stem}.temporal{salida.suffix}")
    if temporal.exists():
        temporal.unlink()

    primera_escritura = True
    total_edificios = 0
    municipios_procesados = 0

    for numero, zip_path in enumerate(sorted(zips), start=1):
        codigo = codigo_municipio(zip_path.name)
        gmls = extraer_gmls(zip_path, directorio_gml)
        edificios_municipio = 0

        for gml_path in gmls:
            for capa in capas_building(gml_path):
                datos = gpd.read_file(gml_path, layer=capa)
                if datos.empty:
                    continue
                if datos.crs is None:
                    raise ValueError(f"{gml_path.name} no tiene CRS definido")

                # EPSG:25830 es ETRS89 / UTM zona 30N, el CRS habitual del PNOA
                # en Cantabria. Así Catastro y las ortofotos usan metros.
                datos = datos.to_crs("EPSG:25830")
                geometrias = datos.geometry.map(convertir_a_multipoligono)
                validas = geometrias.notna() & ~geometrias.is_empty

                bloque = gpd.GeoDataFrame(
                    {
                        "codigo": codigo,
                        "archivo": gml_path.name,
                        "geometry": geometrias[validas],
                    },
                    geometry="geometry",
                    crs="EPSG:25830",
                )
                if bloque.empty:
                    continue

                modo = "w" if primera_escritura else "a"
                bloque.to_file(
                    temporal,
                    layer="edificios",
                    driver="GPKG",
                    mode=modo,
                    engine="pyogrio",
                )
                primera_escritura = False
                edificios_municipio += len(bloque)
                total_edificios += len(bloque)

        if edificios_municipio:
            municipios_procesados += 1
        print(
            f"[{numero}/{len(zips)}] {codigo}: "
            f"{edificios_municipio:,} edificios añadidos"
        )

    if primera_escritura:
        raise RuntimeError("No se encontró ninguna geometría de edificios válida")

    os.replace(temporal, salida)
    print(f"Municipios procesados: {municipios_procesados}")
    print(f"Edificios guardados: {total_edificios:,}")
    print(f"GeoPackage final: {salida}")


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga y prepara los edificios de Catastro de Cantabria."
    )
    parser.add_argument("--feed-url", default=FEED_CANTABRIA)
    parser.add_argument(
        "--zip-dir", type=Path, default=Path("data/raw/catastro/zips")
    )
    parser.add_argument(
        "--gml-dir", type=Path, default=Path("data/processed/catastro/gml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/catastro/edificios_cantabria.gpkg"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/catastro/descargas.csv"),
    )
    parser.add_argument(
        "--limite",
        type=int,
        help="Procesa solo los primeros N municipios; útil para probar.",
    )
    parser.add_argument(
        "--solo-descargar",
        action="store_true",
        help="Descarga los ZIP pero todavía no crea el GeoPackage.",
    )
    return parser.parse_args()


def main() -> None:
    args = argumentos()

    print("1/3 Leyendo el índice XML de Catastro...")
    contenido_xml = descargar_xml(args.feed_url)
    urls = encontrar_zips_en_xml(contenido_xml)
    if not urls:
        raise RuntimeError("El XML no contiene enlaces ZIP")
    if args.limite is not None:
        if args.limite <= 0:
            raise ValueError("--limite debe ser mayor que cero")
        urls = urls[: args.limite]
    print(f"Enlaces ZIP encontrados: {len(urls)}")

    print("2/3 Descargando municipios...")
    zips = descargar_todos_los_zips(urls, args.zip_dir, args.manifest)
    if len(zips) != len(urls):
        print(
            "AVISO: alguna descarga falló. Puedes ejecutar el programa otra vez; "
            "los ZIP correctos no se repetirán."
        )

    if args.solo_descargar:
        print("Descarga terminada. No se ha solicitado crear el GeoPackage.")
        return
    if not zips:
        raise RuntimeError("No hay ningún ZIP válido para procesar")

    print("3/3 Extrayendo y uniendo edificios...")
    unir_edificios(zips, args.gml_dir, args.output)


if __name__ == "__main__":
    main()