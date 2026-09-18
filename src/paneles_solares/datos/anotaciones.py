"""Lectura y conversión de las anotaciones de LabelMe."""

import json
from pathlib import Path

from PIL import Image

CLASE = "panel_solar"


def leer_anotacion(ruta: Path) -> dict:
    """Lee y comprueba una anotación de LabelMe.

    Args:
        ruta (Path): Ruta del archivo JSON.

    Returns:
        dict: Datos de la anotación.
    """

    datos = json.loads(ruta.read_text(encoding="utf-8"))

    campos = ("imageWidth", "imageHeight", "shapes")

    if not isinstance(datos, dict) or any(campo not in datos for campo in campos):
        raise ValueError(f"Anotación incompleta: {ruta}")

    # Las descartadas se conservan en el pool, sin utilizar sus polígonos.
    if excluida(datos):
        return datos

    for forma in datos["shapes"]:
        if forma.get("label") != CLASE:
            raise ValueError(f"Clase desconocida en {ruta}: {forma.get('label')}")

        if forma.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"La forma debe ser un polígono: {ruta}")

        if len(forma.get("points", [])) < 3:
            raise ValueError(f"Polígono incompleto: {ruta}")

        # YOLO necesita coordenadas dentro de la imagen para normalizarlas a 0–1.
        for x, y in forma["points"]:
            if not (0 <= x <= datos["imageWidth"] and 0 <= y <= datos["imageHeight"]):
                raise ValueError(f"Coordenadas fuera de la imagen: {ruta}")

    return datos


def excluida(datos: dict) -> bool:
    """Comprueba si una anotación se marcó para excluir.

    Args:
        datos (dict): Datos obtenidos del JSON de LabelMe.

    Returns:
        bool: True si la imagen es dudosa o se debe descartar.
    """

    flags = datos.get("flags") or {}

    return flags.get("dudosa", False) or flags.get("descartar", False)


def revisar_par(imagen: Path, anotacion: Path) -> dict:
    """Comprueba que una imagen coincide con su anotación.

    Args:
        imagen (Path): Ruta de la imagen PNG.
        anotacion (Path): Ruta del JSON de LabelMe.

    Returns:
        dict: Datos de la anotación validada.
    """

    datos = leer_anotacion(anotacion)

    with Image.open(imagen) as img:
        dimensiones_json = (
            datos["imageWidth"],
            datos["imageHeight"],
        )

        if img.size != dimensiones_json:
            raise ValueError(f"Dimensiones distintas entre PNG y JSON: {imagen.name}")

    return datos


def texto_yolo(datos: dict) -> str:
    """Convierte los polígonos de LabelMe al formato YOLO.

    Args:
        datos (dict): Datos de la anotación de LabelMe.

    Returns:
        str: Contenido que se guardará en el archivo TXT.
    """

    ancho = datos["imageWidth"]
    alto = datos["imageHeight"]
    lineas = []

    for forma in datos["shapes"]:
        coordenadas = []

        for x, y in forma["points"]:
            coordenadas.append(f"{x / ancho:.6f}")
            coordenadas.append(f"{y / alto:.6f}")

        lineas.append("0 " + " ".join(coordenadas))

    return "\n".join(lineas)
