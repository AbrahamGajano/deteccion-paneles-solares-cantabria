"""Revisa las predicciones del test y guarda los casos que interesa inspeccionar.

La clasificación es por imagen: una positiva detectada no garantiza que todas
sus máscaras coincidan con los paneles reales.
"""

from pathlib import Path

import pandas as pd

from paneles_solares.rutas import reemplazar_carpeta, ruta_proyecto

MODELO = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")
IMAGENES = ruta_proyecto("data/datasets/yolo/images/test")
LABELS = ruta_proyecto("data/datasets/yolo/labels/test")
SALIDA = ruta_proyecto("runs/evaluacion/cantabria/yolo11s/revision")

CONFIANZA = 0.30
IMGSZ = 640  # Misma escala de entrada que en el entrenamiento actual.
DEVICE = 0

CATEGORIAS = (
    "positivas_detectadas",
    "positivas_sin_deteccion",
    "posibles_falsos_positivos",
    "negativas_correctas",
)


def comprobar_etiquetas(imagenes: Path, labels: Path) -> None:
    """Comprueba que todas las imágenes del test tienen su TXT.

    Args:
        imagenes (Path): Carpeta de imágenes.
        labels (Path): Carpeta de etiquetas YOLO.
    """

    cantidad = 0

    for imagen in imagenes.iterdir():
        if not imagen.is_file() or imagen.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        etiqueta = labels / f"{imagen.stem}.txt"

        # Un TXT vacío es un negativo. Un TXT ausente es un dato sin revisar.
        if not etiqueta.is_file():
            raise FileNotFoundError(f"Falta la etiqueta: {etiqueta}")

        cantidad += 1

    if cantidad == 0:
        raise ValueError(f"No hay imágenes en {imagenes}")


def clasificar_imagen(tiene_panel: bool, tiene_deteccion: bool) -> str:
    """Compara la presencia de paneles reales con la predicción.

    Args:
        tiene_panel (bool): La anotación contiene algún panel.
        tiene_deteccion (bool): El modelo ha detectado algún panel.

    Returns:
        str: Categoría a la que pertenece la imagen.
    """

    if tiene_panel and tiene_deteccion:
        return "positivas_detectadas"

    if tiene_panel:
        return "positivas_sin_deteccion"

    if tiene_deteccion:
        return "posibles_falsos_positivos"

    return "negativas_correctas"


def obtener_detecciones(resultado) -> tuple[int, float | None]:
    """Resume las detecciones de una predicción de YOLO.

    Args:
        resultado: Predicción de una imagen.

    Returns:
        tuple[int, float | None]: Número de detecciones y confianza máxima.
    """

    if resultado.boxes is None or len(resultado.boxes) == 0:
        return 0, None

    detecciones = len(resultado.boxes)
    confianza = float(resultado.boxes.conf.max().item())

    return detecciones, confianza


def guardar_imagen(
    resultado,
    categoria: str,
    confianza: float | None,
    salida: Path,
) -> None:
    """Guarda la imagen dibujada en su categoría de revisión.

    Args:
        resultado: Predicción con la imagen y las detecciones.
        categoria (str): Categoría asignada a la imagen.
        confianza (float | None): Confianza máxima encontrada.
        salida (Path): Carpeta de la revisión.
    """

    # Los negativos correctos se incluyen en el CSV, sin copiar sus imágenes vacías.
    if categoria == "negativas_correctas":
        return

    carpeta = salida / categoria
    carpeta.mkdir(parents=True, exist_ok=True)

    nombre = Path(resultado.path).name

    if confianza is not None:
        nombre = f"{confianza:.3f}_{nombre}"

    resultado.save(filename=str(carpeta / nombre))


def guardar_resumen(registros: list[dict], salida: Path) -> None:
    """Guarda el CSV de la revisión y muestra el número de casos de cada tipo.

    Args:
        registros (list[dict]): Resultados de todas las imágenes.
        salida (Path): Carpeta de la revisión.
    """

    columnas = ["imagen", "tipo", "detecciones", "confianza_maxima"]
    resumen = pd.DataFrame(registros, columns=columnas)

    resumen.to_csv(
        salida / "resumen_predicciones.csv",
        index=False,
        encoding="utf-8",
    )

    for categoria in CATEGORIAS:
        cantidad = (resumen["tipo"] == categoria).sum()
        print(f"{categoria}: {cantidad}")


def guardar_predicciones(
    modelo,
    confianza: float,
    imagenes: Path,
    labels: Path,
    salida: Path,
    imgsz: int = 640,
    device: int | str = 0,
) -> None:
    """Revisa el test y guarda las imágenes y el resumen.

    Args:
        modelo: Modelo YOLO cargado.
        confianza (float): Umbral mínimo de detección.
        imagenes (Path): Carpeta de imágenes del test.
        labels (Path): Carpeta de etiquetas del test.
        salida (Path): Carpeta temporal para esta revisión.
        imgsz (int): Tamaño de entrada del modelo.
        device (int | str): GPU o CPU que se utilizará.
    """

    if salida.exists():
        raise FileExistsError(f"Quedó una revisión interrumpida: {salida}")

    comprobar_etiquetas(imagenes, labels)
    salida.mkdir(parents=True)

    # Procesamos una imagen cada vez para no acumular todo el test en memoria.
    resultados = modelo.predict(
        source=str(imagenes),
        imgsz=imgsz,
        conf=confianza,
        device=device,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    registros = []

    for resultado in resultados:
        imagen = Path(resultado.path)
        etiqueta = labels / f"{imagen.stem}.txt"

        tiene_panel = bool(etiqueta.read_text(encoding="utf-8").strip())
        detecciones, confianza_maxima = obtener_detecciones(resultado)

        categoria = clasificar_imagen(tiene_panel, detecciones > 0)
        guardar_imagen(resultado, categoria, confianza_maxima, salida)

        confianza_csv = ""
        if confianza_maxima is not None:
            confianza_csv = round(confianza_maxima, 4)

        registro = {
            "imagen": imagen.name,
            "tipo": categoria,
            "detecciones": detecciones,
            "confianza_maxima": confianza_csv,
        }
        registros.append(registro)

    guardar_resumen(registros, salida)


def main() -> None:
    """Revisa el test y sustituye la revisión anterior cuando termina."""

    if not MODELO.is_file():
        raise FileNotFoundError(MODELO)

    from ultralytics import YOLO

    modelo = YOLO(str(MODELO))
    temporal = SALIDA.with_name(f".{SALIDA.name}_preparando")

    guardar_predicciones(
        modelo=modelo,
        confianza=CONFIANZA,
        imagenes=IMAGENES,
        labels=LABELS,
        salida=temporal,
        imgsz=IMGSZ,
        device=DEVICE,
    )

    reemplazar_carpeta(temporal, SALIDA)
    print(f"Revisión actualizada en {SALIDA}")


if __name__ == "__main__":
    main()
