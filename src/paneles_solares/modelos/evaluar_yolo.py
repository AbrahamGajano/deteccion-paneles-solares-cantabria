"""Revisa las predicciones del test y guarda los casos que interesa inspeccionar.

La clasificación es por imagen: una positiva detectada no garantiza que todas
sus máscaras coincidan con los paneles reales.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paneles_solares.rutas import reemplazar_carpeta, ruta_proyecto

MODELO = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")
IMAGENES = ruta_proyecto("data/datasets/yolo/images/test")
LABELS = ruta_proyecto("data/datasets/yolo/labels/test")
IMAGENES_UNET = ruta_proyecto("data/datasets/unet/images/test")
MASCARAS_REALES = ruta_proyecto("data/datasets/unet/masks/test")
SALIDA = ruta_proyecto("runs/evaluacion/cantabria/yolo11s/revision")

CONFIANZA = 0.30
IMGSZ = 512
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
        if not imagen.is_file() or imagen.suffix.lower() not in (
            ".png",
            ".jpg",
            ".jpeg",
        ):
            continue

        etiqueta = labels / f"{imagen.stem}.txt"

        # Un TXT vacío es un negativo. Un TXT ausente es un dato sin revisar.
        if not etiqueta.is_file():
            raise FileNotFoundError(f"Falta la etiqueta: {etiqueta}")

        cantidad += 1

    if cantidad == 0:
        raise ValueError(f"No hay imágenes en {imagenes}")


def comprobar_mismo_test(imagenes: Path) -> None:
    """Exige las mismas imágenes y máscaras reales que evalúa U-Net."""
    ids_yolo = {ruta.stem for ruta in imagenes.glob("*.png")}
    ids_unet = {ruta.stem for ruta in IMAGENES_UNET.glob("*.png")}
    ids_mascaras = {ruta.stem for ruta in MASCARAS_REALES.glob("*.png")}

    if not ids_yolo or ids_yolo != ids_unet or ids_yolo != ids_mascaras:
        raise ValueError(
            "Los tests de YOLO y U-Net no contienen las mismas imágenes "
            "y máscaras: "
            f"YOLO={len(ids_yolo)}, U-Net={len(ids_unet)}, "
            f"máscaras={len(ids_mascaras)}; "
            f"faltan en U-Net={len(ids_yolo - ids_unet)}, "
            f"faltan máscaras={len(ids_yolo - ids_mascaras)}"
        )


def medir_mascara(resultado, ruta_real: Path) -> dict:
    """Compara la unión de máscaras YOLO con la máscara real de U-Net."""
    with Image.open(ruta_real) as archivo:
        real = np.asarray(archivo.convert("L")) > 0

    if real.shape != tuple(resultado.orig_shape):
        raise ValueError(
            f"Imagen y máscara real tienen tamaños distintos en {resultado.path}: "
            f"imagen={resultado.orig_shape}, real={real.shape}"
        )

    if resultado.masks is None:
        if resultado.boxes is not None and len(resultado.boxes) > 0:
            raise ValueError(f"Hay detecciones sin máscaras: {resultado.path}")
        predicha = np.zeros(real.shape, dtype=bool)
    else:
        predicha = resultado.masks.data.cpu().numpy().astype(bool).any(axis=0)

    if real.shape != predicha.shape:
        raise ValueError(
            f"Tamaños de máscaras distintos en {resultado.path}: "
            f"real={real.shape}, YOLO={predicha.shape}"
        )

    tp = int(np.count_nonzero(real & predicha))
    fp = int(np.count_nonzero(~real & predicha))
    fn = int(np.count_nonzero(real & ~predicha))
    pixeles_reales = tp + fn
    pixeles_predichos = tp + fp
    diferencia_superficie = pixeles_predichos - pixeles_reales

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
        "iou": 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn),
        "precision": np.nan if tp + fp == 0 else tp / (tp + fp),
        "recall": np.nan if tp + fn == 0 else tp / (tp + fn),
        "pixeles_reales": pixeles_reales,
        "pixeles_predichos": pixeles_predichos,
        "diferencia_superficie": diferencia_superficie,
        "sesgo_superficie": (
            np.nan if pixeles_reales == 0 else diferencia_superficie / pixeles_reales
        ),
    }


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

    resumen = pd.DataFrame(registros)

    resumen.to_csv(
        salida / "resumen_predicciones.csv",
        index=False,
        encoding="utf-8",
    )

    # Igual que evaluar_unet.py: sumar píxeles antes de calcular las métricas.
    tp = int(resumen["tp"].sum())
    fp = int(resumen["fp"].sum())
    fn = int(resumen["fn"].sum())
    pixeles_reales = tp + fn
    pixeles_predichos = tp + fp
    diferencia_superficie = pixeles_predichos - pixeles_reales
    dice = 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    precision = np.nan if tp + fp == 0 else tp / (tp + fp)
    recall = np.nan if tp + fn == 0 else tp / (tp + fn)
    sesgo_superficie = (
        np.nan if pixeles_reales == 0 else diferencia_superficie / pixeles_reales
    )

    pd.DataFrame(
        [
            {
                "imagenes": len(resumen),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "pixeles_reales": pixeles_reales,
                "pixeles_predichos": pixeles_predichos,
                "diferencia_superficie": diferencia_superficie,
                "sesgo_superficie": sesgo_superficie,
            }
        ]
    ).to_csv(salida / "resumen.csv", index=False)

    print("\nResultados globales")
    print(f"Imágenes: {len(resumen)}")
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Sesgo de superficie: {sesgo_superficie:.2%}")
    print("\nClasificación de imágenes")
    for categoria in CATEGORIAS:
        cantidad = (resumen["tipo"] == categoria).sum()
        print(f"{categoria}: {cantidad}")


def guardar_predicciones(
    modelo,
    confianza: float,
    imagenes: Path,
    labels: Path,
    salida: Path,
    imgsz: int = 512,
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
    comprobar_mismo_test(imagenes)
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
    procesadas = set()

    for resultado in resultados:
        imagen = Path(resultado.path)
        if imagen.stem in procesadas:
            raise ValueError(f"Imagen repetida en predicciones: {imagen.name}")
        procesadas.add(imagen.stem)
        etiqueta = labels / f"{imagen.stem}.txt"

        tiene_panel = bool(etiqueta.read_text(encoding="utf-8").strip())
        detecciones, confianza_maxima = obtener_detecciones(resultado)
        indicadores = medir_mascara(
            resultado,
            MASCARAS_REALES / f"{imagen.stem}.png",
        )
        if tiene_panel != (indicadores["pixeles_reales"] > 0):
            raise ValueError(f"TXT y máscara real no coinciden: {imagen.name}")

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
            **indicadores,
        }
        registros.append(registro)

    esperadas = {ruta.stem for ruta in imagenes.glob("*.png")}
    if procesadas != esperadas:
        raise ValueError(
            f"Faltan predicciones para {len(esperadas - procesadas)} imágenes "
            f"y hay {len(procesadas - esperadas)} inesperadas"
        )

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
