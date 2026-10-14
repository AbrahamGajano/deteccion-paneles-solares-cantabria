"""Revisa las predicciones del test y guarda los casos que interesa inspeccionar.

La clasificación es por imagen: una positiva detectada no garantiza que todas
sus máscaras coincidan con los paneles reales.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paneles_solares.modelos.metricas import calcular_metricas
from paneles_solares.rutas import (
    DATASET_UNET,
    DATASET_YOLO,
    PESOS_YOLO,
    reemplazar_carpeta,
    ruta_proyecto,
)

MODELO = PESOS_YOLO
IMAGENES = DATASET_YOLO / "images/test"
LABELS = DATASET_YOLO / "labels/test"
IMAGENES_UNET = DATASET_UNET / "images/test"
MASCARAS_REALES = DATASET_UNET / "masks/test"
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
        imagenes: Carpeta de imágenes del test.
        labels: Carpeta de etiquetas TXT del test.

    Raises:
        FileNotFoundError: Falta el TXT de una imagen del test.
        ValueError: No hay imágenes en la carpeta del test.
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
    """Exige las mismas imágenes y máscaras reales que evalúa U-Net.

    Args:
        imagenes: Carpeta de imágenes del test.

    Raises:
        ValueError: Los tests de YOLO y U-Net no contienen las mismas imágenes y
            máscaras.
    """

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
    """Compara la unión de máscaras YOLO con la máscara real de U-Net.

    Args:
        resultado: Resultado de la predicción de YOLO.
        ruta_real: PNG con la máscara real de la tesela.

    Returns:
        dict: Métricas de la unión de máscaras YOLO frente a la anotación.

    Raises:
        ValueError: Las dimensiones no coinciden o hay detecciones sin máscaras.
    """

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
    return calcular_metricas(tp, fp, fn)


def clasificar_imagen(tiene_panel: bool, tiene_deteccion: bool) -> str:
    """Compara la presencia de paneles reales con la predicción.

    Args:
        tiene_panel: Indica si la etiqueta contiene paneles.
        tiene_deteccion: Indica si el modelo ha detectado algún objeto.

    Returns:
        str: Categoría de presencia para la revisión de imágenes.
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
        resultado: Resultado de la predicción de YOLO.

    Returns:
        tuple[int, float | None]: Número de detecciones y confianza máxima, o None si no
            hay objetos.
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
        resultado: Resultado de la predicción de YOLO.
        categoria: Categoría de revisión de la imagen.
        confianza: Confianza máxima de la imagen, o None si no hay detecciones.
        salida: Carpeta donde se guardan los resultados.
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
        registros: Resultados individuales con sus categorías de revisión.
        salida: Carpeta donde se guardan los resultados.
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
    metricas = calcular_metricas(tp, fp, fn)

    pd.DataFrame([{"imagenes": len(resumen), **metricas}]).to_csv(
        salida / "resumen.csv", index=False
    )

    print("\nResultados globales")
    print(f"Imágenes: {len(resumen)}")
    print(f"Dice: {metricas['dice']:.4f}")
    print(f"IoU: {metricas['iou']:.4f}")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"Sesgo de superficie: {metricas['sesgo_superficie']:.2%}")
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
        modelo: Modelo cargado para esta fase.
        confianza: Confianza mínima para aceptar una detección de YOLO.
        imagenes: Carpeta de imágenes del test.
        labels: Carpeta de etiquetas TXT del test.
        salida: Carpeta donde se guardan los resultados.
        imgsz: Tamaño de entrada utilizado por YOLO.
        device: Dispositivo utilizado por YOLO.

    Raises:
        FileExistsError: Queda una salida incompleta de una ejecución anterior.
        ValueError: Se repiten imágenes, faltan predicciones o las etiquetas no
            coinciden.
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
    """Revisa el test y sustituye la revisión anterior cuando termina.

    Raises:
        FileNotFoundError: Falta un archivo de entrada o de pesos necesario para esta
            fase.
    """

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
