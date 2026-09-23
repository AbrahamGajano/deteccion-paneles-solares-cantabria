"""Evalúa el modelo U-Net sobre el conjunto de test."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from paneles_solares.modelos.entrenar_unet import DatasetPaneles, crear_modelo
from paneles_solares.rutas import ruta_proyecto

DATASET = ruta_proyecto("data/datasets/unet")

PESOS = ruta_proyecto("weights/unet_resnet34.pt")

SALIDA = ruta_proyecto("runs/evaluacion/unet")

UMBRAL = 0.5

PX_UMBRAL = 10

FRACCION_UMBRAL = 0.10

NUM_WORKERS = 2

COLORES = {
    "tp": (0, 255, 0),
    "fp": (255, 0, 0),
    "fn": (0, 100, 255),
}

ALPHA = 0.5

CATEGORIAS_REVISION = (
    "error_mixto",
    "falso_positivo",
    "falso_negativo",
)


def cargar_modelo(
    ruta_pesos: Path,
    dispositivo: torch.device,
) -> nn.Module:
    """Crea el modelo U-Net y carga los pesos entrenados.

    Args:
        ruta_pesos (Path): Ruta del archivo .pt.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        nn.Module: Modelo preparado para realizar predicciones.
    """
    # Creamos la misma arquitectura utilizada durante el entrenamiento.
    modelo = crear_modelo()

    # Cargamos los pesos en el dispositivo que vamos a utilizar.
    pesos = torch.load(
        ruta_pesos,
        map_location=dispositivo,
        weights_only=True,
    )
    modelo.load_state_dict(pesos)
    modelo.to(dispositivo)

    # Activamos el modo de evaluación.
    modelo.eval()

    return modelo


def crear_cargador_test() -> DataLoader:
    """Crea el cargador de datos para el conjunto de test.

    Returns:
        DataLoader: Cargador con las imágenes, máscaras e identificadores.
    """
    dataset = DatasetPaneles(
        DATASET,
        split="test",
        aumentar=False,
    )

    cargador = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    return cargador


def predecir(
    modelo: nn.Module,
    tensor: torch.Tensor,
    dispositivo: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene el mapa de probabilidades y la máscara binaria.

    Args:
        modelo (nn.Module): Modelo U-Net entrenado.
        tensor (torch.Tensor): Imagen preparada para el modelo.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Probabilidades entre 0 y 1 y máscara binaria.
    """
    tensor = tensor.to(dispositivo)

    # No necesitamos calcular gradientes durante la evaluación.
    with torch.inference_mode():
        prediccion = modelo(tensor)  # Predicción en logits.
        probabilidades = torch.sigmoid(prediccion)

    # Quitamos las dimensiones correspondientes al batch y al canal.
    probabilidades = probabilidades.squeeze(0).squeeze(0)
    probabilidades = probabilidades.cpu().numpy()

    # Consideramos panel los píxeles que superan el umbral.
    mascara = probabilidades >= UMBRAL

    return probabilidades, mascara


def calcular_indicadores(
    real: np.ndarray,
    predicha: np.ndarray,
    tile_id: str,
) -> dict:
    """Calcula los aciertos y errores de segmentación de una imagen.

    Args:
        real (np.ndarray): Máscara real binaria.
        predicha (np.ndarray): Máscara predicha binaria.
        tile_id (str): Identificador de la imagen.

    Returns:
        dict: TP, FP, FN, métricas y superficies en píxeles.
    """
    tp = np.count_nonzero(real & predicha)
    fp = np.count_nonzero(~real & predicha)
    fn = np.count_nonzero(real & ~predicha)

    pixeles_reales = tp + fn
    pixeles_predichos = tp + fp

    # Si ambas máscaras están vacías, la segmentación es perfecta.
    dice = 1.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)

    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)

    # No están definidas si no hay predicciones o positivos reales.
    precision = np.nan if tp + fp == 0 else tp / (tp + fp)
    recall = np.nan if tp + fn == 0 else tp / (tp + fn)

    diferencia_superficie = pixeles_predichos - pixeles_reales

    # Una imagen sin panel real no tiene sesgo relativo calculable.
    sesgo_superficie = np.nan if pixeles_reales == 0 else diferencia_superficie / pixeles_reales

    resultado = {
        "tile_id": tile_id,
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

    return resultado


def clasificar_resultado(resultado: dict) -> str:
    """Clasifica una predicción según sus errores de segmentación.

    Args:
        resultado (dict): Indicadores calculados para una imagen.

    Returns:
        str: Categoría utilizada para organizar la revisión.
    """
    tp = resultado["tp"]
    fp = resultado["fp"]
    fn = resultado["fn"]

    pixeles_reales = tp + fn
    pixeles_predichos = tp + fp

    exceso = fp > PX_UMBRAL and fp / max(pixeles_predichos, 1) > FRACCION_UMBRAL

    defecto = fn > PX_UMBRAL and fn / max(pixeles_reales, 1) > FRACCION_UMBRAL

    if exceso and defecto:
        return "error_mixto"

    if exceso:
        return "falso_positivo"

    if defecto:
        return "falso_negativo"

    return "aceptable"


def crear_visualizacion(
    imagen: Image.Image,
    real: np.ndarray,
    predicha: np.ndarray,
) -> Image.Image:
    """Dibuja visualmente los aciertos y errores del modelo.

    Args:
        imagen (Image.Image): Imagen PNOA original.
        real (np.ndarray): Máscara real binaria.
        predicha (np.ndarray): Máscara predicha binaria.

    Returns:
        Image.Image: Imagen preparada para la revisión manual.
    """
    imagen = np.asarray(
        imagen.convert("RGB"),
        dtype=np.float32,
    ).copy()

    tp = real & predicha
    fp = ~real & predicha
    fn = real & ~predicha

    mascaras = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    # Coloreamos los aciertos y errores sin ocultar la ortofoto.
    for tipo, mascara in mascaras.items():
        color = np.asarray(COLORES[tipo], dtype=np.float32)

        imagen[mascara] = imagen[mascara] * (1 - ALPHA) + color * ALPHA

    imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    return Image.fromarray(imagen)


def preparar_salida() -> None:
    """Elimina la evaluación anterior y crea las carpetas de salida."""
    if SALIDA.exists():
        shutil.rmtree(SALIDA)

    SALIDA.mkdir(parents=True)

    for categoria in CATEGORIAS_REVISION:
        (SALIDA / categoria).mkdir()


def evaluar_dataset(
    modelo: nn.Module,
    cargador: DataLoader,
    dispositivo: torch.device,
) -> list[dict]:
    """Evalúa todas las imágenes del conjunto de test.

    Args:
        modelo (nn.Module): Modelo U-Net entrenado.
        cargador (DataLoader): Cargador del conjunto de test.
        dispositivo (torch.device): CPU o GPU utilizada.
    Returns:
        list[dict]: Resultados individuales de todas las imágenes.
    """
    resultados = []

    for indice, (imagenes, mascaras, tile_ids) in enumerate(cargador, start=1):
        tile_id = tile_ids[0]

        # La máscara del dataset todavía tiene dimensiones de batch y canal.
        real = mascaras.squeeze(0).squeeze(0).numpy() > 0.5

        probabilidades, predicha = predecir(
            modelo,
            imagenes,
            dispositivo,
        )

        resultado = calcular_indicadores(
            real,
            predicha,
            tile_id,
        )

        categoria = clasificar_resultado(resultado)

        resultado["categoria"] = categoria
        resultado["probabilidad_maxima"] = float(probabilidades.max())

        resultados.append(resultado)

        # Las imágenes aceptables aparecen en el CSV, pero no se duplican.
        if categoria != "aceptable":
            ruta_imagen = DATASET / "images" / "test" / f"{tile_id}.png"

            with Image.open(ruta_imagen) as imagen:
                visualizacion = crear_visualizacion(
                    imagen,
                    real,
                    predicha,
                )

            destino = SALIDA / categoria / f"{tile_id}.png"
            visualizacion.save(destino)

        if indice % 100 == 0:
            print(f"Evaluadas {indice} imágenes.")

    return resultados


def guardar_resultados(resultados: list[dict]) -> None:
    """Guarda los resultados individuales y calcula el resumen global.

    Args:
        resultados (list[dict]): Resultados individuales del conjunto de test.
    """
    if not resultados:
        print("No hay imágenes en el conjunto de test.")
        return

    tabla = pd.DataFrame(resultados)

    tabla.to_csv(
        SALIDA / "resultados.csv",
        index=False,
    )

    # Las métricas globales se calculan sumando primero todos los píxeles.
    tp = int(tabla["tp"].sum())
    fp = int(tabla["fp"].sum())
    fn = int(tabla["fn"].sum())

    pixeles_reales = tp + fn
    pixeles_predichos = tp + fp

    dice = 1.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)

    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)

    precision = np.nan if tp + fp == 0 else tp / (tp + fp)
    recall = np.nan if tp + fn == 0 else tp / (tp + fn)

    diferencia_superficie = pixeles_predichos - pixeles_reales

    sesgo_superficie = np.nan if pixeles_reales == 0 else diferencia_superficie / pixeles_reales

    resumen = {
        "imagenes": len(tabla),
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

    pd.DataFrame([resumen]).to_csv(
        SALIDA / "resumen.csv",
        index=False,
    )

    print()
    print("Resultados globales")
    print(f"Imágenes: {len(tabla)}")
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Sesgo de superficie: {sesgo_superficie:.2%}")

    print()
    print("Clasificación de imágenes")

    for categoria, cantidad in tabla["categoria"].value_counts().items():
        print(f"{categoria}: {cantidad}")


def main() -> None:
    """Ejecuta la evaluación completa del modelo U-Net."""
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dispositivo: {dispositivo}")
    preparar_salida()

    modelo = cargar_modelo(
        PESOS,
        dispositivo,
    )

    cargador = crear_cargador_test()

    resultados = evaluar_dataset(
        modelo,
        cargador,
        dispositivo,
    )

    guardar_resultados(resultados)
    print(f"Resultados guardados en: {SALIDA}")


if __name__ == "__main__":
    main()
