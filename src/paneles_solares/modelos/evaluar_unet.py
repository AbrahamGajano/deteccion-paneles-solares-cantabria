from pathlib import Path

import numpy as np
import torch
from torch import nn

from paneles_solares.modelos.entrenar_unet import crear_modelo

UMBRAL = 0.5


def cargar_modelo(ruta_pesos: Path, dispositivo: torch.device) -> nn.Module:
    """Crea el modelo U-Net y carga los pesos entrenados.

    Args:
        ruta_pesos (Path): Ruta del archivo .pt.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        nn.Module: Modelo preparado para realizar predicciones.
    """

    # Cargamos el modelo con los pesos del entreno
    modelo = crear_modelo()
    pesos = torch.load(ruta_pesos, map_location=dispositivo, weights_only=True)

    modelo.load_state_dict(pesos)
    modelo.to(dispositivo)

    # Le ponemos en modo de test
    modelo.eval()

    return modelo


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

    # Simplemente por optimizacion
    with torch.inference_mode():
        pred = modelo(tensor)  # Prediccion (EN LOGITS)
        prob = torch.sigmoid(pred)  # Ya [0-1]

    prob = prob.squeeze(0).squeeze(0)  # Como el batch le puse a 1
    prob = prob.cpu().numpy()

    # Decidimos solo aquellos mayor o igual a la pred
    mask = prob >= UMBRAL

    return prob, mask


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

    # Si ambas máscaras están vacías, la segmentación es perfecta
    dice = 1.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)

    # Estas métricas no están definidas si no existen predicciones o positivos reales
    precision = np.nan if tp + fp == 0 else tp / (tp + fp)
    recall = np.nan if tp + fn == 0 else tp / (tp + fn)

    diferencia_superficie = pixeles_predichos - pixeles_reales

    # Una imagen sin panel real no tiene un sesgo relativo de superficie calculable
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


PX_UMBRAL = 10
FRACCION_UMBRAL = 0.10


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
