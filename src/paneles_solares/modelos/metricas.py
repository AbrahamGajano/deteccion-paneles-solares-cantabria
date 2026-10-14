"""Métricas de segmentación compartidas por U-Net, YOLO y las validaciones."""

import numpy as np


def calcular_metricas(tp: int, fp: int, fn: int) -> dict:
    """Calcula métricas de píxel a partir de los aciertos y errores acumulados.

    Args:
        tp: Píxeles positivos correctamente predichos.
        fp: Píxeles de fondo predichos como panel.
        fn: Píxeles de panel que el modelo no detectó.

    Returns:
        dict: Métricas, recuentos y sesgo; NaN cuando falta un denominador.
    """

    reales = tp + fn
    predichos = tp + fp
    diferencia = predichos - reales

    # Dos máscaras vacías coinciden; precisión, recall y sesgo no están definidos.
    divisor_dice = 2 * tp + fp + fn
    divisor_iou = tp + fp + fn

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": 1.0 if divisor_dice == 0 else 2 * tp / divisor_dice,
        "iou": 1.0 if divisor_iou == 0 else tp / divisor_iou,
        "precision": np.nan if predichos == 0 else tp / predichos,
        "recall": np.nan if reales == 0 else tp / reales,
        "pixeles_reales": reales,
        "pixeles_predichos": predichos,
        "diferencia_superficie": diferencia,
        "sesgo_superficie": np.nan if reales == 0 else diferencia / reales,
    }
