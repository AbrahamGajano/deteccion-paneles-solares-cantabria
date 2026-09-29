"""Entrena una U-Net para segmentar paneles solares."""

import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader

from paneles_solares.modelos.unet import DatasetPaneles, UMBRAL, crear_modelo
from paneles_solares.rutas import PESOS_UNET, ruta_proyecto

DATASET = ruta_proyecto("data/datasets/unet")

PESOS_INICIALES = ruta_proyecto("weights/unet_resnet34_anterior.pt")

BATCH_SIZE = 4
NUM_WORKERS = 2

EPOCHS = 300
PACIENCIA = 30

LEARNING_RATE = 0.00003
WEIGHT_DECAY = 0.0001

SEMILLA = 42


def crear_cargadores(
    dataset: Path,
) -> tuple[DataLoader, DataLoader]:
    """Crea los cargadores de entrenamiento y validación."""
    dataset_train = DatasetPaneles(
        dataset,
        split="train",
        aumentar=True,
    )

    dataset_val = DatasetPaneles(
        dataset,
        split="val",
        aumentar=False,
    )

    cargador_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    cargador_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return cargador_train, cargador_val


class PerdidaSegmentacion(nn.Module):
    """Combina BCE y Dice para entrenar segmentación binaria."""

    def __init__(self):
        """Prepara las dos funciones de pérdida."""
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = smp.losses.DiceLoss(
            mode="binary",
            from_logits=True,
        )

    def forward(
        self,
        predicciones: torch.Tensor,
        mascaras: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula el error total de las predicciones."""
        perdida_bce = self.bce(
            predicciones,
            mascaras,
        )

        perdida_dice = self.dice(
            predicciones,
            mascaras,
        )

        return perdida_bce + perdida_dice


def entrenar_epoca(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: nn.Module,
    optimizador: torch.optim.Optimizer,
    escalador: torch.amp.GradScaler,
    dispositivo: torch.device,
) -> float:
    """Entrena el modelo durante una época completa."""
    modelo.train()

    perdida_total = 0.0
    cantidad = 0

    for imagenes, mascaras, _ in cargador:
        imagenes = imagenes.to(
            dispositivo,
            non_blocking=True,
        )

        mascaras = mascaras.to(
            dispositivo,
            non_blocking=True,
        )

        # Borramos los gradientes calculados en el lote anterior.
        optimizador.zero_grad(set_to_none=True)

        # La precisión mixta reduce memoria y acelera el entrenamiento.
        with torch.autocast(
            device_type=dispositivo.type,
            enabled=dispositivo.type == "cuda",
        ):
            predicciones = modelo(imagenes)
            perdida = criterio(predicciones, mascaras)

        # Calculamos los gradientes y actualizamos los pesos.
        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()

        tamano_lote = imagenes.size(0)

        perdida_total += perdida.item() * tamano_lote
        cantidad += tamano_lote

    return perdida_total / cantidad


def validar_epoca(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
) -> dict[str, float]:
    """Evalúa el modelo utilizando el conjunto de validación."""
    modelo.eval()

    perdida_total = 0.0
    cantidad = 0

    verdaderos_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    # En validación no necesitamos calcular gradientes.
    with torch.inference_mode():
        for imagenes, mascaras, _ in cargador:
            imagenes = imagenes.to(
                dispositivo,
                non_blocking=True,
            )

            mascaras = mascaras.to(
                dispositivo,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=dispositivo.type,
                enabled=dispositivo.type == "cuda",
            ):
                predicciones = modelo(imagenes)
                perdida = criterio(predicciones, mascaras)

            tamano_lote = imagenes.size(0)

            perdida_total += perdida.item() * tamano_lote
            cantidad += tamano_lote

            # Convertimos los logits en probabilidades entre 0 y 1.
            probabilidades = torch.sigmoid(predicciones)

            # El umbral convierte las probabilidades en una máscara binaria.
            mascaras_predichas = probabilidades >= UMBRAL
            mascaras_reales = mascaras >= 0.5

            verdaderos_positivos += (mascaras_predichas & mascaras_reales).sum().item()

            falsos_positivos += (mascaras_predichas & ~mascaras_reales).sum().item()

            falsos_negativos += (~mascaras_predichas & mascaras_reales).sum().item()

    divisor_dice = 2 * verdaderos_positivos + falsos_positivos + falsos_negativos

    divisor_iou = verdaderos_positivos + falsos_positivos + falsos_negativos

    divisor_precision = verdaderos_positivos + falsos_positivos

    divisor_recall = verdaderos_positivos + falsos_negativos

    dice = 2 * verdaderos_positivos / divisor_dice if divisor_dice else 1.0

    iou = verdaderos_positivos / divisor_iou if divisor_iou else 1.0

    precision = verdaderos_positivos / divisor_precision if divisor_precision else 0.0

    recall = verdaderos_positivos / divisor_recall if divisor_recall else 0.0

    return {
        "perdida": perdida_total / cantidad,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def guardar_modelo(
    modelo: nn.Module,
    ruta: Path,
) -> None:
    """Guarda los pesos entrenados del modelo."""
    ruta.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        modelo.state_dict(),
        ruta,
    )


def fijar_semilla(semilla: int) -> None:
    """Fija la semilla utilizada por los generadores aleatorios."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def entrenar() -> None:
    """Prepara y ejecuta el entrenamiento completo."""
    fijar_semilla(SEMILLA)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cargador_train, cargador_val = crear_cargadores(DATASET)

    modelo = crear_modelo()
    modelo = modelo.to(dispositivo)

    pesos = torch.load(
        PESOS_INICIALES,
        map_location=dispositivo,
        weights_only=True,
    )
    modelo.load_state_dict(pesos)

    criterio = PerdidaSegmentacion()
    criterio = criterio.to(dispositivo)

    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Reduce el learning rate cuando la pérdida de validación se estanca.
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=0.000001,
    )

    # Solo se activa cuando utilizamos una GPU CUDA.
    escalador = torch.amp.GradScaler(
        "cuda",
        enabled=dispositivo.type == "cuda",
    )

    mejor_dice = -1.0
    epocas_sin_mejora = 0

    print(f"Dispositivo: {dispositivo}")
    print(f"Imágenes train: {len(cargador_train.dataset)}")
    print(f"Imágenes val: {len(cargador_val.dataset)}")

    for epoca in range(1, EPOCHS + 1):
        perdida_train = entrenar_epoca(
            modelo=modelo,
            cargador=cargador_train,
            criterio=criterio,
            optimizador=optimizador,
            escalador=escalador,
            dispositivo=dispositivo,
        )

        metricas_val = validar_epoca(
            modelo=modelo,
            cargador=cargador_val,
            criterio=criterio,
            dispositivo=dispositivo,
        )

        # El planificador observa la pérdida de validación.
        planificador.step(metricas_val["perdida"])

        learning_rate_actual = optimizador.param_groups[0]["lr"]

        print(
            f"Época {epoca:03d}/{EPOCHS} | "
            f"train_loss: {perdida_train:.4f} | "
            f"val_loss: {metricas_val['perdida']:.4f} | "
            f"dice: {metricas_val['dice']:.4f} | "
            f"iou: {metricas_val['iou']:.4f} | "
            f"precision: {metricas_val['precision']:.4f} | "
            f"recall: {metricas_val['recall']:.4f} | "
            f"lr: {learning_rate_actual:.7f}"
        )

        # Guardamos solamente el modelo con mejor Dice.
        if metricas_val["dice"] > mejor_dice:
            mejor_dice = metricas_val["dice"]
            epocas_sin_mejora = 0

            guardar_modelo(
                modelo,
                PESOS_UNET,
            )

            print(f"Mejor modelo guardado: {PESOS_UNET}")

        else:
            epocas_sin_mejora += 1

        # Terminamos si Dice lleva demasiadas épocas sin mejorar.
        if epocas_sin_mejora >= PACIENCIA:
            print(
                f"Entrenamiento detenido porque Dice no mejoró durante {PACIENCIA} épocas."
            )
            break

    print(f"Mejor Dice de validación: {mejor_dice:.4f}")


if __name__ == "__main__":
    entrenar()
