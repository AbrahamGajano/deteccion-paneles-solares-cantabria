"""Entrena una U-Net para segmentar paneles solares."""

import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from paneles_solares.rutas import ruta_proyecto

DATASET = ruta_proyecto("data/datasets/unet")
PESOS = ruta_proyecto("weights/unet_resnet34.pt")

BATCH_SIZE = 4
NUM_WORKERS = 2

EPOCHS = 300
PACIENCIA = 30

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

UMBRAL = 0.5
SEMILLA = 42

# Normalización utilizada para entrenar ResNet34 con ImageNet.
MEDIA_IMAGENET = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

DESVIACION_IMAGENET = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class DatasetPaneles(Dataset):
    """Carga las imágenes y máscaras utilizadas para entrenar U-Net."""

    def __init__(
        self,
        dataset: Path,
        split: str,
        aumentar: bool = False,
    ):
        """Prepara las rutas de un split del dataset.

        Args:
            dataset (Path): Carpeta principal del dataset U-Net.
            split (str): División que se quiere cargar: train, val o test.
            aumentar (bool): Indica si se aplican aumentos aleatorios.
        """
        self.carpeta_imagenes = dataset / "images" / split
        self.carpeta_mascaras = dataset / "masks" / split
        self.aumentar = aumentar

        self.imagenes = sorted(self.carpeta_imagenes.glob("*.png"))

        if not self.imagenes:
            raise ValueError(f"No hay imágenes en {self.carpeta_imagenes}")

    def __len__(self) -> int:
        """Devuelve el número de imágenes del dataset.

        Returns:
            int: Cantidad de imágenes disponibles.
        """
        return len(self.imagenes)

    def __getitem__(
        self,
        indice: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Carga una imagen y su máscara correspondiente.

        Args:
            indice (int): Posición de la muestra que se quiere cargar.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]:
                Imagen y máscara convertidas en tensores e id_tile.
        """
        ruta_imagen = self.imagenes[indice]
        ruta_mascara = self.carpeta_mascaras / ruta_imagen.name

        imagen = Image.open(ruta_imagen).convert("RGB")
        mascara = Image.open(ruta_mascara).convert("L")

        imagen = np.array(imagen, dtype=np.float32)
        mascara = np.array(mascara)

        # Los aumentos espaciales deben aplicarse igual a imagen y máscara.
        if self.aumentar:
            imagen, mascara = aumentar_datos(imagen, mascara)

        # La imagen pasa del intervalo 0-255 al intervalo 0-1.
        imagen = imagen / 255.0

        # Adaptamos los colores a la distribución usada por ImageNet.
        imagen = (imagen - MEDIA_IMAGENET) / DESVIACION_IMAGENET

        # La máscara queda formada únicamente por ceros y unos.
        mascara = (mascara > 0).astype(np.float32)

        # Pasamos de alto-ancho-canales a canales-alto-ancho.
        imagen = np.transpose(imagen, (2, 0, 1))

        # Añadimos a la máscara su único canal.
        mascara = mascara[np.newaxis, :, :]

        # Algunas transformaciones dejan los arrays no contiguos.
        imagen = np.ascontiguousarray(imagen)
        mascara = np.ascontiguousarray(mascara)

        imagen = torch.from_numpy(imagen)
        mascara = torch.from_numpy(mascara)

        return imagen, mascara, ruta_imagen.stem


def aumentar_datos(
    imagen: np.ndarray,
    mascara: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica transformaciones aleatorias a una imagen y su máscara.

    Args:
        imagen (np.ndarray): Imagen RGB.
        mascara (np.ndarray): Máscara correspondiente.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Imagen y máscara transformadas.
    """
    # Volteo horizontal.
    if random.random() < 0.5:
        imagen = np.flip(imagen, axis=1)
        mascara = np.flip(mascara, axis=1)

    # Volteo vertical.
    if random.random() < 0.5:
        imagen = np.flip(imagen, axis=0)
        mascara = np.flip(mascara, axis=0)

    # Rotación de 0, 90, 180 o 270 grados.
    giros = random.randint(0, 3)

    if giros:
        imagen = np.rot90(imagen, giros)
        mascara = np.rot90(mascara, giros)

    return imagen, mascara


def crear_cargadores(
    dataset: Path,
) -> tuple[DataLoader, DataLoader]:
    """Crea los cargadores de entrenamiento y validación.

    Args:
        dataset (Path): Ruta principal del dataset U-Net.

    Returns:
        tuple[DataLoader, DataLoader]:
            Cargadores de entrenamiento y validación.
    """
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


def crear_modelo() -> nn.Module:
    """Crea una U-Net con encoder ResNet34 preentrenado.

    Returns:
        nn.Module: Modelo de segmentación.
    """
    modelo = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )

    return modelo


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
        """Calcula el error total de las predicciones.

        Args:
            predicciones (torch.Tensor): Salida producida por U-Net.
            mascaras (torch.Tensor): Máscaras correctas.

        Returns:
            torch.Tensor: Pérdida total del lote.
        """
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
    """Entrena el modelo durante una época completa.

    Args:
        modelo (nn.Module): Modelo que se quiere entrenar.
        cargador (DataLoader): Cargador de entrenamiento.
        criterio (nn.Module): Función de pérdida.
        optimizador (torch.optim.Optimizer): Optimizador del modelo.
        escalador (torch.amp.GradScaler): Escalador para precisión mixta.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        float: Pérdida media de entrenamiento.
    """
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
    """Evalúa el modelo utilizando el conjunto de validación.

    Args:
        modelo (nn.Module): Modelo que se quiere validar.
        cargador (DataLoader): Cargador de validación.
        criterio (nn.Module): Función de pérdida.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        dict[str, float]: Pérdida y métricas de validación.
    """
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
    """Guarda los pesos entrenados del modelo.

    Args:
        modelo (nn.Module): Modelo cuyos pesos se quieren guardar.
        ruta (Path): Archivo de destino.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        modelo.state_dict(),
        ruta,
    )


def fijar_semilla(semilla: int) -> None:
    """Fija la semilla utilizada por los generadores aleatorios.

    Args:
        semilla (int): Semilla que se quiere utilizar.
    """
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
                PESOS,
            )

            print(f"Mejor modelo guardado: {PESOS}")

        else:
            epocas_sin_mejora += 1

        # Terminamos si Dice lleva demasiadas épocas sin mejorar.
        if epocas_sin_mejora >= PACIENCIA:
            print(f"Entrenamiento detenido porque Dice no mejoró durante {PACIENCIA} épocas.")
            break

    print(f"Mejor Dice de validación: {mejor_dice:.4f}")


def main() -> None:
    """Ejecuta el entrenamiento de U-Net."""
    entrenar()


if __name__ == "__main__":
    main()
