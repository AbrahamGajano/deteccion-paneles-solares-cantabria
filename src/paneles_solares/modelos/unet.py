"""Arquitectura, entrada RGB y carga de pesos de la U-Net actual."""

import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

TAM_ENTRADA = 512
UMBRAL = 0.5
MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIACION_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalizar_rgb(imagen: np.ndarray) -> np.ndarray:
    """Normaliza RGB con los valores ImageNet usados durante el entrenamiento.

    Args:
        imagen: Imagen RGB que se procesa.

    Returns:
        np.ndarray: Imagen en float32 con normalización ImageNet.
    """

    imagen = imagen.astype(np.float32, copy=False) / 255.0
    return (imagen - MEDIA_IMAGENET) / DESVIACION_IMAGENET


class DatasetPaneles(Dataset):
    """Carga los mismos PNG para entrenamiento, validación y evaluación."""

    def __init__(
        self,
        dataset: Path,
        split: str,
        aumentar: bool = False,
    ):
        """Localiza las imágenes y máscaras de una división del dataset.

        Args:
            dataset: Carpeta con imágenes y máscaras del dataset.
            split: División del dataset: train, val o test.
            aumentar: Activa volteos y giros durante el entrenamiento.

        Raises:
            ValueError: La división del dataset no contiene imágenes PNG.
        """

        self.carpeta_imagenes = dataset / "images" / split
        self.carpeta_mascaras = dataset / "masks" / split
        self.aumentar = aumentar

        self.imagenes = sorted(self.carpeta_imagenes.glob("*.png"))

        if not self.imagenes:
            raise ValueError(f"No hay imágenes en {self.carpeta_imagenes}")

    def __len__(self) -> int:
        """Devuelve el número de imágenes de la división.

        Returns:
            int: Número de imágenes disponibles.
        """

        return len(self.imagenes)

    def __getitem__(
        self,
        indice: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Carga y prepara una imagen, su máscara y su identificador.

        Args:
            indice: Posición de la imagen dentro del dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: Tensores de imagen y máscara, y el
                identificador de la tesela.
        """

        ruta_imagen = self.imagenes[indice]
        ruta_mascara = self.carpeta_mascaras / ruta_imagen.name

        imagen = Image.open(ruta_imagen).convert("RGB")
        mascara = Image.open(ruta_mascara).convert("L")

        # Las campañas antiguas tienen otra resolución, pero el lote necesita
        # un tamaño común. Los originales conservan su escala para medir áreas.
        if imagen.size != (TAM_ENTRADA, TAM_ENTRADA):
            imagen = imagen.resize(
                (TAM_ENTRADA, TAM_ENTRADA), Image.Resampling.BILINEAR
            )
            mascara = mascara.resize(
                (TAM_ENTRADA, TAM_ENTRADA), Image.Resampling.NEAREST
            )

        imagen = np.array(imagen, dtype=np.float32)
        mascara = np.array(mascara)

        # Los aumentos espaciales deben aplicarse igual a imagen y máscara.
        if self.aumentar:
            imagen, mascara = aumentar_datos(imagen, mascara)

        imagen = normalizar_rgb(imagen)

        mascara = (mascara > 0).astype(np.float32)
        imagen = np.transpose(imagen, (2, 0, 1))
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
    """Aplica las mismas transformaciones espaciales al RGB y a la máscara.

    Args:
        imagen: Imagen RGB que se procesa.
        mascara: Máscara binaria de paneles.

    Returns:
        tuple[np.ndarray, np.ndarray]: Imagen y máscara con las mismas transformaciones
            espaciales.
    """

    if random.random() < 0.5:
        imagen = np.flip(imagen, axis=1)
        mascara = np.flip(mascara, axis=1)

    if random.random() < 0.5:
        imagen = np.flip(imagen, axis=0)
        mascara = np.flip(mascara, axis=0)

    giros = random.randint(0, 3)

    if giros:
        imagen = np.rot90(imagen, giros)
        mascara = np.rot90(mascara, giros)

    return imagen, mascara


def crear_modelo(preentrenado: bool = True) -> nn.Module:
    """Crea la arquitectura ResNet34 utilizada por los pesos actuales.

    Returns:
        nn.Module: U-Net ResNet34 con pesos ImageNet en el codificador.
    """

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if preentrenado else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def cargar_modelo(
    ruta_pesos: Path,
    dispositivo: torch.device,
) -> nn.Module:
    """Carga los pesos y prepara la U-Net para inferencia.

    Args:
        ruta_pesos: Archivo de pesos de la U-Net.
        dispositivo: CPU o GPU utilizada por el modelo.

    Returns:
        nn.Module: U-Net con los pesos indicados, en modo de evaluación.
    """

    modelo = crear_modelo(preentrenado=False)
    pesos = torch.load(
        ruta_pesos,
        map_location=dispositivo,
        weights_only=True,
    )
    modelo.load_state_dict(pesos)
    modelo.to(dispositivo)

    modelo.eval()

    return modelo
