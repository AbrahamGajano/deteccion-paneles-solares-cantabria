from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from paneles_solares.rutas import ruta_proyecto

DATASET = ruta_proyecto("data/datasets/unet")
BATCH_SIZE = 2
NUM_WORKERS = 0


class DatasetPaneles(Dataset):
    """Carga las imágenes y máscaras utilizadas para entrenar U-Net."""

    def __init__(self, dataset: Path, split: str):
        """Prepara las rutas de un split del dataset.

        Args:
            dataset (Path): Carpeta principal del dataset U-Net.
            split (str): División que se quiere cargar: train, val o test.
        """
        self.carpeta_imagenes = dataset / "images" / split
        self.carpeta_mascaras = dataset / "masks" / split

        self.imagenes = sorted(self.carpeta_imagenes.glob("*.png"))

    def __len__(self) -> int:
        """Devuelve el número de imágenes del dataset.

        Returns:
            int: Cantidad de imágenes disponibles.
        """
        return len(self.imagenes)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Carga una imagen y su máscara correspondiente.

        Args:
            indice (int): Posición de la muestra que se quiere cargar.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Imagen y máscara.
        """
        ruta_imagen = self.imagenes[indice]
        ruta_mascara = self.carpeta_mascaras / ruta_imagen.name

        imagen = Image.open(ruta_imagen).convert("RGB")
        mascara = Image.open(ruta_mascara).convert("L")

        imagen = np.array(imagen, dtype=np.float32) / 255.0
        mascara = np.array(mascara) > 0

        imagen = np.transpose(imagen, (2, 0, 1))

        mascara = mascara[np.newaxis, :, :].astype(np.float32)

        imagen = torch.from_numpy(imagen)
        mascara = torch.from_numpy(mascara)

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
    dataset_train = DatasetPaneles(dataset, "train")
    dataset_val = DatasetPaneles(dataset, "val")

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


def main() -> None:
    dataset = DatasetPaneles(DATASET, "train")

    imagen, mascara = dataset[0]

    print(f"Cantidad: {len(dataset)}")
    print(f"Imagen: {imagen.shape}, {imagen.dtype}")
    print(f"Máscara: {mascara.shape}, {mascara.dtype}")
    print(f"Valores máscara: {torch.unique(mascara)}")


if __name__ == "__main__":
    main()
