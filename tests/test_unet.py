"""Comprueba que el test recibe fotografías y sus máscaras correspondientes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from paneles_solares.modelos import evaluar_unet
from paneles_solares.modelos.entrenar_unet import (
    DESVIACION_IMAGENET,
    MEDIA_IMAGENET,
)


class TestDatasetUnet(unittest.TestCase):
    def test_cargador_lee_fotografia_y_mascara_por_separado(self):
        with tempfile.TemporaryDirectory() as temporal:
            dataset = Path(temporal)
            imagen = np.full((32, 32, 3), [40, 120, 200], dtype=np.uint8)
            mascara = np.zeros((32, 32), dtype=np.uint8)
            mascara[8:16, 8:16] = 255

            for carpeta, datos in (("images", imagen), ("masks", mascara)):
                destino = dataset / carpeta / "test"
                destino.mkdir(parents=True)
                Image.fromarray(datos).save(destino / "ejemplo.png")

            with (
                patch.object(evaluar_unet, "DATASET", dataset),
                patch.object(evaluar_unet, "NUM_WORKERS", 0),
            ):
                cargador = evaluar_unet.crear_cargador_test()
                imagenes, mascaras, ids = next(iter(cargador))

            esperada = (imagen.astype(np.float32) / 255 - MEDIA_IMAGENET) / DESVIACION_IMAGENET
            np.testing.assert_allclose(imagenes[0].numpy(), esperada.transpose(2, 0, 1))
            np.testing.assert_array_equal(mascaras[0, 0].numpy(), mascara > 0)
            self.assertEqual(ids[0], "ejemplo")
            self.assertEqual(cargador.dataset.carpeta_imagenes, dataset / "images" / "test")
            self.assertEqual(cargador.dataset.carpeta_mascaras, dataset / "masks" / "test")

    def test_prediccion_conserva_aciertos_y_errores(self):
        # Logits conocidos: dos píxeles predichos, uno correcto y otro falso.
        logits = torch.tensor([[[[-2.0, 2.0], [2.0, -2.0]]]])
        _, predicha = evaluar_unet.predecir(torch.nn.Identity(), logits, torch.device("cpu"))
        real = np.array([[False, True], [False, True]])
        resultado = evaluar_unet.calcular_indicadores(real, predicha, "ejemplo")

        self.assertEqual((resultado["tp"], resultado["fp"], resultado["fn"]), (1, 1, 1))
        self.assertEqual(resultado["dice"], 0.5)


if __name__ == "__main__":
    unittest.main()
