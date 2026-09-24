"""Decide qué imágenes válidas se utilizan para entrenar y validar."""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from paneles_solares.datos.coleccion import (
    MANIFEST,
    POOL,
    cargar_manifest,
    guardar_manifest,
)
from paneles_solares.rutas import ruta_proyecto

# La validación solo sale de imágenes aleatorias para que sea representativa.
FRACCION_VAL_ALEATORIAS = 0.25
REHACER_VALIDACION = True

# Además de los fallos de los modelos, añadimos algunos fondos fáciles.
MAX_NEGATIVAS_FACILES = 100

# Umbrales bajos para encontrar falsos positivos útiles para el entrenamiento.
UMBRAL_YOLO = 0.20
UMBRAL_UNET = 0.30
MINIMO_PIXELES = 10

MODELO_YOLO = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")
MODELO_UNET = ruta_proyecto("weights/unet_resnet34.pt")

IMGSZ = 640
DEVICE_YOLO = 0
SEMILLA = 42

MEDIA_IMAGENET = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)
DESVIACION_IMAGENET = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

MOTIVO_VALIDACION = "validacion aleatoria representativa"


def asignar_uso(
    manifest: pd.DataFrame,
    indices: pd.Index,
    uso: str,
    motivo: str,
) -> None:
    """Asigna el uso de varias imágenes y conserva su valor anterior.

    Args:
        manifest (pd.DataFrame): Manifest completo del pool.
        indices (pd.Index): Filas que se quieren modificar.
        uso (str): Nuevo uso de las imágenes.
        motivo (str): Explicación breve de la decisión.
    """

    if len(indices) == 0:
        return

    manifest.loc[indices, "uso_anterior"] = manifest.loc[indices, "uso"]
    manifest.loc[indices, "uso"] = uso
    manifest.loc[indices, "motivo"] = motivo


def elegir_validacion(aleatorias: pd.DataFrame) -> pd.Index:
    """Elige una parte de las aleatorias manteniendo positivos y negativos.

    Args:
        aleatorias (pd.DataFrame): Imágenes aleatorias válidas y pendientes.

    Returns:
        pd.Index: Filas que se utilizarán para validación.
    """

    if not 0 <= FRACCION_VAL_ALEATORIAS < 1:
        raise ValueError("La fracción de validación debe estar entre 0 y 1")

    if aleatorias.empty or FRACCION_VAL_ALEATORIAS == 0:
        return pd.Index([])

    # Se toma la misma fracción de positivas y negativas.
    validacion = aleatorias.groupby(
        "tipo",
        group_keys=False,
    ).sample(
        frac=FRACCION_VAL_ALEATORIAS,
        random_state=SEMILLA,
    )

    return validacion.index


def cargar_modelos() -> tuple[object, torch.nn.Module, torch.device]:
    """Carga una sola vez YOLO y U-Net para revisar las negativas.

    Returns:
        tuple[object, torch.nn.Module, torch.device]:
            Modelo YOLO, modelo U-Net y dispositivo de U-Net.
    """

    if not MODELO_YOLO.is_file():
        raise FileNotFoundError(MODELO_YOLO)

    if not MODELO_UNET.is_file():
        raise FileNotFoundError(MODELO_UNET)

    from ultralytics import YOLO

    from paneles_solares.modelos.entrenar_unet import crear_modelo

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo_yolo = YOLO(str(MODELO_YOLO))

    modelo_unet = crear_modelo()
    pesos = torch.load(
        MODELO_UNET,
        map_location=dispositivo,
        weights_only=True,
    )
    modelo_unet.load_state_dict(pesos)
    modelo_unet.to(dispositivo)
    modelo_unet.eval()

    return modelo_yolo, modelo_unet, dispositivo


def predecir_yolo(
    modelo: object,
    imagen: np.ndarray,
) -> tuple[float, int, int]:
    """Predice una imagen y resume la salida de YOLO.

    Args:
        modelo (object): Modelo YOLO cargado.
        imagen (np.ndarray): Imagen RGB.

    Returns:
        tuple[float, int, int]: Confianza máxima, detecciones y píxeles.
    """

    # Ultralytics interpreta los arrays como imágenes BGR.
    imagen_bgr = imagen[:, :, ::-1].copy()
    resultado = modelo.predict(
        source=imagen_bgr,
        conf=UMBRAL_YOLO,
        imgsz=IMGSZ,
        device=DEVICE_YOLO,
        retina_masks=True,
        verbose=False,
    )[0]

    if resultado.boxes is None or len(resultado.boxes) == 0:
        return 0.0, 0, 0

    confianza = float(resultado.boxes.conf.max().item())
    detecciones = len(resultado.boxes)

    if resultado.masks is None:
        raise ValueError("YOLO ha detectado objetos, pero no ha devuelto máscaras")

    mascara = resultado.masks.data.any(dim=0)
    pixeles = int(mascara.sum().item())

    return confianza, detecciones, pixeles


def predecir_unet(
    modelo: torch.nn.Module,
    imagen: np.ndarray,
    dispositivo: torch.device,
) -> int:
    """Cuenta los píxeles que U-Net considera panel solar.

    Args:
        modelo (torch.nn.Module): Modelo U-Net cargado.
        imagen (np.ndarray): Imagen RGB.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        int: Número de píxeles predichos como panel.
    """

    preparada = imagen.astype(np.float32) / 255.0
    preparada = (preparada - MEDIA_IMAGENET) / DESVIACION_IMAGENET
    preparada = preparada.transpose(2, 0, 1).copy()

    tensor = torch.from_numpy(preparada).unsqueeze(0).to(dispositivo)

    with torch.inference_mode():
        logits = modelo(tensor)
        probabilidades = torch.sigmoid(logits)

    pixeles = torch.count_nonzero(probabilidades >= UMBRAL_UNET)

    return int(pixeles.item())


def analizar_negativas(
    manifest: pd.DataFrame,
    negativas: pd.DataFrame,
) -> pd.Index:
    """Busca falsos positivos de YOLO y U-Net en imágenes negativas.

    Args:
        manifest (pd.DataFrame): Manifest completo del pool.
        negativas (pd.DataFrame): Negativas pendientes de decidir.

    Returns:
        pd.Index: Negativas fáciles donde ninguno de los modelos falla.
    """

    if negativas.empty:
        return pd.Index([])

    modelo_yolo, modelo_unet, dispositivo = cargar_modelos()

    fallos_yolo = []
    fallos_unet = []
    fallos_ambos = []
    faciles = []

    total = len(negativas)

    for numero, (indice, fila) in enumerate(negativas.iterrows(), start=1):
        ruta_imagen = POOL / "images" / f"{fila['tile_id']}.png"

        if not ruta_imagen.is_file():
            raise FileNotFoundError(ruta_imagen)

        with Image.open(ruta_imagen) as archivo:
            imagen = np.asarray(archivo.convert("RGB"))

        confianza, detecciones, pixeles_yolo = predecir_yolo(
            modelo_yolo,
            imagen,
        )
        pixeles_unet = predecir_unet(
            modelo_unet,
            imagen,
            dispositivo,
        )

        falla_yolo = detecciones > 0 and pixeles_yolo >= MINIMO_PIXELES
        falla_unet = pixeles_unet >= MINIMO_PIXELES

        # Guardamos los datos de YOLO en las columnas que ya tiene el manifest.
        manifest.at[indice, "modelo_yolo"] = str(MODELO_YOLO)
        manifest.at[indice, "confianza_yolo"] = confianza
        manifest.at[indice, "detecciones_yolo"] = detecciones
        manifest.at[indice, "umbral_yolo"] = UMBRAL_YOLO

        if falla_yolo and falla_unet:
            fallos_ambos.append(indice)
        elif falla_yolo:
            fallos_yolo.append(indice)
        elif falla_unet:
            fallos_unet.append(indice)
        else:
            faciles.append(indice)

        if numero % 100 == 0 or numero == total:
            print(f"Analizadas {numero}/{total} negativas.")

    asignar_uso(
        manifest,
        pd.Index(fallos_ambos),
        "train",
        "falso positivo de YOLO y U-Net",
    )
    asignar_uso(
        manifest,
        pd.Index(fallos_yolo),
        "train",
        "falso positivo de YOLO",
    )
    asignar_uso(
        manifest,
        pd.Index(fallos_unet),
        "train",
        "falso positivo de U-Net",
    )

    return pd.Index(faciles)


def decidir_dataset(manifest: pd.DataFrame) -> pd.DataFrame:
    """Decide el uso de las imágenes válidas que continúan pendientes.

    Args:
        manifest (pd.DataFrame): Manifest completo del pool.

    Returns:
        pd.DataFrame: Manifest con las decisiones aplicadas.
    """

    if MAX_NEGATIVAS_FACILES < 0:
        raise ValueError("El máximo de negativas fáciles no puede ser negativo")

    # Una descartada se conserva en el pool, pero nunca entra en un dataset.
    descartadas = manifest.index[manifest["estado"] == "descartada"]
    asignar_uso(manifest, descartadas, "no_usar", "descartada en LabelMe")

    # Podemos aprovechar la validación anterior para entrenar el nuevo modelo.
    if REHACER_VALIDACION:
        validacion_antigua = manifest.index[
            (manifest["uso"] == "val") & (manifest["motivo"] != MOTIVO_VALIDACION)
        ]
        asignar_uso(
            manifest,
            validacion_antigua,
            "train",
            "antigua validacion pasada a train",
        )

    pendientes = manifest[(manifest["estado"] == "valida") & (manifest["uso"] == "pendiente")]

    tipos_incorrectos = pendientes[~pendientes["tipo"].isin(("positiva", "negativa"))]

    if not tipos_incorrectos.empty:
        ids = ", ".join(tipos_incorrectos["tile_id"].head(5))
        raise ValueError(f"Hay imágenes pendientes sin tipo válido: {ids}")

    # La validación se decide antes de mirar los resultados de los modelos.
    aleatorias = pendientes[pendientes["origen"] == "aleatorias"]
    indices_val = elegir_validacion(aleatorias)
    asignar_uso(
        manifest,
        indices_val,
        "val",
        MOTIVO_VALIDACION,
    )

    pendientes = manifest[(manifest["estado"] == "valida") & (manifest["uso"] == "pendiente")]

    # Todos los positivos sirven para ampliar la variedad del entrenamiento.
    positivas = pendientes.index[pendientes["tipo"] == "positiva"]
    asignar_uso(manifest, positivas, "train", "positiva etiquetada")

    pendientes = manifest[(manifest["estado"] == "valida") & (manifest["uso"] == "pendiente")]
    negativas = pendientes[pendientes["tipo"] == "negativa"]

    # De las negativas guardamos todos los falsos positivos y algunos fondos fáciles.
    faciles = analizar_negativas(manifest, negativas)
    cantidad = min(MAX_NEGATIVAS_FACILES, len(faciles))

    if cantidad > 0:
        faciles_train = pd.Index(
            pd.Series(faciles).sample(
                n=cantidad,
                random_state=SEMILLA,
            )
        )
    else:
        faciles_train = pd.Index([])

    asignar_uso(
        manifest,
        faciles_train,
        "train",
        "negativa facil seleccionada",
    )

    faciles_no_usar = faciles.difference(faciles_train)
    asignar_uso(
        manifest,
        faciles_no_usar,
        "no_usar",
        "negativa facil no necesaria",
    )

    return manifest


def mostrar_resumen(manifest: pd.DataFrame) -> None:
    """Muestra el resultado de la selección.

    Args:
        manifest (pd.DataFrame): Manifest actualizado.
    """

    print("\nUso de las imágenes")

    for uso, cantidad in manifest["uso"].value_counts().items():
        print(f"{uso}: {cantidad}")

    print("\nNuevas imágenes por origen y uso")
    nuevas = manifest[manifest["origen"].isin(("aleatorias", "discrepancias"))]
    print(pd.crosstab(nuevas["origen"], nuevas["uso"]))


def main() -> None:
    """Carga el manifest, decide el dataset y guarda el resultado."""

    manifest = cargar_manifest(MANIFEST)
    manifest = decidir_dataset(manifest)
    guardar_manifest(manifest, MANIFEST)
    mostrar_resumen(manifest)


if __name__ == "__main__":
    main()
