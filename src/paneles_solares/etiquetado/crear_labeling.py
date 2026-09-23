"""Selecciona imágenes nuevas y las guarda en labeling para anotarlas."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paneles_solares.datos.coleccion import (
    LABELING,
    MANIFEST,
    MANIFEST_LABELING,
    POOL,
    cargar_manifest,
    guardar_manifest,
    nueva_fila,
)
from paneles_solares.geografia.ortofotos import leer_tesela
from paneles_solares.rutas import ruta_proyecto

# Deja a cero las categorías que no quieras utilizar.
CANTIDAD_ALEATORIAS = 3000
CANTIDAD_DISCREPANCIAS = 1000
CANTIDAD_INCIERTAS = 0
CANTIDAD_MUCHAS_DETECCIONES = 0
CANTIDAD_SIN_DETECCION = 0

# Los modelos solo se cargan si alguna categoría los necesita.
MODELO_YOLO = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")
MODELO_UNET = ruta_proyecto("weights/unet_resnet34.pt")

CONFIANZA_MIN = 0.40
CONFIANZA_MAX = 0.60
UMBRAL_DETECCION = 0.30
UMBRAL_UNET = 0.30
MINIMO_DETECCIONES = 4

# Si el IoU entre ambas predicciones es menor, interesa revisar la imagen.
IOU_MINIMO_ACUERDO = 0.50
MINIMO_PIXELES_PREDICHOS = 10

# Normalización utilizada por el encoder ResNet34 preentrenado en ImageNet.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIACION_IMAGENET = (0.229, 0.224, 0.225)

IMGSZ = 640  # Entrada del modelo; los PNG originales se guardan a 512 × 512.
DEVICE = 0

SEMILLA = 42
MAXIMO_CANDIDATAS = 20000  # None permite recorrer todo el índice.

INDICE = ruta_proyecto("data/manifests/teselas.csv")
PNOA = ruta_proyecto("data/pnoa")


def obtener_cantidades(pendientes: pd.DataFrame) -> dict[str, int]:
    """Obtiene cuántas imágenes se quieren de cada categoría.

    Args:
        pendientes (pd.DataFrame): Imágenes que ya están en labeling.

    Returns:
        dict[str, int]: Cantidades pendientes de seleccionar.
    """

    solicitadas = {
        "aleatorias": CANTIDAD_ALEATORIAS,
        "discrepancias": CANTIDAD_DISCREPANCIAS,
        "inciertas": CANTIDAD_INCIERTAS,
        "muchas_detecciones": CANTIDAD_MUCHAS_DETECCIONES,
        "sin_deteccion": CANTIDAD_SIN_DETECCION,
    }

    if any(cantidad < 0 for cantidad in solicitadas.values()):
        raise ValueError("Las cantidades no pueden ser negativas")

    if sum(solicitadas.values()) == 0:
        raise ValueError("Debes pedir al menos una imagen")

    # Así podemos detener el proceso y continuar sin repetir la cuota completa.
    guardadas = pendientes["origen"].value_counts()
    faltan = {
        categoria: max(0, cantidad - guardadas.get(categoria, 0))
        for categoria, cantidad in solicitadas.items()
    }

    return faltan


def cargar_candidatas(pendientes: pd.DataFrame) -> pd.DataFrame:
    """Obtiene teselas completas que todavía no se han utilizado.

    Args:
        pendientes (pd.DataFrame): Imágenes que ya están en labeling.

    Returns:
        pd.DataFrame: Candidatas ordenadas aleatoriamente.
    """

    manifest = cargar_manifest(MANIFEST)

    usados = set(manifest["tile_id"])
    usados.update(pendientes["tile_id"])

    # También contamos los PNG guardados antes de una posible interrupción.
    for carpeta in (POOL, LABELING):
        for imagen in (carpeta / "images").glob("*.png"):
            usados.add(imagen.stem)

    indice = pd.read_csv(INDICE).fillna("")

    completas = (indice["ancho"] == 512) & (indice["alto"] == 512)
    nuevas = ~indice["tile_id"].isin(usados)

    candidatas = indice[completas & nuevas].sample(
        frac=1,
        random_state=SEMILLA,
    )

    if MAXIMO_CANDIDATAS is not None:
        if MAXIMO_CANDIDATAS < 1:
            raise ValueError("El límite de candidatas debe ser positivo o None")

        candidatas = candidatas.head(MAXIMO_CANDIDATAS)

    return candidatas


def cargar_modelo_yolo(cantidades: dict):
    """Carga YOLO únicamente cuando lo necesita alguna categoría.

    Args:
        cantidades (dict): Cantidad solicitada de cada tipo.

    Returns:
        YOLO | None: Modelo cargado o None para muestreo solo aleatorio.
    """

    cantidad_yolo = (
        cantidades["discrepancias"]
        + cantidades["inciertas"]
        + cantidades["muchas_detecciones"]
        + cantidades["sin_deteccion"]
    )

    if cantidad_yolo == 0:
        return None

    if not 0 <= UMBRAL_DETECCION <= CONFIANZA_MIN <= CONFIANZA_MAX <= 1:
        raise ValueError("Revisa los umbrales de confianza")

    if MINIMO_DETECCIONES < 1:
        raise ValueError("El mínimo de detecciones debe ser positivo")

    if not MODELO_YOLO.is_file():
        raise FileNotFoundError(MODELO_YOLO)

    from ultralytics import YOLO

    return YOLO(str(MODELO_YOLO))


def cargar_modelo_unet(cantidades: dict):
    """Carga U-Net únicamente cuando se solicitan discrepancias.

    Args:
        cantidades (dict): Cantidad solicitada de cada tipo.

    Returns:
        tuple: Modelo U-Net y dispositivo, o dos None si no se necesitan.
    """

    if cantidades["discrepancias"] == 0:
        return None, None

    if not 0 <= UMBRAL_UNET <= 1:
        raise ValueError("El umbral de U-Net debe estar entre 0 y 1")

    if not 0 <= IOU_MINIMO_ACUERDO <= 1:
        raise ValueError("El IoU mínimo debe estar entre 0 y 1")

    if MINIMO_PIXELES_PREDICHOS < 1:
        raise ValueError("El mínimo de píxeles debe ser positivo")

    if not MODELO_UNET.is_file():
        raise FileNotFoundError(MODELO_UNET)

    import torch

    from paneles_solares.modelos.entrenar_unet import crear_modelo

    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo()
    pesos = torch.load(
        MODELO_UNET,
        map_location=dispositivo,
        weights_only=True,
    )
    modelo.load_state_dict(pesos)
    modelo.to(dispositivo)
    modelo.eval()

    return modelo, dispositivo


def categoria_yolo(
    confianza: float,
    detecciones: int,
    faltan: dict,
) -> str:
    """Elige una categoría de YOLO que todavía tenga plazas disponibles.

    Args:
        confianza (float): Mayor confianza encontrada en la imagen.
        detecciones (int): Número de detecciones.
        faltan (dict): Cantidades pendientes de cada categoría.

    Returns:
        str: Categoría elegida o vacío si la imagen no interesa.
    """

    confianza_intermedia = CONFIANZA_MIN <= confianza <= CONFIANZA_MAX

    if faltan["inciertas"] > 0 and detecciones > 0 and confianza_intermedia:
        return "inciertas"

    if faltan["muchas_detecciones"] > 0 and detecciones >= MINIMO_DETECCIONES:
        return "muchas_detecciones"

    if faltan["sin_deteccion"] > 0 and detecciones == 0:
        return "sin_deteccion"

    return ""


def predecir_yolo(modelo, imagen) -> tuple[float, int, np.ndarray]:
    """Obtiene las detecciones de YOLO y las une en una máscara.

    Args:
        modelo: Modelo YOLO cargado.
        imagen: Array RGB leído desde PNOA.

    Returns:
        tuple[float, int, np.ndarray]: Confianza, detecciones y máscara binaria.
    """

    # YOLO recibe los arrays en BGR; PNOA se ha leído en RGB.
    imagen_bgr = imagen[:, :, ::-1].copy()

    resultado = modelo.predict(
        source=imagen_bgr,
        conf=UMBRAL_DETECCION,
        imgsz=IMGSZ,
        device=DEVICE,
        retina_masks=True,
        verbose=False,
    )[0]

    if resultado.boxes is None or len(resultado.boxes) == 0:
        mascara = np.zeros(imagen.shape[:2], dtype=bool)
        return 0.0, 0, mascara

    confianza = float(resultado.boxes.conf.max().item())
    detecciones = len(resultado.boxes)

    if resultado.masks is None:
        mascara = np.zeros(imagen.shape[:2], dtype=bool)
    else:
        mascara = resultado.masks.data.any(dim=0).cpu().numpy()

    # retina_masks debería conservar el tamaño original, pero lo aseguramos.
    if mascara.shape != imagen.shape[:2]:
        ancho = imagen.shape[1]
        alto = imagen.shape[0]
        mascara = Image.fromarray(mascara.astype(np.uint8)).resize(
            (ancho, alto),
            Image.Resampling.NEAREST,
        )
        mascara = np.asarray(mascara, dtype=bool)

    return confianza, detecciones, mascara


def predecir_unet(modelo, imagen, dispositivo) -> np.ndarray:
    """Obtiene la máscara binaria predicha por U-Net.

    Args:
        modelo: Modelo U-Net cargado.
        imagen: Array RGB leído desde PNOA.
        dispositivo: CPU o GPU utilizada por U-Net.

    Returns:
        np.ndarray: Máscara binaria con el mismo tamaño que la imagen.
    """

    import torch

    imagen = imagen.astype(np.float32) / 255.0
    media = np.asarray(MEDIA_IMAGENET, dtype=np.float32)
    desviacion = np.asarray(DESVIACION_IMAGENET, dtype=np.float32)
    imagen = (imagen - media) / desviacion

    # PyTorch espera canales × alto × ancho y una dimensión para el batch.
    tensor = np.transpose(imagen, (2, 0, 1)).copy()
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(dispositivo)

    with torch.inference_mode():
        logits = modelo(tensor)
        probabilidades = torch.sigmoid(logits)

    mascara = probabilidades[0, 0].cpu().numpy() >= UMBRAL_UNET

    return mascara


def son_discrepantes(mascara_yolo: np.ndarray, mascara_unet: np.ndarray) -> bool:
    """Comprueba si las predicciones de YOLO y U-Net difieren lo suficiente.

    Args:
        mascara_yolo (np.ndarray): Máscara binaria predicha por YOLO.
        mascara_unet (np.ndarray): Máscara binaria predicha por U-Net.

    Returns:
        bool: True cuando la imagen resulta interesante para revisar.
    """

    union = np.count_nonzero(mascara_yolo | mascara_unet)

    # Ignoramos pequeñas manchas que ninguno considera una instalación real.
    if union < MINIMO_PIXELES_PREDICHOS:
        return False

    interseccion = np.count_nonzero(mascara_yolo & mascara_unet)
    iou = interseccion / union

    return iou < IOU_MINIMO_ACUERDO


def guardar_muestra(
    imagen,
    fila: dict,
    pendientes: pd.DataFrame,
) -> Path:
    """Guarda un PNG y lo registra inmediatamente en el manifest de labeling.

    Args:
        imagen: Array RGB que se quiere guardar.
        fila (dict): Información de la imagen seleccionada.
        pendientes (pd.DataFrame): Manifest que se ampliará con la nueva imagen.

    Returns:
        Path: Ruta del PNG guardado.
    """

    destino = LABELING / "images" / f"{fila['tile_id']}.png"

    if destino.exists():
        raise FileExistsError(destino)

    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imagen).save(destino)

    pendientes.loc[len(pendientes)] = fila
    guardar_manifest(pendientes, MANIFEST_LABELING)

    return destino


def seleccionar_muestras(
    candidatas: pd.DataFrame,
    pendientes: pd.DataFrame,
    faltan: dict,
    modelo_yolo=None,
    modelo_unet=None,
    dispositivo=None,
) -> None:
    """Recorre las teselas y guarda las que cubren las cantidades solicitadas.

    Args:
        candidatas (pd.DataFrame): Teselas que todavía no se han utilizado.
        pendientes (pd.DataFrame): Manifest de labeling, que se irá ampliando.
        faltan (dict): Cantidades que se irán descontando al guardar imágenes.
        modelo_yolo: YOLO cargado o None si no se necesita.
        modelo_unet: U-Net cargado o None si no se necesita.
        dispositivo: CPU o GPU utilizada por U-Net.
    """

    for _, registro in candidatas.iterrows():
        if sum(faltan.values()) == 0:
            break

        try:
            imagen = leer_tesela(registro, PNOA)
        except (OSError, ValueError) as error:
            print(f"No se pudo leer {registro['tile_id']}: {error}")
            continue

        # Las aleatorias se eligen independientemente de lo que detecte el modelo.
        categoria = ""
        if faltan["aleatorias"] > 0:
            categoria = "aleatorias"

        confianza = None
        detecciones = None

        # Mientras falten aleatorias no hace falta ejecutar ningún modelo.
        if not categoria and modelo_yolo is not None:
            confianza, detecciones, mascara_yolo = predecir_yolo(
                modelo_yolo,
                imagen,
            )

            if modelo_unet is not None and faltan["discrepancias"] > 0:
                mascara_unet = predecir_unet(
                    modelo_unet,
                    imagen,
                    dispositivo,
                )

                if son_discrepantes(mascara_yolo, mascara_unet):
                    categoria = "discrepancias"

            if not categoria:
                categoria = categoria_yolo(confianza, detecciones, faltan)

        if not categoria:
            continue

        fila = nueva_fila(registro.to_dict(), categoria)

        if confianza is not None:
            fila["modelo_yolo"] = str(MODELO_YOLO)
            fila["confianza_yolo"] = confianza
            fila["detecciones_yolo"] = detecciones
            fila["umbral_yolo"] = UMBRAL_DETECCION

        destino = guardar_muestra(imagen, fila, pendientes)
        faltan[categoria] -= 1

        print(f"Guardada: {destino.name} ({categoria})")


def main() -> None:
    """Prepara labeling y selecciona las imágenes que se van a etiquetar."""

    pendientes = cargar_manifest(MANIFEST_LABELING)
    faltan = obtener_cantidades(pendientes)

    if sum(faltan.values()) == 0:
        print("Las cantidades solicitadas ya están completas en labeling.")
        return

    candidatas = cargar_candidatas(pendientes)
    modelo_yolo = cargar_modelo_yolo(faltan)
    modelo_unet, dispositivo = cargar_modelo_unet(faltan)

    (LABELING / "annotations").mkdir(parents=True, exist_ok=True)

    print(f"Candidatas disponibles: {len(candidatas)}")

    try:
        seleccionar_muestras(
            candidatas,
            pendientes,
            faltan,
            modelo_yolo,
            modelo_unet,
            dispositivo,
        )
    except KeyboardInterrupt:
        print("Proceso detenido. Las imágenes guardadas ya están registradas.")

    print(f"Imágenes en labeling: {len(pendientes)}")
    print(f"Cuotas que faltan por completar: {faltan}")


if __name__ == "__main__":
    main()
