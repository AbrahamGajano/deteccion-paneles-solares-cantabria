import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RUTA_IMAGENES = Path("./data/extern/bdappv/ign/img")
RUTA_MASCARAS = Path("./data/extern/bdappv/ign/mask")

RUTA_YOLO = Path("./data/processed/bdappv_yolo")

RATIO_TRAIN = 80
NUM_POS = 4000
NUM_NEG = 4000


def buscar_datos() -> pd.DataFrame:
    """Funcion que busca los datos de las imagenes y sus mascaras

    Returns:
        pd.DataFrame: Retorna el df que marca para cada imagen si tiene mascara
    """
    # Listas de todas las imagenes y mascaras
    imagenes = sorted(RUTA_IMAGENES.glob(pattern="*.png"))
    mascaras = sorted(RUTA_MASCARAS.glob(pattern="*.png"))

    # Crear los dataframes
    imgs_dict = []
    for ruta in imagenes:
        img = {"id": ruta.stem, "ruta_imagen": str(ruta)}
        imgs_dict.append(img)

    mask_dict = []
    for ruta in mascaras:
        mask = {"id": ruta.stem, "ruta_mascara": str(ruta)}
        mask_dict.append(mask)

    df_img = pd.DataFrame(imgs_dict)
    df_mask = pd.DataFrame(mask_dict)

    # Igual que en un SQL al hacer el merge si falta en la izq se pondra None
    df_union = df_img.merge(right=df_mask, how="left", on="id")
    # En cada uno que no tenga su igual en mascara se considerara que no tiene
    df_union["tiene_mascara"] = df_union["ruta_mascara"].notna()

    return df_union


def seleccionar_datos(
    datos: pd.DataFrame, positivos: int, negativos: int
) -> pd.DataFrame:
    """Funcion que selecciona los datos en las cantidades solicitadas dividiendo
    entre datos de entreno y validacion

    Args:
        datos (pd.DataFrame): Dataframe de los datos y con informacion de si son pos
        positivos (int): Numero de positivos
        negativos (int): Numero de negativos

    Raises:
        ValueError: En caso de solicitar mas de los existentes

    Returns:
        pd.DataFrame: Dataframe de los seleccionados que indica si son positivos o no
        y tambien de si son para entreno o no
    """
    # Separamos en positivos y negativos
    positivos_solicitados = datos[datos["tiene_mascara"]]
    negativos_solicitados = datos[~datos["tiene_mascara"]]

    # Debemos comprobar el no pasarnos del limite
    if len(positivos_solicitados) < positivos or len(negativos_solicitados) < negativos:
        raise ValueError("Se han soliciado mas de los posibles")

    # En cada caso nos quedamos con las cantidades solicitadas
    df_positivos = positivos_solicitados.sample(n=positivos)
    df_negativos = negativos_solicitados.sample(n=negativos)

    # Dentro de esa seleccion tenemos que decidir si es un valor de entreno o test
    num_train_pos = int(len(df_positivos) * RATIO_TRAIN)
    num_train_neg = int(len(df_negativos) * RATIO_TRAIN)

    # Inicializamos a val y despues simplemente lo sobreescribo a train los aleatorios
    # IMP: AQUI llamo val a este subgrupo ya que nos servira para ver como de bien
    # Funciona con el dataset frances no con el del PNOA que nos interesa pero al ser
    # De una resolucion similar 20 vs 15 y 400*400px vs 512*512px nos queda al
    # Redimensionar muy parecido pero lo adecuado seria corroborarlo con los test
    # manuales del labelme
    df_positivos["grupo"] = "val"
    df_negativos["grupo"] = "val"

    indx_pos_train = df_positivos.sample(n=num_train_pos).index

    df_positivos.loc[indx_pos_train, "grupo"] = "train"

    indx_neg_train = df_negativos.sample(n=num_train_neg).index

    df_negativos.loc[indx_neg_train, "grupo"] = "train"

    # Debemos reconcatenar
    return pd.concat([df_negativos, df_positivos], ignore_index=True)


def convertir_mascara_yolo(ruta_mascara: Path, ruta_txt: Path) -> None:
    """Funcion que convierte una mascaraen un txt valido para YOLO

    Args:
        ruta_mascara (Path): Ruta de la mascara a procesar
        ruta_txt (Path): Ruta destino del txt
    """
    # Creo la mascara para diff panel de no panel
    mascara = cv2.imread(str(ruta_mascara), cv2.IMREAD_GRAYSCALE)

    # Solo porsiacaso
    mascara_binaria = (mascara > 0).astype(np.uint8) * 255

    # Para Yolo tenemos que feedearle solo los paneles
    contornos, _ = cv2.findContours(
        mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Necesitamos el donde se encuentran los pixeles relativamente [0, 1] no el num abs
    alto, ancho = mascara.shape
    lineas = []

    for contorno in contornos:
        puntos = contorno.reshape(-1, 2)

        # Si es poligono ha de ser mayor a 3 sus vertices
        if len(puntos) < 3:
            continue

        coords = []

        for x, y in puntos:
            coords.append(x / ancho)
            coords.append(y / alto)

        # Yolo necesita un formato tipo {clase} x0 y0 x1 y1 ...
        # Por cada panel siendo la clase 0 para panel_solar
        coordenadas_txt = " ".join(str(val) for val in coords)

        linea = f"0 {coordenadas_txt}"
        lineas.append(linea)

    # A modo de precaucion por si es la primera creada y no existe todavia la ruta
    ruta_txt.parent.mkdir(parents=True, exist_ok=True)
    # Guardamos el txt codificado en utf-8
    ruta_txt.write_text(data="\n".join(lineas), encoding="utf-8")


def preparar_grupo(datos: pd.DataFrame, grupo: str) -> None:
    """Organiza los datos del dataset para el entrenamiento en sus grupos

    Args:
        datos (pd.DataFrame): DataFrame que contiene los datos de todas las imagenes con
        sus mascaras en caso de existir y si son "train" o "val"
        grupo (str): "train" o "val" para saber a que grupo pertenecen

    Raises:
        ValueError: En caso de no haber usado "train" o "val" como argumento
    """

    # Hay que validar que sea un grupo valido
    if grupo not in ["train", "val"]:
        raise ValueError("El grupo debe ser train o val")

    # Preparamos las rutas basandonos en el grupo
    # Ademas de filtrar el df al del grupo
    datos_grupo = datos[datos["grupo"] == grupo]

    ruta_yolo_imagenes = RUTA_YOLO / "images" / grupo
    ruta_yolo_labels = RUTA_YOLO / "labels" / grupo

    ruta_yolo_imagenes.mkdir(parents=True, exist_ok=True)
    ruta_yolo_labels.mkdir(parents=True, exist_ok=True)

    # Recorremos todas las lineas
    for _, row in datos_grupo.iterrows():
        ruta_imagen_origen = Path(row["ruta_imagen"])

        ruta_imagen_destino = ruta_yolo_imagenes / ruta_imagen_origen.name

        ruta_label_destino = ruta_yolo_labels / f"{ruta_imagen_origen.stem}.txt"

        # Copiamos pero NO movemos las imagnes
        shutil.copy2(
            src=ruta_imagen_origen,
            dst=ruta_imagen_destino,
        )

        # Solo en caso de existir llamaremos a la funcion sino
        # ponemos uno vacio
        if row["tiene_mascara"]:
            convertir_mascara_yolo(
                ruta_mascara=Path(row["ruta_mascara"]),
                ruta_txt=ruta_label_destino,
            )
        else:
            ruta_label_destino.write_text("", encoding="utf-8")


def guardar_configuracion() -> None:
    """Funcion para crear el .yaml"""
    contenido = (
        f"path: {RUTA_YOLO.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: panel_solar\n"
    )

    ruta_yaml = RUTA_YOLO / "bdappv.yaml"
    ruta_yaml.write_text(contenido, encoding="utf-8")


def main() -> None:
    """Ejecuta la preparación completa del dataset"""

    print("\nBuscando datos [1/4]")
    df = buscar_datos()

    print("\nSeleccionando datos [2/4]")
    df = seleccionar_datos(datos=df, positivos=NUM_POS, negativos=NUM_NEG)

    print("\nPreparando grupos [3/4]")
    preparar_grupo(datos=df, grupo="train")
    preparar_grupo(datos=df, grupo="val")

    print("\nGuardando .yaml [4/4]")
    guardar_configuracion()
