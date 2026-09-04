from pathlib import Path

import pandas as pd

RUTA_IMAGENES = Path("./data/extern/bdappv/ign/img")
RUTA_MASCARAS = Path("./data/extern/bdappv/ign/mask")

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
        _type_: _description_
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
    """Convierte las zonas blancas de una máscara en polígonos YOLO."""


def preparar_grupo(datos: pd.DataFrame, grupo: str) -> None:
    """Copia imágenes y genera etiquetas para train o val."""


def guardar_configuracion() -> None:
    """Crea el archivo bdappv.yaml utilizado por YOLO."""


def main() -> None:
    """Ejecuta la preparación completa del dataset."""
