import json
import random
import shutil
from pathlib import Path

from preparar_test_cantabria import convierte_json_yolo

CARPETA_IMAGENES = Path("./data/labeling/train_val/images")
CARPETA_JSON = Path("./data/labeling/train_val/annotations")

RUTA_YOLO = Path("./data/processed/cantabria_yolo")
RUTA_YAML = RUTA_YOLO / "cantabria.yaml"

PROPORCION_VAL = 0.20
NEGATIVAS_TRAIN = 600
SEMILLA = 42


def cargar_registros() -> list[dict]:
    """Funcion que carga la informacion necesaria de los json

    Se eliminan las imagenes dudosas y se distingue entre imagenes
    positivas, negativas normales y negativas dificiles.

    Returns:
        list[dict]: Lista con los registros que se pueden utilizar
    """

    registros = []

    for ruta_json in sorted(CARPETA_JSON.glob("*.json")):
        datos = json.loads(ruta_json.read_text(encoding="utf-8"))
        flags = datos.get("flags") or {}

        # Las dudosas no las utilizaremos por ahora
        if flags.get("dudosa") is True:
            continue

        # Comprobamos si la imagen contiene algun panel valido
        formas = [
            forma
            for forma in datos.get("shapes", [])
            if forma.get("label") == "panel_solar" and len(forma.get("points", [])) >= 3
        ]

        registros.append(
            {
                "json": ruta_json,
                "imagen": CARPETA_IMAGENES / f"{ruta_json.stem}.png",
                "positiva": bool(formas),
                "dificil": flags.get("dificil") is True,
            }
        )

    return registros


def separar(
    grupo: list[dict],
    proporcion_val: float,
    generador: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Funcion que divide un grupo entre entrenamiento y validacion

    Args:
        grupo (list[dict]): Registros que se quieren dividir
        proporcion_val (float): Proporcion reservada para validacion
        generador (random.Random): Generador aleatorio utilizado

    Returns:
        tuple[list[dict], list[dict]]: Registros de train y validacion
    """

    grupo = grupo.copy()
    generador.shuffle(grupo)

    cantidad_val = round(len(grupo) * proporcion_val)

    val = grupo[:cantidad_val]
    train = grupo[cantidad_val:]

    return train, val


def dividir(
    registros: list[dict],
    proporcion_val: float = PROPORCION_VAL,
    cantidad_negativas_train: int = NEGATIVAS_TRAIN,
    semilla: int = SEMILLA,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Funcion que prepara el reparto de entrenamiento y validacion

    Se utilizan todas las imagenes positivas. Para las negativas se da
    prioridad a las dificiles y se completa con negativas normales.

    Args:
        registros (list[dict]): Registros disponibles
        proporcion_val (float): Proporcion utilizada para validacion
        cantidad_negativas_train (int): Numero de negativas para train
        semilla (int): Semilla utilizada en el reparto

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Train, val y pool
    """

    generador = random.Random(semilla)

    # Dividimos los registros en los tres tipos que nos interesan
    positivas = [registro for registro in registros if registro["positiva"]]

    negativas_dificiles = [
        registro
        for registro in registros
        if not registro["positiva"] and registro["dificil"]
    ]

    negativas_normales = [
        registro
        for registro in registros
        if not registro["positiva"] and not registro["dificil"]
    ]

    # Cada tipo se divide por separado para mantener las proporciones
    train_positivas, val_positivas = separar(
        positivas,
        proporcion_val,
        generador,
    )

    candidatas_dificiles, val_dificiles = separar(
        negativas_dificiles,
        proporcion_val,
        generador,
    )

    candidatas_normales, val_normales = separar(
        negativas_normales,
        proporcion_val,
        generador,
    )

    # Al poner primero las dificiles les damos prioridad
    candidatas_negativas = candidatas_dificiles + candidatas_normales

    train_negativas = candidatas_negativas[:cantidad_negativas_train]

    # Guardamos las sobrantes por si se quieren utilizar mas adelante
    pool = candidatas_negativas[cantidad_negativas_train:]

    train = train_positivas + train_negativas
    val = val_positivas + val_dificiles + val_normales

    generador.shuffle(train)
    generador.shuffle(val)

    return train, val, pool


def preparar_division(
    registros: list[dict],
    division: str,
) -> None:
    """Funcion que copia las imagenes y genera las etiquetas de YOLO

    Args:
        registros (list[dict]): Registros que se quieren preparar
        division (str): Division de destino, train o val

    Raises:
        FileNotFoundError: Salta si no se encuentra una imagen
    """

    carpeta_imagenes = RUTA_YOLO / "images" / division
    carpeta_labels = RUTA_YOLO / "labels" / division

    carpeta_imagenes.mkdir(parents=True, exist_ok=True)
    carpeta_labels.mkdir(parents=True, exist_ok=True)

    for registro in registros:
        if not registro["imagen"].exists():
            print(f"No se encuentra la imagen {registro['imagen']}")
            raise FileNotFoundError(registro["imagen"])

        shutil.copy2(
            registro["imagen"],
            carpeta_imagenes / registro["imagen"].name,
        )

        # Reutilizamos el conversor que ya teniamos para el test
        convierte_json_yolo(
            ruta_json_ori=registro["json"],
            ruta_txt_dst=(carpeta_labels / f"{registro['json'].stem}.txt"),
        )


def preparar_yaml() -> None:
    """Funcion que crea el archivo de configuracion para Ultralytics"""

    contenido = (
        f"path: {RUTA_YOLO.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: panel_solar\n"
    )

    RUTA_YAML.write_text(
        contenido,
        encoding="utf-8",
    )


def contar_positivas(registros: list[dict]) -> int:
    """Funcion que cuenta cuantas imagenes positivas hay

    Args:
        registros (list[dict]): Lista de registros

    Returns:
        int: Numero de imagenes positivas
    """

    return sum(registro["positiva"] for registro in registros)


def main() -> None:
    """Programa que prepara los datos de Cantabria para entrenar YOLO"""

    # Si tiene contenido evitamos mezclarlo con una ejecucion anterior
    if RUTA_YOLO.exists() and any(RUTA_YOLO.iterdir()):
        print(f"La carpeta {RUTA_YOLO} ya contiene datos")
        raise FileExistsError

    RUTA_YOLO.mkdir(parents=True, exist_ok=True)

    print("\nCargando anotaciones [1/4]")
    registros = cargar_registros()

    print("\nSeparando train y validacion [2/4]")
    train, val, pool = dividir(
        registros=registros,
        proporcion_val=PROPORCION_VAL,
        cantidad_negativas_train=NEGATIVAS_TRAIN,
        semilla=SEMILLA,
    )

    print("\nPreparando imagenes y etiquetas [3/4]")
    preparar_division(train, "train")
    preparar_division(val, "val")

    print("\nPreparando YAML [4/4]")
    preparar_yaml()

    positivas_train = contar_positivas(train)
    positivas_val = contar_positivas(val)

    print("\nReparto terminado")
    print(
        f"Train: {positivas_train} positivas y {len(train) - positivas_train} negativas"
    )
    print(f"Val: {positivas_val} positivas y {len(val) - positivas_val} negativas")
    print(f"Negativas reservadas: {len(pool)}")


if __name__ == "__main__":
    main()
