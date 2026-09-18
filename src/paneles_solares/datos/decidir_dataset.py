"""Decide qué muestras se utilizan en train, val y test."""

import random

import pandas as pd

from paneles_solares.datos.anotaciones import excluida, revisar_par
from paneles_solares.datos.coleccion import (
    MANIFEST,
    POOL,
    SPLITS,
    cargar_manifest,
    guardar_manifest,
    rutas_muestra,
)
from paneles_solares.rutas import ruta_proyecto

# Porcentaje de las nuevas muestras aleatorias que irá a cada split.
FRACCIONES = {
    "train": 1.0,
    "val": 0.0,
    "test": 0.0,
}

# Cantidad de fondos que queremos añadir a entrenamiento.
MAX_NEGATIVAS_DIFICILES = None
MAX_NEGATIVAS_FACILES = 0

# Modelo utilizado para localizar negativas difíciles.
MODELO_YOLO = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")

UMBRAL_NEGATIVA_DIFICIL = 0.30
IMGSZ = 640
DEVICE = 0

# Las teselas próximas deben permanecer en el mismo split.
TAM_GRUPO = 2048
MARGEN_SEPARACION = 512

SEMILLA = 42


def obtener_grupo(fila: pd.Series) -> tuple:
    """Obtiene el grupo espacial de una muestra.

    Args:
        fila (pd.Series): Fila del manifest.

    Returns:
        tuple: Identificador del grupo espacial.
    """

    # Las imágenes manuales sin posición forman grupos independientes.
    if not fila["tif"] or fila["fila"] == "" or fila["columna"] == "":
        return "manual", fila["tile_id"]

    grupo_fila = int(float(fila["fila"])) // TAM_GRUPO
    grupo_columna = int(float(fila["columna"])) // TAM_GRUPO

    return fila["tif"], grupo_fila, grupo_columna


def son_cercanas(a: dict, b: dict) -> bool:
    """Comprueba si dos muestras pertenecen a zonas cercanas.

    Args:
        a (dict): Primera muestra.
        b (dict): Segunda muestra.

    Returns:
        bool: True si las muestras son cercanas.
    """

    if not a["tif"] or a["tif"] != b["tif"]:
        return False

    campos = ("fila", "columna", "ancho", "alto")

    if any(a[campo] == "" or b[campo] == "" for campo in campos):
        return False

    fila_a = float(a["fila"])
    columna_a = float(a["columna"])
    ancho_a = float(a["ancho"])
    alto_a = float(a["alto"])

    fila_b = float(b["fila"])
    columna_b = float(b["columna"])
    ancho_b = float(b["ancho"])
    alto_b = float(b["alto"])

    separados_horizontalmente = (
        columna_a >= columna_b + ancho_b + MARGEN_SEPARACION
        or columna_b >= columna_a + ancho_a + MARGEN_SEPARACION
    )

    separados_verticalmente = (
        fila_a >= fila_b + alto_b + MARGEN_SEPARACION
        or fila_b >= fila_a + alto_a + MARGEN_SEPARACION
    )

    return not separados_horizontalmente and not separados_verticalmente


def hay_conflicto(
    muestras: list[dict],
    reservadas: list[dict],
    split: str,
) -> bool:
    """Comprueba si un grupo está cerca de otro split.

    Args:
        muestras (list[dict]): Muestras que se quieren asignar.
        reservadas (list[dict]): Muestras que ya tienen un split.
        split (str): Split elegido para el nuevo grupo.

    Returns:
        bool: True si existe contaminación entre splits.
    """

    for muestra in muestras:
        for reservada in reservadas:
            if reservada["uso"] != split and son_cercanas(muestra, reservada):
                return True

    return False


def asignar_splits(
    pendientes: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, str]:
    """Reparte las muestras pendientes entre train, val y test.

    Args:
        pendientes (pd.DataFrame): Muestras válidas pendientes.
        manifest (pd.DataFrame): Manifest completo.

    Returns:
        dict[str, str]: Split asignado a cada tile_id.
    """

    if abs(sum(FRACCIONES.values()) - 1) > 1e-6:
        raise ValueError("Las fracciones de train, val y test deben sumar 1")

    # Puede no quedar ninguna pendiente después de releer las marcas de LabelMe.
    if pendientes.empty:
        return {}

    pendientes = pendientes.copy()

    # Las teselas próximas se sortean como un único grupo.
    pendientes["_grupo"] = pendientes.apply(
        obtener_grupo,
        axis=1,
    )

    grupos = [grupo.to_dict("records") for _, grupo in pendientes.groupby("_grupo", sort=False)]

    rng = random.Random(SEMILLA)
    rng.shuffle(grupos)

    pesos = [FRACCIONES[split] for split in SPLITS]

    # Partimos de las imágenes que ya tienen un split asignado.
    reservadas = manifest[manifest["uso"].isin(SPLITS)].to_dict("records")

    asignaciones = {}

    for muestras in grupos:
        # Las muestras buscadas mediante fallos de YOLO solo pueden ir a train.
        muestra_aleatoria = all(
            muestra["origen"] in ("aleatorias", "manual") for muestra in muestras
        )

        if muestra_aleatoria:
            split = rng.choices(
                SPLITS,
                weights=pesos,
            )[0]
        else:
            split = "train"

        conflicto = hay_conflicto(
            muestras,
            reservadas,
            split,
        )

        for muestra in muestras:
            tile_id = muestra["tile_id"]

            if conflicto:
                asignaciones[tile_id] = "no_usar"
            else:
                asignaciones[tile_id] = split

                nueva_reserva = muestra.copy()
                nueva_reserva["uso"] = split
                reservadas.append(nueva_reserva)

    return asignaciones


def actualizar_anotaciones(manifest: pd.DataFrame) -> pd.DataFrame:
    """Actualiza la información de los JSON pendientes.

    Args:
        manifest (pd.DataFrame): Manifest completo.

    Returns:
        pd.DataFrame: Manifest actualizado.
    """

    pendientes = manifest[(manifest["estado"] == "valida") & (manifest["uso"] == "pendiente")]

    # Releemos los JSON por si se modificaron después de incorporarlos.
    for indice, fila in pendientes.iterrows():
        imagen, anotacion = rutas_muestra(
            POOL,
            fila["tile_id"],
        )

        datos = revisar_par(imagen, anotacion)

        if excluida(datos):
            manifest.loc[indice, "estado"] = "descartada"
            manifest.loc[indice, "uso"] = "no_usar"
            manifest.loc[indice, "motivo"] = "marcada en LabelMe"
            continue

        manifest.loc[indice, "tipo"] = "positiva" if datos["shapes"] else "negativa"

        manifest.loc[indice, "instancias"] = len(datos["shapes"])

    return manifest


def predecir_yolo(modelo, tile_id: str) -> tuple[float, int]:
    """Ejecuta YOLO sobre una imagen negativa.

    Args:
        modelo: Modelo YOLO cargado.
        tile_id (str): Identificador de la imagen.

    Returns:
        tuple[float, int]: Confianza máxima y número de detecciones.
    """

    imagen, _ = rutas_muestra(POOL, tile_id)

    resultado = modelo.predict(
        source=str(imagen),
        conf=UMBRAL_NEGATIVA_DIFICIL,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )[0]

    if resultado.boxes is None or len(resultado.boxes) == 0:
        return 0.0, 0

    confianza = float(resultado.boxes.conf.max().item())
    detecciones = len(resultado.boxes)

    return confianza, detecciones


def seleccionar_muestras(manifest: pd.DataFrame) -> pd.DataFrame:
    """Decide qué muestras pendientes se utilizarán finalmente.

    Args:
        manifest (pd.DataFrame): Manifest completo.

    Returns:
        pd.DataFrame: Manifest con las decisiones actualizadas.
    """

    pendientes = manifest[
        (manifest["estado"] == "valida") & (manifest["uso"] == "pendiente")
    ].copy()

    # El orden aleatorio evita favorecer los primeros tile_id al llenar los cupos.
    pendientes = pendientes.sample(
        frac=1,
        random_state=SEMILLA,
    )

    asignaciones = asignar_splits(
        pendientes,
        manifest,
    )

    negativas_dificiles = 0
    negativas_faciles = 0
    modelo = None

    for indice, fila in pendientes.iterrows():
        split = asignaciones[fila["tile_id"]]

        if split == "no_usar":
            manifest.loc[indice, "uso"] = "no_usar"
            manifest.loc[indice, "motivo"] = "cercana a otro conjunto"
            continue

        # Val y test conservan la distribución natural de las imágenes.
        if split in ("val", "test"):
            manifest.loc[indice, "uso"] = split
            manifest.loc[indice, "motivo"] = "evaluacion aleatoria"
            continue

        # Todas las imágenes positivas se utilizan en train.
        if fila["tipo"] == "positiva":
            manifest.loc[indice, "uso"] = "train"
            manifest.loc[indice, "motivo"] = "positiva"
            continue

        # Una detección de YOLO en una imagen negativa es un falso positivo.
        if modelo is None:
            if not MODELO_YOLO.is_file():
                raise FileNotFoundError(MODELO_YOLO)

            from ultralytics import YOLO

            modelo = YOLO(str(MODELO_YOLO))

        confianza, detecciones = predecir_yolo(
            modelo,
            fila["tile_id"],
        )

        manifest.loc[indice, "modelo_yolo"] = str(MODELO_YOLO)
        manifest.loc[indice, "confianza_yolo"] = confianza
        manifest.loc[indice, "detecciones_yolo"] = detecciones
        manifest.loc[indice, "umbral_yolo"] = UMBRAL_NEGATIVA_DIFICIL

        # YOLO ya aplica el umbral, por lo que cualquier detección es difícil.
        if detecciones > 0:
            incluir = (
                MAX_NEGATIVAS_DIFICILES is None or negativas_dificiles < MAX_NEGATIVAS_DIFICILES
            )

            if incluir:
                negativas_dificiles += 1
                motivo = "negativa dificil"
            else:
                motivo = "cupo de negativas dificiles cubierto"

        else:
            incluir = negativas_faciles < MAX_NEGATIVAS_FACILES

            if incluir:
                negativas_faciles += 1
                motivo = "negativa facil"
            else:
                motivo = "negativa facil no seleccionada"

        manifest.loc[indice, "uso"] = "train" if incluir else "no_usar"

        manifest.loc[indice, "motivo"] = motivo

    return manifest


def main() -> None:
    """Actualiza el manifest con el uso de las muestras pendientes."""

    manifest = cargar_manifest(MANIFEST)

    pendientes = (manifest["estado"] == "valida") & (manifest["uso"] == "pendiente")

    cantidad = pendientes.sum()

    if cantidad == 0:
        print("No hay muestras válidas pendientes de decidir.")
        return

    print(f"Revisando {cantidad} muestras pendientes.")

    manifest = actualizar_anotaciones(manifest)
    manifest = seleccionar_muestras(manifest)

    guardar_manifest(manifest, MANIFEST)

    # Resumen del reparto guardado en el manifest.
    for uso in (*SPLITS, "pendiente", "no_usar"):
        cantidad = (manifest["uso"] == uso).sum()
        print(f"{uso}: {cantidad}")

    print("Ahora puedes ejecutar sincronizar.py.")


if __name__ == "__main__":
    main()
