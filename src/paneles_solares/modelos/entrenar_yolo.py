"""Entrena YOLO y conserva la última ejecución completa de cada modelo."""

from pathlib import Path

from paneles_solares.rutas import reemplazar_carpeta, ruta_proyecto

# Modelo que se quiere mejorar y ubicación del dataset.
NOMBRE_MODELO = "yolo11s"
PESOS_INICIALES = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")
DATASET = ruta_proyecto("data/datasets/yolo/dataset.yaml")
CARPETA_ENTRENOS = ruta_proyecto("runs/entrenamiento/cantabria")

# Límites del entrenamiento.
EPOCHS = 200
HORAS_MAXIMAS = 3.0
PATIENCE = 20

# Tamaño de entrada y uso de la GPU.
IMGSZ = 640  # Los pesos actuales se entrenaron a 640; los PNG siguen a 512.
BATCH = 2
WORKERS = 2
DEVICE = 0

SEMILLA = 42


def entrenar(carpeta: Path) -> None:
    """Carga los pesos elegidos y ejecuta el entrenamiento.

    Args:
        carpeta (Path): Carpeta donde se guardará la ejecución en curso.
    """

    from ultralytics import YOLO

    modelo = YOLO(str(PESOS_INICIALES))

    modelo.train(
        data=str(DATASET),
        epochs=EPOCHS,
        time=HORAS_MAXIMAS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=SEMILLA,
        deterministic=True,
        cache=False,
        optimizer="auto",
        amp=True,
        save=True,
        save_period=-1,  # Solo best.pt y last.pt, sin checkpoints periódicos.
        plots=False,
        project=str(carpeta.parent),
        name=carpeta.name,
        exist_ok=True,
    )


def guardar_entrenamiento(temporal: Path, salida: Path) -> None:
    """Comprueba los pesos y sustituye el entrenamiento anterior.

    Args:
        temporal (Path): Carpeta de la ejecución terminada.
        salida (Path): Ubicación definitiva del modelo.
    """

    for nombre in ("best.pt", "last.pt"):
        pesos = temporal / "weights" / nombre

        if not pesos.is_file():
            raise RuntimeError(f"No se generó {nombre}; se conserva el entrenamiento anterior")

    # Algunas versiones de YOLO guardan muestras aunque plots sea False.
    for archivo in temporal.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in (".png", ".jpg", ".jpeg"):
            archivo.unlink()

    reemplazar_carpeta(temporal, salida)


def main() -> None:
    """Entrena sin retirar los pesos anteriores hasta que los nuevos estén listos."""

    if not PESOS_INICIALES.is_file():
        raise FileNotFoundError(PESOS_INICIALES)

    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)

    salida = CARPETA_ENTRENOS / NOMBRE_MODELO
    temporal = salida.with_name(f".{salida.name}_preparando")

    if temporal.exists():
        raise FileExistsError(f"Quedó un entrenamiento interrumpido: {temporal}")

    entrenar(temporal)
    guardar_entrenamiento(temporal, salida)

    print(f"Pesos y resultados: {salida}")


if __name__ == "__main__":
    main()
