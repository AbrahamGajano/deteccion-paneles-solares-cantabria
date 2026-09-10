from pathlib import Path

from ultralytics import YOLO

RUTA_YAML = Path("./data/processed/cantabria_yolo/cantabria.yaml")

MODELO_FRANCES = Path("./runs/segment/bdappv_completo/weights/best.pt")


def entrenar(
    nombre: str,
    pesos_iniciales: str | Path = "yolo11n-seg.pt",
) -> None:
    """Entrena un modelo con los datos de Cantabria."""

    print(f"\nEntrenando: {nombre}")
    print(f"Pesos iniciales: {pesos_iniciales}\n")

    modelo = YOLO(str(pesos_iniciales))

    modelo.train(
        data=str(RUTA_YAML),
        imgsz=512,
        epochs=150,
        time=4,
        patience=20,
        batch=16,
        device=0,
        workers=4,
        cache=False,
        optimizer="auto",
        amp=True,
        seed=42,
        deterministic=True,
        project="runs/cantabria",
        name=nombre,
    )


def main():
    entrenar(
        nombre="desde_generico_n",
    )

    entrenar(
        nombre="desde_francia_n",
        pesos_iniciales=MODELO_FRANCES,
    )


if __name__ == "__main__":
    main()
