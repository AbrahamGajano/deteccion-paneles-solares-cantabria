from pathlib import Path

from ultralytics import YOLO

RUTA_MODELO = Path("runs/segment/bdappv_completo/weights/best.pt")
RUTA_YAML = Path("data/processed/cantabria_test_yolo/cantabria_test.yaml")


def main() -> None:
    """Evalua el modelo entrenado con BDAPPV sobre las teselas de Cantabria."""

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {RUTA_MODELO}")

    if not RUTA_YAML.exists():
        raise FileNotFoundError(f"No se encontro el YAML: {RUTA_YAML}")

    print("\nCargando el modelo...")
    modelo = YOLO(str(RUTA_MODELO))

    print("\nEvaluando con las imagenes de Cantabria...")
    metricas = modelo.val(
        data=str(RUTA_YAML),
        split="test",
        imgsz=512,
        batch=16,
        device=0,
        workers=4,
        plots=True,
        project="runs/segment",
        name="test_cantabria",
    )

    print("\nMetricas de segmentacion:")
    print(f"Precision: {metricas.seg.mp:.3f}")
    print(f"Recall: {metricas.seg.mr:.3f}")
    print(f"mAP50: {metricas.seg.map50:.3f}")
    print(f"mAP50-95: {metricas.seg.map:.3f}")


if __name__ == "__main__":
    main()
