from ultralytics import YOLO

RUTA_YAML = "./data/processed/bdappv_yolo/bdappv.yaml"


def main():
    """Programa que entrena el modelo"""
    modelo = YOLO("runs/segment/runs/segment/prueba_bdappv/weights/best.pt")

    modelo.train(
        data="data/processed/bdappv_yolo/bdappv.yaml",
        imgsz=512,
        time=7,
        patience=15,
        batch=16,
        device=0,
        workers=4,
        cache=False,
        optimizer="auto",
        amp=True,
        name="bdappv_completo",
    )


if __name__ == "__main__":
    main()
