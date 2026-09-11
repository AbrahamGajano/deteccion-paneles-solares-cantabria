from pathlib import Path

from ultralytics import YOLO

RUTA_MODELO = Path("runs/segment/runs/cantabria/desde_francia_n/weights/best.pt")
RUTA_YAML = Path("data/processed/cantabria_test_yolo/cantabria_test.yaml")

CARPETA_IMAGENES = Path("data/processed/cantabria_test_yolo/images")
CARPETA_PREDICCIONES = Path("runs/segment/predicciones_test_francia_050")


def guardar_predicciones(
    modelo: YOLO,
    confianza: float = 0.50,
) -> None:
    """Guarda las imagenes donde el modelo encuentra algun panel."""

    CARPETA_PREDICCIONES.mkdir(parents=True, exist_ok=True)

    resultados = modelo.predict(
        source=str(CARPETA_IMAGENES),
        imgsz=512,
        conf=confianza,
        device=0,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    cantidad = 0

    for resultado in resultados:
        if resultado.boxes is None or len(resultado.boxes) == 0:
            continue

        nombre_imagen = Path(resultado.path).name
        ruta_salida = CARPETA_PREDICCIONES / nombre_imagen

        resultado.save(filename=str(ruta_salida))
        cantidad += 1

    print(
        f"\nImagenes con predicciones mayores o iguales a {confianza:.2f}: {cantidad}"
    )
    print(f"Guardadas en: {CARPETA_PREDICCIONES}")


def main() -> None:
    """Evalua y guarda las predicciones del modelo."""

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {RUTA_MODELO}")

    if not RUTA_YAML.exists():
        raise FileNotFoundError(f"No se encontro el YAML: {RUTA_YAML}")

    if not CARPETA_IMAGENES.exists():
        raise FileNotFoundError(f"No se encontraron las imagenes: {CARPETA_IMAGENES}")

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

    print("\nGenerando predicciones visuales...")
    guardar_predicciones(
        modelo=modelo,
        confianza=0.50,
    )


if __name__ == "__main__":
    main()
