import csv
from pathlib import Path

from ultralytics import YOLO

RUTA_MODELO = Path("runs/segment/cantabria/desde_cantabria_ronda_2/weights/best.pt")

CARPETA_PREDICCIONES = Path("runs/segment/revision_cantabria_ronda_2_050")

RUTA_YAML = Path("data/processed/cantabria_test_yolo/cantabria_test.yaml")

CARPETA_IMAGENES = Path("data/processed/cantabria_test_yolo/images")

CARPETA_LABELS = Path("data/processed/cantabria_test_yolo/labels")


CONFIANZA_VISUAL = 0.50


def guardar_predicciones(
    modelo: YOLO,
    confianza: float,
) -> None:
    """Guarda las predicciones separadas segun el tipo de resultado."""

    carpeta_fp = CARPETA_PREDICCIONES / "posibles_falsos_positivos"
    carpeta_detectadas = CARPETA_PREDICCIONES / "positivas_detectadas"
    carpeta_no_detectadas = CARPETA_PREDICCIONES / "positivas_sin_deteccion"

    carpeta_fp.mkdir(parents=True, exist_ok=True)
    carpeta_detectadas.mkdir(parents=True, exist_ok=True)
    carpeta_no_detectadas.mkdir(parents=True, exist_ok=True)

    resultados = modelo.predict(
        source=str(CARPETA_IMAGENES),
        imgsz=512,
        conf=confianza,
        device=0,
        retina_masks=True,
        stream=True,
        verbose=False,
    )

    resumen = []

    cantidad_fp = 0
    cantidad_detectadas = 0
    cantidad_no_detectadas = 0

    for resultado in resultados:
        nombre_imagen = Path(resultado.path).name
        nombre_txt = f"{Path(nombre_imagen).stem}.txt"
        ruta_label = CARPETA_LABELS / nombre_txt

        tiene_panel_real = ruta_label.exists() and bool(
            ruta_label.read_text(encoding="utf-8").strip()
        )

        tiene_predicciones = resultado.boxes is not None and len(resultado.boxes) > 0

        if not tiene_predicciones:
            if tiene_panel_real:
                resultado.save(filename=str(carpeta_no_detectadas / nombre_imagen))

                resumen.append(
                    {
                        "imagen": nombre_imagen,
                        "tipo": "positiva_sin_deteccion",
                        "detecciones": 0,
                        "confianza_maxima": "",
                    }
                )

                cantidad_no_detectadas += 1

            continue

        confianzas = resultado.boxes.conf.cpu().tolist()
        confianza_maxima = max(confianzas)

        nombre_salida = f"{confianza_maxima:.3f}_{nombre_imagen}"

        if tiene_panel_real:
            carpeta_salida = carpeta_detectadas
            tipo = "positiva_detectada"
            cantidad_detectadas += 1
        else:
            carpeta_salida = carpeta_fp
            tipo = "posible_falso_positivo"
            cantidad_fp += 1

        resultado.save(filename=str(carpeta_salida / nombre_salida))

        resumen.append(
            {
                "imagen": nombre_imagen,
                "tipo": tipo,
                "detecciones": len(confianzas),
                "confianza_maxima": round(confianza_maxima, 4),
            }
        )

    ruta_csv = CARPETA_PREDICCIONES / "resumen_predicciones.csv"

    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=[
                "imagen",
                "tipo",
                "detecciones",
                "confianza_maxima",
            ],
        )

        escritor.writeheader()
        escritor.writerows(resumen)

    print(f"\nRevision visual con confianza minima {confianza:.2f}")
    print(f"Posibles falsos positivos: {cantidad_fp}")
    print(f"Positivas detectadas: {cantidad_detectadas}")
    print(f"Positivas sin deteccion: {cantidad_no_detectadas}")
    print(f"Resultados guardados en: {CARPETA_PREDICCIONES}")


def main() -> None:
    """Evalua el modelo y prepara su revision visual."""

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {RUTA_MODELO}")

    if not RUTA_YAML.exists():
        raise FileNotFoundError(f"No se encontro el YAML: {RUTA_YAML}")

    print("\nCargando modelo...")
    modelo = YOLO(str(RUTA_MODELO))

    print("\nEvaluando sobre las 1000 imagenes de test...")
    metricas = modelo.val(
        data=str(RUTA_YAML),
        split="test",
        imgsz=512,
        batch=16,
        device=0,
        workers=4,
        plots=True,
        project="runs/segment",
        name="test_cantabria_ronda_1",
    )

    print("\nMetricas de segmentacion:")
    print(f"Precision: {metricas.seg.mp:.3f}")
    print(f"Recall: {metricas.seg.mr:.3f}")
    print(f"mAP50: {metricas.seg.map50:.3f}")
    print(f"mAP50-95: {metricas.seg.map:.3f}")

    print("\nPreparando revision visual...")
    guardar_predicciones(
        modelo=modelo,
        confianza=CONFIANZA_VISUAL,
    )


if __name__ == "__main__":
    main()
