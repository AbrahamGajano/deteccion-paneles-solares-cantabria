"""Aplica YOLO o U-Net a las teselas con edificios de Cantabria."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from paneles_solares.rutas import ruta_proyecto

TIPO_MODELO = "yolo"  # "yolo" o "unet".
NOMBRE_MODELO = "yolo11s"
PESOS = ruta_proyecto("runs/entrenamiento/cantabria/yolo11s/weights/best.pt")

INDICE = ruta_proyecto("data/manifests/teselas.csv")
PNOA = ruta_proyecto("data/pnoa")
SALIDA = ruta_proyecto(f"runs/inferencia/cantabria/{NOMBRE_MODELO}")

UMBRAL = 0.30  # Confianza para YOLO; probabilidad de píxel para U-Net.
MIN_PIXELES_UNET = 10
IMGSZ = 640  # Solo YOLO.
DEVICE = 0
TAM_BLOQUE = 128
LOTE = 4


def cargar_modelo():
    """Carga únicamente el modelo seleccionado."""
    if TIPO_MODELO == "yolo":
        from ultralytics import YOLO

        return YOLO(str(PESOS)), DEVICE

    if TIPO_MODELO == "unet":
        import torch

        from paneles_solares.modelos.entrenar_unet import crear_modelo

        dispositivo = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
        modelo = crear_modelo()
        modelo.load_state_dict(torch.load(PESOS, map_location=dispositivo, weights_only=True))
        modelo.to(dispositivo).eval()
        return modelo, dispositivo

    raise ValueError(f"Tipo de modelo desconocido: {TIPO_MODELO}")


def leer_imagen(tif, fila) -> np.ndarray:
    """Lee el RGB de una tesela sin cargar toda la ortofoto."""
    ventana = Window(fila.columna, fila.fila, fila.ancho, fila.alto)
    return np.moveaxis(tif.read([1, 2, 3], window=ventana), 0, -1)


def predecir_yolo(modelo, imagenes: list[np.ndarray]) -> list[tuple]:
    """Devuelve presencia, píxeles, objetos y confianza por imagen."""
    imagenes_bgr = [imagen[:, :, ::-1].copy() for imagen in imagenes]
    resultados = modelo.predict(
        source=imagenes_bgr,
        conf=UMBRAL,
        imgsz=IMGSZ,
        device=DEVICE,
        retina_masks=True,
        verbose=False,
    )
    predicciones = []
    for resultado in resultados:
        objetos = 0 if resultado.boxes is None else len(resultado.boxes)
        pixeles = 0 if resultado.masks is None else int(resultado.masks.data.any(dim=0).sum())
        confianza = "" if not objetos else round(float(resultado.boxes.conf.max().item()), 4)
        predicciones.append((int(objetos > 0), pixeles, objetos, confianza))
    return predicciones


def predecir_unet(modelo, dispositivo, imagenes: list[np.ndarray]) -> list[tuple]:
    """Devuelve presencia y área de la máscara para cada imagen."""
    import torch

    from paneles_solares.modelos.entrenar_unet import DESVIACION_IMAGENET, MEDIA_IMAGENET

    alto, ancho = imagenes[0].shape[:2]
    preparadas = np.stack(imagenes).astype(np.float32) / 255.0
    preparadas = (preparadas - MEDIA_IMAGENET) / DESVIACION_IMAGENET
    preparadas = np.pad(
        preparadas,
        ((0, 0), (0, -alto % 32), (0, -ancho % 32), (0, 0)),
    )
    tensor = torch.from_numpy(preparadas.transpose(0, 3, 1, 2).copy()).to(dispositivo)

    with torch.inference_mode():
        mascara = torch.sigmoid(modelo(tensor))[:, 0, :alto, :ancho] >= UMBRAL
    pixeles = mascara.count_nonzero(dim=(1, 2)).cpu().tolist()
    return [
        (int(cantidad >= MIN_PIXELES_UNET), cantidad if cantidad >= MIN_PIXELES_UNET else 0, "", "")
        for cantidad in pixeles
    ]


def procesar_bloque(modelo, dispositivo, tif, filas: pd.DataFrame) -> pd.DataFrame:
    """Procesa lotes del mismo tamaño y sitúa cada resultado en el mapa."""
    registros = []
    area_pixel = abs(tif.transform.a * tif.transform.e - tif.transform.b * tif.transform.d)

    for _, grupo in filas.groupby(["ancho", "alto"]):
        for inicio in range(0, len(grupo), LOTE):
            lote = grupo.iloc[inicio : inicio + LOTE]
            imagenes = [leer_imagen(tif, fila) for fila in lote.itertuples()]
            if TIPO_MODELO == "yolo":
                predicciones = predecir_yolo(modelo, imagenes)
            else:
                predicciones = predecir_unet(modelo, dispositivo, imagenes)

            for fila, (con_panel, pixeles, objetos, confianza) in zip(
                lote.itertuples(), predicciones
            ):
                x, y = tif.xy(fila.fila + fila.alto / 2, fila.columna + fila.ancho / 2, offset="ul")
                registros.append(
                    {
                        "tile_id": fila.tile_id,
                        "x": x,
                        "y": y,
                        "con_panel": con_panel,
                        "area_paneles_m2": round(pixeles * area_pixel, 2),
                        "detecciones_yolo": objetos,
                        "confianza_yolo": confianza,
                    }
                )

    return pd.DataFrame(registros)


def configuracion() -> dict:
    """Datos que identifican una inferencia para poder continuarla sin mezclarla."""
    return {
        "tipo": TIPO_MODELO,
        "pesos": str(PESOS.resolve()),
        "pesos_fecha": PESOS.stat().st_mtime_ns,
        "indice_fecha": INDICE.stat().st_mtime_ns,
        "umbral": UMBRAL,
        "min_pixeles_unet": MIN_PIXELES_UNET,
        "imgsz": IMGSZ,
        "tam_bloque": TAM_BLOQUE,
    }


def preparar_salida() -> Path:
    """Reanuda la misma ejecución o retira sus bloques si cambió el modelo."""
    partes = SALIDA / "partes"
    archivo_config = SALIDA / "config.json"
    actual = configuracion()
    partes.mkdir(parents=True, exist_ok=True)

    if archivo_config.exists():
        anterior = json.loads(archivo_config.read_text(encoding="utf-8"))
        if anterior != actual:
            for parte in partes.iterdir():
                if parte.suffix in (".csv", ".tmp"):
                    parte.unlink()
            visualizacion = ruta_proyecto(f"runs/visualizacion/cantabria/{NOMBRE_MODELO}")
            for nombre in (
                "mapa_cantabria.html",
                "municipios.csv",
                "energia_mensual.csv",
                "energia_mensual.png",
                "energia_muestras.csv",
                "energia_config.json",
                "mapa_energia.html",
            ):
                (visualizacion / nombre).unlink(missing_ok=True)
    elif any(partes.glob("*.csv")):
        raise FileExistsError(f"Hay bloques sin config.json en {partes}")

    archivo_config.write_text(json.dumps(actual, indent=2), encoding="utf-8")
    return partes


def main() -> None:
    """Recorre el índice y guarda cada bloque al terminarlo."""
    partes = preparar_salida()
    modelo, dispositivo = cargar_modelo()
    indice = pd.read_csv(INDICE)

    for nombre_tif, filas in indice.groupby("tif", sort=True):
        with rasterio.open(PNOA / nombre_tif) as tif:
            for inicio in range(0, len(filas), TAM_BLOQUE):
                salida = partes / f"{Path(nombre_tif).stem}_{inicio:06d}.csv"
                if salida.exists():
                    continue

                bloque = procesar_bloque(
                    modelo, dispositivo, tif, filas.iloc[inicio : inicio + TAM_BLOQUE]
                )
                temporal = salida.with_suffix(".tmp")
                bloque.to_csv(temporal, index=False)
                temporal.replace(salida)
                print(f"{nombre_tif}: {inicio + len(bloque)}/{len(filas)}", flush=True)


if __name__ == "__main__":
    main()
