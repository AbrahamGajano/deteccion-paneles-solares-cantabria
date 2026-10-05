"""Explora umbrales en las mismas teselas de dos campañas, sin cambiar el modelo."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from paneles_solares.datos.anotaciones import crear_mascara_unet, revisar_par
from paneles_solares.datos.validacion import TAM_MODELO, cargar_campania, carpeta_campania, rutas_tesela
from paneles_solares.modelos.evaluar_unet import PX_UMBRAL, calcular_indicadores, clasificar_resultado, crear_visualizacion, predecir
from paneles_solares.modelos.evaluar_validacion import AREA_PIXEL, comparar, entrada_modelo, revisar_etiquetas, resumir
from paneles_solares.modelos.unet import DatasetPaneles, UMBRAL, cargar_modelo
from paneles_solares.rutas import PESOS_UNET, reemplazar_carpeta, ruta_proyecto

UMBRALES = (0.00001, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.05,
            0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70)


def resultados_umbrales(real, probabilidades, tile_id, grupo, umbrales) -> list[dict]:
    resultados = []
    for umbral in umbrales:
        fila = calcular_indicadores(real, probabilidades >= umbral, tile_id)
        fila.update(umbral=umbral, grupo=grupo, categoria=clasificar_resultado(fila))
        resultados.append(fila)
    return resultados


def barrer_campania(modelo, dispositivo, carpeta, seleccion, umbrales) -> pd.DataFrame:
    """Hace una sola inferencia por imagen; no guarda mapas de probabilidades."""
    resultados = []
    for indice, fila in enumerate(seleccion.to_dict("records"), start=1):
        ruta, anotacion = rutas_tesela(carpeta, fila)
        real = crear_mascara_unet(revisar_par(ruta, anotacion)).resize(
            (TAM_MODELO, TAM_MODELO), Image.Resampling.NEAREST)
        with Image.open(ruta) as imagen:
            probabilidades, _ = predecir(modelo, entrada_modelo(imagen), dispositivo)
        resultados.extend(resultados_umbrales(np.asarray(real) > 0, probabilidades,
                                             fila["tile_id"], fila["grupo"], umbrales))
        if indice % 50 == 0:
            print(f"Campaña: {indice}/{len(seleccion)} imágenes.")
    return pd.DataFrame(resultados)


def barrer_referencia(modelo, dispositivo, seleccion, umbrales) -> pd.DataFrame:
    dataset = DatasetPaneles(ruta_proyecto("data/datasets/unet"), "test", aumentar=False)
    posiciones = {ruta.stem: indice for indice, ruta in enumerate(dataset.imagenes)}
    resultados = []
    for indice, fila in enumerate(seleccion.to_dict("records"), start=1):
        imagen, mascara, tile_id = dataset[posiciones[fila["tile_id"]]]
        probabilidades, _ = predecir(modelo, imagen.unsqueeze(0), dispositivo)
        resultados.extend(resultados_umbrales(mascara.squeeze(0).numpy() > 0.5, probabilidades,
                                             tile_id, fila["grupo"], umbrales))
        if indice % 50 == 0:
            print(f"Referencia 2023: {indice}/{len(seleccion)} imágenes.")
    return pd.DataFrame(resultados)


def resumir_barrido(tabla, referencia, umbrales) -> pd.DataFrame:
    resumenes = []
    for umbral in umbrales:
        actual = tabla[tabla["umbral"] == umbral]
        anterior = referencia[referencia["umbral"] == umbral]
        resumen = comparar(actual, anterior)
        resumen.insert(0, "umbral", umbral)
        for indice, fila in resumen.iterrows():
            grupo = anterior if fila["grupo"] == "total" else anterior[anterior["grupo"] == fila["grupo"]]
            base = resumir(grupo)
            resumen.at[indice, "imagenes_fp_2023"] = base["imagenes_falso_positivo"]
            resumen.at[indice, "imagenes_fn_2023"] = base["imagenes_falso_negativo"]
            grupo_actual = actual if fila["grupo"] == "total" else actual[actual["grupo"] == fila["grupo"]]
            negativas = grupo_actual[grupo_actual["pixeles_reales"] == 0]
            resumen.at[indice, "imagenes_negativas_fp_revisable"] = int((negativas["fp"] > PX_UMBRAL).sum())
            resumen.at[indice, "fp_en_negativas"] = int(negativas["fp"].sum())
            resumen.at[indice, "superficie_fp_en_negativas_m2"] = negativas["fp"].sum() * AREA_PIXEL
        resumenes.append(resumen)
    return pd.concat(resumenes, ignore_index=True)


def comparar_presencia(tabla, referencia, umbrales) -> pd.DataFrame:
    """Diagnóstico por tesela; no identifica ni fecha instalaciones individuales."""
    registros = []
    for umbral in umbrales:
        actual = tabla[tabla["umbral"] == umbral].set_index("tile_id")
        anterior = referencia[referencia["umbral"] == umbral].set_index("tile_id").loc[actual.index]
        aparece = (actual["pixeles_predichos"] > 0) & (anterior["pixeles_predichos"] == 0)
        desaparece = (actual["pixeles_predichos"] == 0) & (anterior["pixeles_predichos"] > 0)
        nueva_etiqueta = (actual["pixeles_reales"] > 0) & (anterior["pixeles_reales"] == 0)
        registros.append({
            "umbral": umbral, "apariciones_predichas": int(aparece.sum()),
            "apariciones_en_teselas_ya_positivas_2023": int((aparece & (anterior["pixeles_reales"] > 0)).sum()),
            "desapariciones_predichas": int(desaparece.sum()),
            "desapariciones_en_teselas_positivas_2024": int((desaparece & (actual["pixeles_reales"] > 0)).sum()),
            "cambios_negativa_a_positiva_en_etiquetas": int(nueva_etiqueta.sum()),
            "cambios_de_etiqueta_detectados": int((nueva_etiqueta & aparece).sum()),
        })
    return pd.DataFrame(registros)


def guardar_grafica(resumen, salida) -> None:
    total = resumen[resumen["grupo"] == "total"]
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for metrica in ("dice", "precision", "recall"):
        ejes[0].plot(total["umbral"], total[metrica], marker=".", label=metrica)
    ejes[0].set(xlabel="Umbral", ylabel="Métrica de píxel", ylim=(0, 1))
    ejes[0].legend()
    ejes[1].plot(total["umbral"], 100 * total["sesgo_superficie"], marker=".")
    ejes[1].axhline(0, color="grey", linewidth=0.8)
    ejes[1].set(xlabel="Umbral", ylabel="Sesgo de superficie (%)")
    for eje in ejes:
        eje.set_xscale("log")
        eje.axvline(UMBRAL, color="grey", linestyle="--", label="Umbral original")
        eje.grid(alpha=0.2)
    figura.suptitle("Barrido exploratorio en la muestra etiquetada; no valida un umbral elegido")
    figura.tight_layout()
    figura.savefig(salida / "barrido.png", dpi=150)
    plt.close(figura)


def guardar_comparaciones(modelo, dispositivo, carpeta, seleccion, tabla, candidato, salida) -> None:
    base = tabla[tabla["umbral"] == UMBRAL].set_index("tile_id")
    actual = tabla[tabla["umbral"] == candidato].set_index("tile_id")
    mejora = (base["fn"] - actual["fn"]).sort_values(ascending=False)
    elegidas = mejora.head(2).index.tolist()
    negativos = actual[actual["pixeles_reales"] == 0]
    if not negativos.empty and negativos["fp"].max() > 0:
        elegidas.append(negativos["fp"].idxmax())
    seleccion = seleccion.set_index("tile_id")
    for tile_id in dict.fromkeys(elegidas):
        fila = seleccion.loc[tile_id].to_dict()
        fila["tile_id"] = tile_id
        ruta, anotacion = rutas_tesela(carpeta, fila)
        with Image.open(ruta) as original:
            imagen = original.convert("RGB").resize((TAM_MODELO, TAM_MODELO), Image.Resampling.BILINEAR)
        real = np.asarray(crear_mascara_unet(revisar_par(ruta, anotacion)).resize(
            (TAM_MODELO, TAM_MODELO), Image.Resampling.NEAREST)) > 0
        probabilidades, _ = predecir(modelo, entrada_modelo(imagen), dispositivo)
        comparacion = Image.new("RGB", (2 * TAM_MODELO, TAM_MODELO + 30), "white")
        dibujo = ImageDraw.Draw(comparacion)
        for posicion, umbral in enumerate((UMBRAL, candidato)):
            comparacion.paste(crear_visualizacion(imagen, real, probabilidades >= umbral),
                              (posicion * TAM_MODELO, 0))
            dibujo.text((posicion * TAM_MODELO + 8, TAM_MODELO + 8), f"Umbral {umbral:g}", fill="black")
        comparacion.save(salida / f"comparacion_{tile_id}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campania")
    parser.add_argument("--config", default="validaciones.json")
    parser.add_argument("--umbrales", type=float, nargs="+", default=UMBRALES)
    args = parser.parse_args()
    config = cargar_campania(args.campania, args.config)
    umbrales = sorted(set([*args.umbrales, UMBRAL]))
    if any(not 0 < umbral < 1 for umbral in umbrales):
        parser.error("Los umbrales deben estar entre 0 y 1, sin incluir los extremos")
    carpeta = carpeta_campania(args.campania)
    seleccion, excluidas = revisar_etiquetas(carpeta, pd.read_csv(carpeta / "manifest.csv", keep_default_na=False))
    guardada = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    campos = ("anio", "resolucion_m", "procedencia", "fuentes", "semilla", "aleatorias", "positivas")
    if any(config[campo] != guardada[campo] for campo in campos):
        raise ValueError("La configuración no coincide con la muestra preparada")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = cargar_modelo(PESOS_UNET, dispositivo)
    modelo.requires_grad_(False)
    tabla = barrer_campania(modelo, dispositivo, carpeta, seleccion, umbrales)
    referencia = barrer_referencia(modelo, dispositivo, seleccion, umbrales)
    resumen = resumir_barrido(tabla, referencia, umbrales)
    total = resumen[resumen["grupo"] == "total"]
    candidato = float(total.sort_values("dice", ascending=False).iloc[0]["umbral"])
    salida = ruta_proyecto(f"runs/evaluacion/validacion_{args.campania}/umbrales")
    temporal = salida.with_name(f".{salida.name}_preparando")
    if temporal.exists():
        raise FileExistsError(f"Revisa la salida incompleta: {temporal}")
    temporal.mkdir(parents=True)
    try:
        resumen.to_csv(temporal / "resumen.csv", index=False)
        pd.concat([tabla.assign(anio=config["anio"]), referencia.assign(anio=2023)], ignore_index=True).to_csv(
            temporal / "resultados.csv", index=False)
        comparar_presencia(tabla, referencia, umbrales).to_csv(temporal / "presencia_entre_campanias.csv", index=False)
        with PESOS_UNET.open("rb") as archivo:
            huella = hashlib.sha256()
            for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
                huella.update(bloque)
        informacion = {**config, "umbrales": umbrales, "umbral_original": UMBRAL,
                       "candidato_por_dice": candidato, "sha256_pesos": huella.hexdigest(),
                       "imagenes": len(seleccion), "excluidas": len(excluidas),
                       "fecha_utc": datetime.now(timezone.utc).isoformat(),
                       "alcance": "exploratorio: los umbrales se comparan en las mismas etiquetas; requiere validación independiente",
                       "px_umbral_revision": PX_UMBRAL,
                       "presencia": "por tesela y cualquier píxel predicho; no fecha instalaciones individuales"}
        (temporal / "config.json").write_text(json.dumps(informacion, ensure_ascii=False, indent=2), encoding="utf-8")
        guardar_grafica(resumen, temporal)
        guardar_comparaciones(modelo, dispositivo, carpeta, seleccion, tabla, candidato, temporal)
        reemplazar_carpeta(temporal, salida)
    finally:
        if temporal.exists():
            shutil.rmtree(temporal)
    print(total[["umbral", "dice", "precision", "recall", "sesgo_superficie",
                 "imagenes_falso_positivo", "imagenes_falso_negativo"]].to_string(index=False))
    print(f"Candidato exploratorio por Dice: {candidato:g}. Resultados: {salida}")


if __name__ == "__main__":
    main()
