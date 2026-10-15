"""Evalúa una campaña etiquetada y compara las mismas localizaciones a 15 cm."""

import argparse
import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from PIL import Image

from paneles_solares.datos.anotaciones import crear_mascara_unet, excluida, revisar_par
from paneles_solares.datos.validacion import (
    CAMPOS_MUESTRA,
    EXTENSION_M,
    GRUPOS,
    TAM_MODELO,
    cargar_campania,
    carpeta_campania,
    rutas_tesela,
)
from paneles_solares.modelos.evaluar_unet import (
    CATEGORIAS_REVISION,
    calcular_indicadores,
    clasificar_resultado,
    crear_visualizacion,
    predecir,
)
from paneles_solares.modelos.metricas import calcular_metricas
from paneles_solares.modelos.unet import (
    UMBRAL,
    DatasetPaneles,
    cargar_modelo,
    normalizar_rgb,
)
from paneles_solares.rutas import (
    DATASET_UNET,
    EVALUACION_UNET,
    PESOS_UNET,
    huella_archivo,
    reemplazar_carpeta,
    ruta_proyecto,
)

METRICAS = ("dice", "iou", "precision", "recall", "sesgo_superficie")
AREA_PIXEL = (EXTENSION_M / TAM_MODELO) ** 2


def revisar_etiquetas(
    carpeta, seleccion: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No confunde imágenes pendientes con negativos; exige revisión explícita.

    Args:
        carpeta: Carpeta de datos de la campaña.
        seleccion: Teselas seleccionadas y sus metadatos.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Selección válida y exclusiones explícitas con
            su motivo.

    Raises:
        ValueError: Faltan imágenes, JSON o revisión explícita, o sus dimensiones son
            incompatibles.
    """

    pendientes, descartadas = [], []
    for fila in seleccion.to_dict("records"):
        imagen, anotacion = rutas_tesela(carpeta, fila)
        if not imagen.is_file() or not anotacion.is_file():
            pendientes.append(fila["tile_id"])
            continue
        datos = revisar_par(imagen, anotacion)
        if (datos["imageWidth"], datos["imageHeight"]) != (
            fila["ancho_nativo"],
            fila["alto_nativo"],
        ):
            raise ValueError(f"Dimensiones distintas del manifiesto: {imagen}")
        if excluida(datos):
            descartadas.append({**fila, "motivo": "dudosa o descartar en LabelMe"})
        elif not datos.get("flags", {}).get("revisada", False):
            pendientes.append(fila["tile_id"])
    if pendientes:
        raise ValueError(
            f"Faltan imágenes, JSON o revisión explícita en {len(pendientes)} "
            f"teselas. Primera: {pendientes[0]}. Marca revisada y guarda también "
            f"los negativos."
        )
    ids = {fila["tile_id"] for fila in descartadas}
    validas = seleccion[~seleccion["tile_id"].isin(ids)].copy()
    if validas.empty:
        raise ValueError("No hay teselas revisadas válidas")
    return validas, pd.DataFrame(descartadas, columns=[*seleccion.columns, "motivo"])


def entrada_modelo(imagen: Image.Image) -> torch.Tensor:
    """Prepara una imagen RGB a 512 píxeles con normalización ImageNet.

    Args:
        imagen: Imagen RGB que se procesa.

    Returns:
        torch.Tensor: Tensor RGB normalizado de una imagen a 512 × 512.
    """

    imagen = imagen.convert("RGB").resize(
        (TAM_MODELO, TAM_MODELO), Image.Resampling.BILINEAR
    )
    rgb = normalizar_rgb(np.asarray(imagen))
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)).copy()).unsqueeze(0)


def evaluar_campania(
    modelo, dispositivo, carpeta, seleccion: pd.DataFrame
) -> pd.DataFrame:
    """Evalúa las teselas revisadas y calcula sus superficies planimétricas.

    Args:
        modelo: Modelo cargado para esta fase.
        dispositivo: CPU o GPU utilizada por el modelo.
        carpeta: Carpeta de datos de la campaña.
        seleccion: Teselas seleccionadas y sus metadatos.

    Returns:
        pd.DataFrame: Métricas y superficies por tesela revisada.
    """

    resultados = []
    for indice, fila in enumerate(seleccion.to_dict("records"), start=1):
        ruta, anotacion = rutas_tesela(carpeta, fila)
        datos = revisar_par(ruta, anotacion)
        real = crear_mascara_unet(datos).resize(
            (TAM_MODELO, TAM_MODELO), Image.Resampling.NEAREST
        )
        with Image.open(ruta) as imagen:
            probabilidades, predicha = predecir(
                modelo, entrada_modelo(imagen), dispositivo
            )
        resultado = calcular_indicadores(
            np.asarray(real) > 0, predicha, fila["tile_id"]
        )
        resultado["categoria"] = clasificar_resultado(resultado)
        resultado["probabilidad_maxima"] = float(probabilidades.max())
        resultados.append({**fila, **resultado})
        if indice % 25 == 0:
            print(f"Evaluadas {indice}/{len(seleccion)} teselas de la campaña.")
    tabla = pd.DataFrame(resultados)
    tabla["superficie_real_m2"] = tabla["pixeles_reales"] * AREA_PIXEL
    tabla["superficie_predicha_m2"] = tabla["pixeles_predichos"] * AREA_PIXEL
    tabla["diferencia_superficie_m2"] = tabla["diferencia_superficie"] * AREA_PIXEL
    return tabla


def evaluar_referencia(modelo, dispositivo, seleccion: pd.DataFrame) -> pd.DataFrame:
    """Vuelve a evaluar solo el subconjunto de 2023 con los mismos pesos actuales.

    Args:
        modelo: Modelo cargado para esta fase.
        dispositivo: CPU o GPU utilizada por el modelo.
        seleccion: Teselas seleccionadas y sus metadatos.

    Returns:
        pd.DataFrame: Métricas de 2023 para las localizaciones seleccionadas.
    """

    dataset = DatasetPaneles(DATASET_UNET, "test", aumentar=False)
    posiciones = {ruta.stem: indice for indice, ruta in enumerate(dataset.imagenes)}
    resultados = []
    for fila in seleccion.to_dict("records"):
        imagen, mascara, tile_id = dataset[posiciones[fila["tile_id"]]]
        _, predicha = predecir(modelo, imagen.unsqueeze(0), dispositivo)
        real = mascara.squeeze(0).numpy() > 0.5
        resultado = calcular_indicadores(real, predicha, tile_id)
        resultado["categoria"] = clasificar_resultado(resultado)
        resultados.append({"grupo": fila["grupo"], **resultado})
    return pd.DataFrame(resultados)


def resumir(tabla: pd.DataFrame) -> dict:
    """Agrega píxeles como evaluar_unet; separa errores de píxel e imagen.

    Args:
        tabla: Resultados por tesela que se resumen.

    Returns:
        dict: Métricas agregadas de píxel, imagen y superficie.
    """

    tp, fp, fn = (int(tabla[columna].sum()) for columna in ("tp", "fp", "fn"))
    metricas = calcular_metricas(tp, fp, fn)
    reales = metricas["pixeles_reales"]
    predichos = metricas["pixeles_predichos"]
    positivas = tabla["pixeles_reales"] > 0
    detectadas = tabla["pixeles_predichos"] > 0
    resumen = {
        "imagenes": len(tabla),
        "imagenes_positivas": int(positivas.sum()),
        "imagenes_negativas": int((~positivas).sum()),
        "imagenes_falso_positivo": int((~positivas & detectadas).sum()),
        "imagenes_falso_negativo": int((positivas & ~detectadas).sum()),
        **metricas,
        "superficie_real_m2": reales * AREA_PIXEL,
        "superficie_predicha_m2": predichos * AREA_PIXEL,
        "diferencia_superficie_m2": (predichos - reales) * AREA_PIXEL,
    }
    for categoria in CATEGORIAS_REVISION:
        resumen[f"categoria_{categoria}"] = int((tabla["categoria"] == categoria).sum())
    return resumen


def comparar(tabla: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    """Compara las métricas de la campaña y la referencia por grupo.

    Args:
        tabla: Resultados por tesela que se resumen.
        referencia: Resultados de las mismas localizaciones de 2023.

    Returns:
        pd.DataFrame: Métricas y diferencias respecto a 2023 por grupo de selección.
    """

    registros = []
    for grupo in ("total", *GRUPOS):
        actual = tabla if grupo == "total" else tabla[tabla["grupo"] == grupo]
        anterior = (
            referencia if grupo == "total" else referencia[referencia["grupo"] == grupo]
        )
        if actual.empty:
            continue
        resumen = resumir(actual)
        base = resumir(anterior)
        registro = {"grupo": grupo, **resumen}
        for metrica in METRICAS:
            registro[f"{metrica}_15cm"] = base[metrica]
        for metrica in METRICAS:
            registro[f"delta_{metrica}"] = resumen[metrica] - base[metrica]
        registros.append(registro)
    return pd.DataFrame(registros)


def intervalos_bootstrap(
    tabla, referencia, repeticiones: int, semilla: int
) -> pd.DataFrame:
    """Remuestrea teselas por grupo; las diferencias usan pares de localizaciones.

    Args:
        tabla: Resultados por tesela que se resumen.
        referencia: Resultados de las mismas localizaciones de 2023.
        repeticiones: Número de réplicas de bootstrap.
        semilla: Semilla del muestreo reproducible.

    Returns:
        pd.DataFrame: Intervalos percentiles y número de réplicas válidas por métrica.
    """

    rng = np.random.default_rng(semilla)
    registros = []
    referencia = referencia.set_index("tile_id").loc[tabla["tile_id"]].reset_index()
    for grupo in ("total", *GRUPOS):
        posiciones = [
            np.flatnonzero(tabla["grupo"].to_numpy() == estrato)
            for estrato in (GRUPOS if grupo == "total" else (grupo,))
        ]
        posiciones = [indices for indices in posiciones if len(indices)]
        if not posiciones:
            continue
        valores = {
            metrica: [] for metrica in (*METRICAS, *(f"delta_{m}" for m in METRICAS))
        }
        for _ in range(repeticiones):
            indices = np.concatenate(
                [rng.choice(p, len(p), replace=True) for p in posiciones]
            )
            actual = resumir(tabla.iloc[indices])
            base = resumir(referencia.iloc[indices])
            for metrica in METRICAS:
                valores[metrica].append(actual[metrica])
                valores[f"delta_{metrica}"].append(actual[metrica] - base[metrica])
        for metrica, muestras in valores.items():
            finitas = np.asarray(muestras)[np.isfinite(muestras)]
            inferior, superior = (
                np.quantile(finitas, [0.025, 0.975])
                if len(finitas)
                else (np.nan, np.nan)
            )
            registros.append(
                {
                    "grupo": grupo,
                    "metrica": metrica,
                    "ic95_inferior": inferior,
                    "ic95_superior": superior,
                    "replicas_validas": len(finitas),
                    "repeticiones": repeticiones,
                }
            )
    return pd.DataFrame(registros)


def guardar_ejemplos(
    modelo, dispositivo, carpeta, tabla, salida, cantidad: int
) -> None:
    """Guarda solo los peores ejemplos de cada categoría, sin mapas de probabilidad.

    Args:
        modelo: Modelo cargado para esta fase.
        dispositivo: CPU o GPU utilizada por el modelo.
        carpeta: Carpeta de datos de la campaña.
        tabla: Resultados por tesela que se resumen.
        salida: Carpeta donde se guardan los resultados.
        cantidad: Máximo de ejemplos por categoría.
    """

    if cantidad == 0:
        return
    ejemplos = salida / "ejemplos"
    ejemplos.mkdir()
    for categoria in CATEGORIAS_REVISION:
        errores = (
            tabla[tabla["categoria"] == categoria].sort_values("dice").head(cantidad)
        )
        for fila in errores.to_dict("records"):
            ruta, anotacion = rutas_tesela(carpeta, fila)
            with Image.open(ruta) as original:
                imagen = original.convert("RGB").resize(
                    (TAM_MODELO, TAM_MODELO), Image.Resampling.BILINEAR
                )
            real = crear_mascara_unet(revisar_par(ruta, anotacion)).resize(
                (TAM_MODELO, TAM_MODELO), Image.Resampling.NEAREST
            )
            _, predicha = predecir(modelo, entrada_modelo(imagen), dispositivo)
            crear_visualizacion(imagen, np.asarray(real) > 0, predicha).save(
                ejemplos / f"{categoria}_{fila['tile_id']}.png"
            )


def main() -> None:
    """Evalúa la campaña configurada y publica su comparación con 2023.

    Raises:
        ValueError: La configuración no coincide con la muestra preparada.
        FileExistsError: Queda una salida incompleta de una ejecución anterior.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campania")
    parser.add_argument("--config", help="JSON opcional con otras campañas")
    args = parser.parse_args()
    config = cargar_campania(args.campania, args.config)
    carpeta = carpeta_campania(args.campania)
    guardada = json.loads((carpeta / "config.json").read_text(encoding="utf-8"))
    if any(config[campo] != guardada[campo] for campo in CAMPOS_MUESTRA):
        raise ValueError("La configuración no coincide con la muestra preparada")
    seleccion = pd.read_csv(carpeta / "manifest.csv", keep_default_na=False)
    seleccion, excluidas = revisar_etiquetas(carpeta, seleccion)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = cargar_modelo(PESOS_UNET, dispositivo)
    modelo.requires_grad_(False)
    tabla = evaluar_campania(modelo, dispositivo, carpeta, seleccion)
    print("Evaluando las mismas localizaciones de 2023 a 15 cm...")
    referencia = evaluar_referencia(modelo, dispositivo, seleccion)
    resumen = comparar(tabla, referencia)
    intervalos = (
        intervalos_bootstrap(tabla, referencia, config["bootstrap"], config["semilla"])
        if config["bootstrap"]
        else None
    )
    salida = ruta_proyecto(f"runs/evaluacion/validacion_{args.campania}")
    temporal = salida.with_name(f".{salida.name}_preparando")
    if temporal.exists():
        raise FileExistsError(
            f"Revisa la salida incompleta antes de repetir: {temporal}"
        )
    temporal.mkdir(parents=True)
    try:
        tabla.to_csv(temporal / "resultados.csv", index=False)
        referencia.to_csv(temporal / "referencia_15cm.csv", index=False)
        resumen.to_csv(temporal / "resumen.csv", index=False)
        excluidas.to_csv(temporal / "excluidas.csv", index=False)
        if intervalos is not None:
            intervalos.to_csv(temporal / "intervalos.csv", index=False)
        if (EVALUACION_UNET / "resumen.csv").is_file():
            historico = pd.read_csv(EVALUACION_UNET / "resumen.csv").iloc[0]
            total = resumen.iloc[0]
            pd.DataFrame(
                [
                    {
                        "metrica": m,
                        "test_15cm_historico": historico[m],
                        "campania": total[m],
                        "delta": total[m] - historico[m],
                    }
                    for m in METRICAS
                ]
            ).to_csv(temporal / "comparacion_test_historico.csv", index=False)
        firma = huella_archivo(PESOS_UNET)
        informacion = {
            **config,
            "pesos": str(PESOS_UNET),
            "sha256_pesos": firma,
            "umbral": UMBRAL,
            "tam_modelo": TAM_MODELO,
            "normalizacion": "ImageNet",
            "imagen_resize": "bilinear",
            "mascara_resize": "nearest",
            "area_pixel_modelo_m2": AREA_PIXEL,
            "correccion_superficie": None,
            "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "imagenes_excluidas": len(excluidas),
            "referencia": "mismas localizaciones de test 2023, reevaluadas con los mismos pesos",
            "metodo_bootstrap": (
                "percentil 95%, teselas por grupo y pares para diferencias; no corrige "
                "dependencia espacial"
            ),
            "test_historico": "referencia contextual; sus pesos no constan en el CSV original",
        }
        (temporal / "config.json").write_text(
            json.dumps(informacion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        guardar_ejemplos(
            modelo,
            dispositivo,
            carpeta,
            tabla,
            temporal,
            config["ejemplos_por_categoria"],
        )
        reemplazar_carpeta(temporal, salida)
    finally:
        if temporal.exists():
            shutil.rmtree(temporal)
    print(
        resumen[["grupo", "imagenes", *METRICAS, "delta_dice"]].to_string(index=False)
    )
    print(f"Resultados: {salida}")


if __name__ == "__main__":
    main()
