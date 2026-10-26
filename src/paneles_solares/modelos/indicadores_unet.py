"""Conserva detecciones territoriales U-Net y describe su contexto de 2023."""

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
from shapely import union_all
from shapely.geometry import box, shape
from torch import nn

from paneles_solares.geografia.catastro import relacionar
from paneles_solares.modelos.unet import (
    TAM_ENTRADA,
    UMBRAL,
    cargar_modelo,
    normalizar_rgb,
)
from paneles_solares.rutas import (
    EDIFICIOS_CATASTRO,
    INDICADORES_UNET,
    INDICE_TESELAS,
    MUNICIPIOS_CANTABRIA,
    PESOS_UNET,
    PNOA,
    huella_archivo,
    reemplazar_carpeta,
    ruta_proyecto,
)

INDICE = INDICE_TESELAS
EDIFICIOS = EDIFICIOS_CATASTRO
MUNICIPIOS = MUNICIPIOS_CANTABRIA
SALIDA = INDICADORES_UNET

LOTE = 4
RESOLUCION_PNOA_M = 0.15
FACTOR_CALIBRACION_TEST = 0.9246  # Sesgo global de superficie del test: -7,54 %.


def cargar_datos() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Lee el índice y las geometrías necesarias antes de la inferencia.

    Returns:
        tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]: Tablas de entrada de
            esta fase, con sus geometrías e indicadores.

    Raises:
        FileNotFoundError: Falta un archivo de entrada o de pesos necesario para esta
            fase.
        ValueError: El índice o las geometrías están vacíos, repetidos o tienen un CRS
            incompatible.
    """

    for ruta in (PESOS_UNET, INDICE, EDIFICIOS, MUNICIPIOS):
        if not ruta.is_file():
            raise FileNotFoundError(ruta)

    indice = pd.read_csv(INDICE)
    edificios = gpd.read_file(EDIFICIOS, layer="edificios")
    municipios = gpd.read_file(MUNICIPIOS, layer="municipios")[["codigo", "municipio", "tipo_unidad", "geometry"]]

    if edificios.crs is None or municipios.crs is None:
        raise ValueError("Los edificios y municipios necesitan un CRS definido")
    if edificios.crs.to_epsg() != 25830 or municipios.crs != edificios.crs:
        raise ValueError(
            "Los edificios y municipios deben estar en EPSG:25830 (metros)"
        )
    if indice.empty or municipios.empty:
        raise ValueError(
            "El índice y los municipios no pueden estar vacíos"
        )
    if indice.tile_id.duplicated().any():
        raise ValueError("El índice contiene teselas repetidas")

    if "id_edificio" not in edificios or edificios.anio_fuente.ne(2023).any():
        raise ValueError("Prepara primero la referencia oficial de 2023")
    edificios = edificios.reset_index(drop=True)
    municipios["codigo"] = municipios["codigo"].astype(str)
    return indice, edificios, municipios


def comprobar_ortofotos(indice: pd.DataFrame, crs: object) -> dict[str, object]:
    """Valida las ortofotos y conserva sus extensiones para resolver solapes.

    Args:
        indice: Posición de la imagen dentro del dataset.
        crs: Sistema de coordenadas esperado para las ortofotos.

    Returns:
        dict[str, object]: Extensión métrica de cada ortofoto comprobada.

    Raises:
        ValueError: Hay un CRS, RGB, resolución o ventana incompatible con el índice.
        FileNotFoundError: Falta un archivo de entrada o de pesos necesario para esta
            fase.
    """

    extensiones = {}
    for nombre, filas in indice.groupby("tif", sort=True):
        ruta = PNOA / nombre
        if not ruta.is_file():
            raise FileNotFoundError(ruta)
        with rasterio.open(ruta) as tif:
            if (
                tif.crs != crs
                or tif.count < 3
                or any(tipo != "uint8" for tipo in tif.dtypes[:3])
            ):
                raise ValueError(f"CRS o canales RGB incompatibles: {ruta}")
            if (
                tif.transform.a <= 0
                or tif.transform.e >= 0
                or tif.transform.b != 0
                or tif.transform.d != 0
                or not np.allclose(tif.res, (RESOLUCION_PNOA_M, RESOLUCION_PNOA_M))
            ):
                raise ValueError(
                    f"Se espera una ortofoto norte-arriba de 0,15 m: {ruta}"
                )
            if (
                (filas["fila"] < 0).any()
                or (filas["columna"] < 0).any()
                or (filas["fila"] + filas["alto"] > tif.height).any()
                or (filas["columna"] + filas["ancho"] > tif.width).any()
                or (filas[["ancho", "alto"]] <= 0).any().any()
                or (filas[["ancho", "alto"]] > TAM_ENTRADA).any().any()
            ):
                raise ValueError(f"Hay ventanas fuera de la ortofoto: {ruta}")
            extensiones[nombre] = box(*tif.bounds)
    return extensiones


def asociar_municipios(
    edificios: gpd.GeoDataFrame, municipios: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Asigna cada edificio al municipio de un punto interior.

    Args:
        edificios: Geometrías e indicadores de edificios.
        municipios: Indicadores y geometrías municipales.

    Returns:
        gpd.GeoDataFrame: Edificios con código municipal y superficie de cubierta.

    Raises:
        ValueError: Un edificio tiene más de una asignación municipal igualmente válida.
    """

    puntos = gpd.GeoDataFrame(
        geometry=edificios.representative_point(), crs=edificios.crs
    )
    asociados = gpd.sjoin(puntos, municipios, how="left", predicate="within")
    if asociados.index.duplicated().any():
        raise ValueError("Los límites municipales se solapan en edificios")
    sin_municipio = asociados.codigo.isna()
    if sin_municipio.any():
        cercanos = gpd.sjoin_nearest(puntos.loc[sin_municipio], municipios, how="left")
        if cercanos.index.duplicated().any():
            raise ValueError("Hay edificios con varios municipios igualmente cercanos")
        asociados.loc[cercanos.index, ["codigo", "municipio"]] = cercanos[
            ["codigo", "municipio"]
        ]
        print(
            f"Edificios de borde asignados al municipio más cercano: {sin_municipio.sum()}"
        )

    asociados = asociados.reindex(edificios.index)
    edificios = edificios.copy()
    edificios["codigo_municipal"] = asociados.codigo.to_numpy()
    edificios["municipio"] = asociados.municipio.to_numpy()
    edificios["superficie_cubierta_m2"] = edificios.geometry.area
    return edificios


def predecir_lote(
    modelo: nn.Module, dispositivo: torch.device, imagenes: list[np.ndarray]
) -> list[np.ndarray]:
    """Aplica la normalización y el umbral de evaluar_unet.py.

    Args:
        modelo: Modelo cargado para esta fase.
        dispositivo: CPU o GPU utilizada por el modelo.
        imagenes: Carpeta de imágenes del test.

    Returns:
        list[np.ndarray]: Máscaras binarias recortadas al tamaño original de cada
            imagen.
    """

    tamanos = [imagen.shape[:2] for imagen in imagenes]
    preparadas = np.zeros((len(imagenes), TAM_ENTRADA, TAM_ENTRADA, 3), dtype=np.float32)
    for indice, (imagen, (alto, ancho)) in enumerate(zip(imagenes, tamanos)):
        preparadas[indice, :alto, :ancho] = imagen.astype(np.float32)
    preparadas = normalizar_rgb(preparadas)
    tensor = torch.from_numpy(np.ascontiguousarray(preparadas.transpose(0, 3, 1, 2)))
    with torch.inference_mode():
        probabilidades = (
            torch.sigmoid(modelo(tensor.to(dispositivo)))[:, 0].cpu().numpy()
        )
    return [
        probabilidades[i, :alto, :ancho] >= UMBRAL
        for i, (alto, ancho) in enumerate(tamanos)
    ]


def acumular_ventana(
    mascara: np.ndarray,
    transformacion: object,
    zona: object,
    edificios: gpd.GeoDataFrame,
    pixeles: np.ndarray,
    superficie_cubierta: np.ndarray,
) -> None:
    """Cuenta cada píxel en un único edificio y registra el área cubierta.

    Args:
        mascara: Máscara binaria de paneles.
        transformacion: Transformación afín de píxeles a coordenadas métricas.
        zona: Parte de la ventana que todavía no se ha contado.
        edificios: Geometrías e indicadores de edificios.
        pixeles: Recuento acumulado de píxeles predichos por edificio.
        superficie_cubierta: Superficie de cubierta cubierta por las ventanas, en m².
    """

    if zona.is_empty:
        return
    posiciones = edificios.sindex.query(zona, predicate="intersects")
    if not len(posiciones):
        return

    geometrias = edificios.geometry.iloc[posiciones]
    superficie_cubierta[posiciones] += np.array(
        [geometria.intersection(zona).area for geometria in geometrias]
    )
    etiquetas = rasterize(
        ((geometria, numero) for numero, geometria in enumerate(geometrias, start=1)),
        out_shape=mascara.shape,
        transform=transformacion,
        dtype="int32",
    )
    ventana = box(*rasterio.transform.array_bounds(*mascara.shape, transformacion))
    if not zona.equals(ventana):
        dentro = rasterize(
            [(zona, 1)],
            out_shape=mascara.shape,
            transform=transformacion,
            dtype="uint8",
        )
        mascara = mascara & dentro.astype(bool)
    cantidades = np.bincount(etiquetas[mascara], minlength=len(posiciones) + 1)[1:]
    np.add.at(pixeles, posiciones, cantidades)


def limitar_a_cobertura(tif, ventana, imagen, mascara, zona):
    """Excluye píxeles sin imagen válida y zonas ya cubiertas por otra hoja."""
    transformacion = tif.window_transform(ventana)
    valida = (tif.read_masks(1, window=ventana) > 0) & np.any(imagen != 0, axis=2)
    if not valida.all():
        geometrias = [
            shape(geometria)
            for geometria, _ in shapes(
                valida.astype("uint8"), mask=valida, transform=transformacion
            )
        ]
        zona = zona.intersection(union_all(geometrias))
    if zona.is_empty:
        return mascara & False, zona

    dentro = rasterize(
        [(zona, 1)], out_shape=mascara.shape, transform=transformacion, dtype="uint8"
    )
    return mascara & valida & dentro.astype(bool), zona


def vectorizar_mascara(mascara, transformacion, zona, registro, borde):
    """Conserva cada componente completo y su tesela, sin recorte por edificios."""
    componentes = []
    for geometria, _ in shapes(
        mascara.astype("uint8"), mask=mascara, transform=transformacion
    ):
        poligono = shape(geometria).intersection(zona)
        if poligono.is_empty or poligono.area == 0:
            continue
        identidad = hashlib.sha256(str(registro.tile_id).encode() + poligono.wkb)
        componentes.append({
            "id_deteccion": "fv_" + identidad.hexdigest()[:24],
            "tile_id": registro.tile_id,
            "tif": registro.tif,
            "anio": 2023,
            "area_m2": poligono.area,
            "tesela_con_edificios": bool(registro.num_edificios > 0),
            "borde_tesela": poligono.intersects(borde),
            "geometry": poligono,
        })
    return componentes


def procesar_teselas(
    indice: pd.DataFrame,
    edificios: gpd.GeoDataFrame,
    extensiones: dict[str, object],
    modelo: nn.Module,
    dispositivo: torch.device,
    territorio=None,
    avance: Path | None = None,
    referencia: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame, object]:
    """Infiere por lotes y permite reanudar desde la última hoja guardada.

    Args:
        indice: Posición de la imagen dentro del dataset.
        edificios: Geometrías e indicadores de edificios.
        extensiones: Extensión métrica de cada ortofoto.
        modelo: Modelo cargado para esta fase.
        dispositivo: CPU o GPU utilizada por el modelo.

    Returns:
        tuple: Píxeles y cobertura por unidad edificada, componentes completos
            y geometría del territorio realmente procesado.
    """

    pixeles = np.zeros(len(edificios), dtype=np.int64)
    superficie_cubierta = np.zeros(len(edificios), dtype=np.float64)
    procesadas = 0
    detecciones = []
    coberturas = []
    completadas = []
    segundos_previos = 0.0
    if avance is not None and avance.exists():
        with avance.open("rb") as archivo:
            estado = pickle.load(archivo)
        if estado["referencia"] != referencia:
            raise ValueError(f"El avance corresponde a otras entradas o código: {avance}")
        pixeles = estado["pixeles"]
        superficie_cubierta = estado["superficie_cubierta"]
        detecciones = estado["detecciones"]
        coberturas = estado["coberturas"]
        completadas = estado["completadas"]
        procesadas = estado["procesadas"]
        segundos_previos = estado["segundos"]
        print(f"Reanudando: {len(completadas)} hojas y {procesadas} teselas guardadas", flush=True)
    inicio_ejecucion = time.perf_counter()

    for nombre, filas in indice.groupby("tif", sort=True):
        if nombre in completadas:
            continue
        zona_hoja = (
            extensiones[nombre].difference(union_all(coberturas))
            if coberturas
            else extensiones[nombre]
        )
        if territorio is not None:
            zona_hoja = zona_hoja.intersection(territorio)
        zonas_procesadas = []
        with rasterio.open(PNOA / nombre) as tif:
            for inicio in range(0, len(filas), LOTE):
                lote = filas.iloc[inicio : inicio + LOTE]
                ventanas = [
                    Window(fila.columna, fila.fila, fila.ancho, fila.alto)
                    for fila in lote.itertuples()
                ]
                zonas = [
                    zona_hoja.intersection(box(*tif.window_bounds(v)))
                    for v in ventanas
                ]
                activas = [i for i, zona in enumerate(zonas) if not zona.is_empty]
                if activas:
                    imagenes = [
                        np.moveaxis(tif.read([1, 2, 3], window=ventanas[i]), 0, -1)
                        for i in activas
                    ]
                    mascaras = predecir_lote(modelo, dispositivo, imagenes)
                    for i, mascara, imagen in zip(activas, mascaras, imagenes):
                        transformacion = tif.window_transform(ventanas[i])
                        mascara, zona = limitar_a_cobertura(
                            tif, ventanas[i], imagen, mascara, zonas[i]
                        )
                        if zona.is_empty:
                            continue
                        zonas_procesadas.append(zona)
                        borde = box(*tif.window_bounds(ventanas[i])).boundary
                        detecciones.extend(vectorizar_mascara(
                            mascara, transformacion, zona, lote.iloc[i], borde
                        ))
                        acumular_ventana(
                            mascara,
                            transformacion,
                            zona,
                            edificios,
                            pixeles,
                            superficie_cubierta,
                        )
                procesadas += len(lote)
                if procesadas % 1000 < len(lote):
                    segundos = segundos_previos + time.perf_counter() - inicio_ejecucion
                    horas = (len(indice) - procesadas) * segundos / procesadas / 3600
                    print(
                        f"Teselas: {procesadas}/{len(indice)}; "
                        f"restante estimado: {horas:.1f} h; hoja: {nombre}", flush=True
                    )
        # Solo lo realmente procesado evita que un lote parcial tape otra hoja.
        cobertura_hoja = union_all(zonas_procesadas)
        coberturas.append(cobertura_hoja)
        completadas.append(nombre)
        if avance is not None:
            avance.parent.mkdir(parents=True, exist_ok=True)
            estado = {
                "referencia": referencia,
                "pixeles": pixeles,
                "superficie_cubierta": superficie_cubierta,
                "detecciones": detecciones,
                "coberturas": coberturas,
                "completadas": completadas,
                "procesadas": procesadas,
                "segundos": segundos_previos + time.perf_counter() - inicio_ejecucion,
            }
            temporal = avance.with_suffix(".part")
            with temporal.open("wb") as archivo:
                pickle.dump(estado, archivo, protocol=pickle.HIGHEST_PROTOCOL)
            temporal.replace(avance)
            resumen = {campo: estado[campo] for campo in ("referencia", "completadas", "procesadas", "segundos")}
            avance.with_suffix(".json").write_text(
                json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Avance guardado: {nombre}; componentes acumulados: {len(detecciones)}", flush=True)
    columnas = [
        "id_deteccion", "tile_id", "tif", "anio", "area_m2",
        "tesela_con_edificios", "borde_tesela", "geometry",
    ]
    detecciones = gpd.GeoDataFrame(
        detecciones, columns=columnas, geometry="geometry", crs=edificios.crs
    )
    return pixeles, superficie_cubierta, detecciones, union_all(coberturas)


def calcular_edificios(
    edificios: gpd.GeoDataFrame, pixeles: np.ndarray, superficie_cubierta: np.ndarray
) -> gpd.GeoDataFrame:
    """Conserva unidades observadas, incluida cobertura parcial, y sus áreas.

    Args:
        edificios: Geometrías e indicadores de edificios.
        pixeles: Recuento acumulado de píxeles predichos por edificio.
        superficie_cubierta: Superficie de cubierta cubierta por las ventanas, en m².

    Returns:
        gpd.GeoDataFrame: Unidades observadas con cobertura explícita e indicadores.
    """

    completa = np.isclose(
        superficie_cubierta,
        edificios.superficie_cubierta_m2.to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    ) & (superficie_cubierta > 0)
    print(f"Unidades sin cobertura completa (se conserva la parte observada): {(~completa).sum()}")
    observados = superficie_cubierta > 0
    resultado = edificios.loc[observados].copy()
    resultado["cobertura_completa"] = completa[observados]
    resultado["superficie_observada_m2"] = superficie_cubierta[observados]
    resultado["fraccion_observada"] = np.minimum(
        superficie_cubierta[observados] / resultado.superficie_cubierta_m2, 1)
    resultado["superficie_fv_detectada_m2"] = pixeles[observados] * RESOLUCION_PNOA_M**2
    resultado["superficie_fv_corregida_test_m2"] = (
        resultado.superficie_fv_detectada_m2 / FACTOR_CALIBRACION_TEST
    )
    resultado["porcentaje_cubierta_ocupado"] = (
        100 * resultado.superficie_fv_detectada_m2 / resultado.superficie_cubierta_m2
    )
    resultado["presencia_fotovoltaica"] = pixeles[observados] > 0
    return resultado.drop(columns="index_right", errors="ignore")


def resumir_municipios(
    edificios: gpd.GeoDataFrame, municipios: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Agrega los indicadores por municipio, incluidos los que tienen cero edificios.

    Args:
        edificios: Geometrías e indicadores de edificios.
        municipios: Indicadores y geometrías municipales.

    Returns:
        gpd.GeoDataFrame: Geometrías municipales con recuentos, superficies y
            proporciones.
    """

    agrupados = edificios.groupby("codigo_municipal").agg(
        edificios_analizados=("id_edificio", "size"),
        edificios_con_deteccion=("presencia_fotovoltaica", "sum"),
        superficie_fv_detectada_total_m2=("superficie_fv_detectada_m2", "sum"),
        superficie_fv_corregida_test_total_m2=(
            "superficie_fv_corregida_test_m2",
            "sum",
        ),
        superficie_cubiertas_analizadas_m2=("superficie_observada_m2", "sum"),
    )
    positivos = edificios.loc[edificios.presencia_fotovoltaica]
    estadisticas = positivos.groupby("codigo_municipal").superficie_fv_detectada_m2.agg(
        superficie_fv_media_positivo_m2="mean",
        superficie_fv_mediana_positivo_m2="median",
    )
    resumen = municipios.merge(
        agrupados.join(estadisticas), left_on="codigo", right_index=True, how="left"
    )
    cantidades = ["edificios_analizados", "edificios_con_deteccion"]
    superficies = [
        "superficie_fv_detectada_total_m2",
        "superficie_fv_corregida_test_total_m2",
        "superficie_cubiertas_analizadas_m2",
    ]
    resumen[cantidades] = resumen[cantidades].fillna(0).astype(int)
    resumen[superficies] = resumen[superficies].fillna(0)
    base = resumen.edificios_analizados.replace(0, np.nan)
    resumen["porcentaje_edificios_con_deteccion"] = (
        100 * resumen.edificios_con_deteccion / base
    )
    resumen["edificios_con_deteccion_por_1000"] = (
        1000 * resumen.edificios_con_deteccion / base
    )
    techo = resumen.superficie_cubiertas_analizadas_m2.replace(0, np.nan)
    resumen["porcentaje_cubierta_ocupado"] = (
        100 * resumen.superficie_fv_detectada_total_m2 / techo
    )
    return resumen


def guardar_resultados(
    edificios: gpd.GeoDataFrame, municipios: gpd.GeoDataFrame
) -> None:
    """Sobrescribe las tres salidas fijas de esta fase.

    Args:
        edificios: Geometrías e indicadores de edificios.
        municipios: Indicadores y geometrías municipales.
    """

    SALIDA.mkdir(parents=True, exist_ok=True)
    rutas = [
        SALIDA / "edificios.gpkg",
        SALIDA / "municipios.csv",
        SALIDA / "municipios.gpkg",
    ]
    for ruta in rutas:
        ruta.unlink(missing_ok=True)
    edificios.to_file(rutas[0], layer="edificios", driver="GPKG", index=False)
    municipios.drop(columns="geometry").to_csv(rutas[1], index=False)
    municipios.to_file(rutas[2], layer="municipios", driver="GPKG", index=False)
    print(f"Indicadores: {SALIDA}")


def comparar_cobertura(detecciones, edificios):
    """Mide el efecto del filtro de edificios sobre las mismas predicciones de 2023."""
    interiores = []
    for geometria in detecciones.geometry:
        posiciones = edificios.sindex.query(geometria, predicate="intersects")
        huellas = union_all(edificios.geometry.iloc[posiciones])
        interiores.append(geometria.intersection(huellas).area)
    detecciones["area_en_edificios_m2"] = pd.Series(
        interiores, index=detecciones.index, dtype=float
    ).clip(lower=0, upper=detecciones.area_m2)
    detecciones["area_fuera_edificios_m2"] = (
        detecciones.area_m2 - detecciones.area_en_edificios_m2
    ).clip(lower=0)
    zonas = {
        "dentro_edificaciones": detecciones.area_en_edificios_m2,
        "fuera_edificaciones_en_teselas_con_edificios": detecciones.area_fuera_edificios_m2.where(
            detecciones.tesela_con_edificios.astype(bool), 0
        ),
        "teselas_sin_edificios": detecciones.area_fuera_edificios_m2.where(
            ~detecciones.tesela_con_edificios.astype(bool), 0
        ),
    }
    return pd.DataFrame([
        {"zona": zona, "componentes_con_area": int((areas > 0).sum()), "area_m2": areas.sum()}
        for zona, areas in zonas.items()
    ])


def guardar_detecciones(detecciones, edificios, municipios, cobertura, contexto=None):
    """Publica máscaras vectoriales, contexto y superficies sin filtro catastral."""
    parcelas = gpd.read_file(EDIFICIOS, layer="parcelas", bbox=cobertura.bounds)
    if contexto is None:
        contexto = gpd.read_file(EDIFICIOS, layer="edificios")
    comparar_cobertura(detecciones, contexto).to_csv(
        SALIDA / "comparacion_cobertura.csv", index=False
    )
    for entidades, campo, nombre in [
        (contexto, "id_edificio", "edificios"),
        (parcelas, "id_parcela", "parcelas"),
    ]:
        relaciones = relacionar(detecciones, entidades, "id_deteccion", campo)
        if campo == "id_edificio":
            relaciones["id_cubierta"] = "catastro_" + relaciones.id_edificio + "_parte_1"
            relaciones = relaciones.merge(
                contexto[["id_edificio", "uso_del_suelo", "subtipo"]],
                on="id_edificio",
                how="left",
            )
        relaciones.to_csv(SALIDA / f"detecciones_{nombre}.csv", index=False)
    if len(detecciones) and len(contexto):
        # La distancia describe; un edificio cercano no se asigna como soporte.
        _, distancias = contexto.sindex.nearest(
            detecciones.geometry, return_distance=True, return_all=False
        )
        detecciones["distancia_edificio_m"] = distancias
    else:
        detecciones["distancia_edificio_m"] = np.nan
    partes = []
    for municipio in municipios.itertuples():
        posiciones = detecciones.sindex.query(municipio.geometry, predicate="intersects")
        for posicion in posiciones:
            fila = detecciones.iloc[posicion]
            geometria = fila.geometry.intersection(municipio.geometry)
            if geometria.area > 0:
                punto = geometria.representative_point()
                partes.append({
                    "id_deteccion": fila.id_deteccion,
                    "codigo": municipio.codigo,
                    "area_m2": geometria.area,
                    "x": punto.x,
                    "y": punto.y,
                })
    partes = pd.DataFrame(partes, columns=["id_deteccion", "codigo", "area_m2", "x", "y"])
    partes.to_csv(SALIDA / "detecciones_municipios.csv", index=False)
    superficies = partes.groupby("codigo").area_m2.sum()
    municipios["superficie_fv_en_cubiertas_m2"] = municipios.superficie_fv_detectada_total_m2
    municipios["superficie_fv_detectada_total_m2"] = municipios.codigo.map(superficies).fillna(0)
    # Se conserva la calibración anterior solo para los indicadores de cubierta.
    # La superficie exterior no dispone de una calibración independiente.
    municipios["superficie_fv_corregida_test_total_m2"] = np.nan
    municipios["superficie_territorio_procesado_m2"] = municipios.geometry.intersection(cobertura).area
    municipios["fraccion_territorio_procesado"] = (
        municipios.superficie_territorio_procesado_m2 / municipios.geometry.area
    )
    detecciones.to_file(SALIDA / "detecciones.gpkg", layer="detecciones", index=False)
    gpd.GeoDataFrame({"geometry": [cobertura]}, crs=edificios.crs).to_file(
        SALIDA / "detecciones.gpkg", layer="cobertura_procesada", index=False
    )


def main() -> None:
    """Aplica la U-Net actual a las teselas de Cantabria y resume sus edificios."""

    global SALIDA, INDICE, EDIFICIOS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=float, nargs=4, help="Zona EPSG:25830 para una prueba")
    parser.add_argument("--indice", type=Path, default=INDICE)
    parser.add_argument("--catastro", type=Path, default=EDIFICIOS)
    parser.add_argument("--limite", type=int, help="Número máximo de teselas para medir tiempos")
    args = parser.parse_args()
    INDICE, EDIFICIOS = args.indice, args.catastro
    if args.bbox or args.limite:
        SALIDA = ruta_proyecto("runs/prueba_2023/indicadores")
    destino = SALIDA
    indice, edificios, municipios = cargar_datos()
    territorio = union_all(municipios.geometry)
    if args.bbox:
        territorio = territorio.intersection(box(*args.bbox))
        seleccion = []
        for nombre, filas in indice.groupby("tif", sort=True):
            with rasterio.open(PNOA / nombre) as tif:
                ventana = tif.window(*args.bbox)
                activas = (
                    (filas.columna < ventana.col_off + ventana.width)
                    & (filas.columna + filas.ancho > ventana.col_off)
                    & (filas.fila < ventana.row_off + ventana.height)
                    & (filas.fila + filas.alto > ventana.row_off)
                )
                seleccion.extend(filas.index[activas])
        indice = indice.loc[seleccion]
    if indice.empty:
        parser.error("No hay teselas disponibles para la zona solicitada")
    if args.limite is not None:
        if args.limite < 1:
            parser.error("El límite debe ser positivo")
        indice = indice.head(args.limite)
    extensiones = comprobar_ortofotos(indice, edificios.crs)
    edificios = asociar_municipios(edificios, municipios)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = cargar_modelo(PESOS_UNET, dispositivo)
    print(
        f"Dispositivo: {dispositivo}; teselas: {len(indice)}; edificios: {len(edificios)}"
    )
    inicio = time.perf_counter()
    referencia = {
        "modelo_sha256": huella_archivo(PESOS_UNET),
        "catastro_sha256": huella_archivo(EDIFICIOS),
        "municipios_sha256": huella_archivo(MUNICIPIOS),
        "indice_sha256": huella_archivo(INDICE),
        "codigo_sha256": huella_archivo(Path(__file__)),
        "umbral": UMBRAL,
    }
    avance = None
    if not (args.bbox or args.limite):
        avance = destino.with_name(f".{destino.name}_avance") / "estado.pkl"
    pixeles, superficie_cubierta, detecciones, cobertura = procesar_teselas(
        indice, edificios, extensiones, modelo, dispositivo, territorio, avance, referencia
    )
    fin_inferencia = time.perf_counter()
    segundos_inferencia = fin_inferencia - inicio
    if avance is not None:
        segundos_inferencia = json.loads(avance.with_suffix(".json").read_text(encoding="utf-8"))["segundos"]
    resultado_edificios = calcular_edificios(edificios, pixeles, superficie_cubierta)
    resultado_municipios = resumir_municipios(resultado_edificios, municipios)
    SALIDA = destino.with_name(f".{destino.name}_preparando")
    SALIDA.mkdir(parents=True, exist_ok=True)
    guardar_detecciones(
        detecciones, resultado_edificios, resultado_municipios, cobertura, edificios
    )
    guardar_resultados(resultado_edificios, resultado_municipios)
    metadata = {
        "anio": 2023,
        "alcance": "prueba" if args.bbox or args.limite else "regional",
        "bbox": args.bbox,
        "teselas": len(indice),
        "componentes": len(detecciones),
        **referencia,
        "area_m2": float(detecciones.area_m2.sum()),
        "segundos_inferencia_y_contexto": segundos_inferencia + time.perf_counter() - fin_inferencia,
        "nota": "Componentes completos por tesela, sin agrupar en instalaciones. "
        "La calibración territorial del sesgo está pendiente.",
        "atribucion": "Realizado a partir de datos del Gobierno de Cantabria disponibles "
        "de forma gratuita en https://mapas.cantabria.es; Catastro e IGN/CNIG (PNOA).",
    }
    (SALIDA / "ejecucion.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    reemplazar_carpeta(SALIDA, destino)
    if avance is not None:
        avance.unlink(missing_ok=True)
    print(f"Ejecución terminada: {destino}", flush=True)


if __name__ == "__main__":
    main()
