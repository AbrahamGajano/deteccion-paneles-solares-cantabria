"""Calcula indicadores fotovoltaicos por edificio y municipio con U-Net."""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely import union_all
from shapely.geometry import box
from torch import nn

from paneles_solares.geografia.teselas import TAM_TES
from paneles_solares.modelos.entrenar_unet import DESVIACION_IMAGENET, MEDIA_IMAGENET
from paneles_solares.modelos.evaluar_unet import UMBRAL, cargar_modelo
from paneles_solares.rutas import ruta_proyecto

PESOS = ruta_proyecto("weights/unet_resnet34.pt")
INDICE = ruta_proyecto("data/manifests/teselas.csv")
PNOA = ruta_proyecto("data/pnoa")
EDIFICIOS = ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg")
MUNICIPIOS = ruta_proyecto("data/geografia/municipios_cantabria.gpkg")
SALIDA = ruta_proyecto("runs/indicadores/unet_actual")

LOTE = 4
RESOLUCION_PNOA_M = 0.15
FACTOR_CALIBRACION_TEST = 0.9246  # Sesgo global de superficie del test: -7,54 %.


def cargar_datos() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Lee el índice y las geometrías necesarias antes de la inferencia.

    Returns:
        tuple: Teselas, edificios únicos y límites municipales.
    """
    for ruta in (PESOS, INDICE, EDIFICIOS, MUNICIPIOS):
        if not ruta.is_file():
            raise FileNotFoundError(ruta)

    indice = pd.read_csv(INDICE)
    edificios = gpd.read_file(EDIFICIOS, fid_as_index=True)
    municipios = gpd.read_file(MUNICIPIOS)[["codigo", "municipio", "geometry"]]

    if edificios.crs is None or municipios.crs is None:
        raise ValueError("Los edificios y municipios necesitan un CRS definido")
    if edificios.crs.to_epsg() != 25830 or municipios.crs != edificios.crs:
        raise ValueError(
            "Los edificios y municipios deben estar en EPSG:25830 (metros)"
        )
    if indice.empty or edificios.empty or municipios.empty:
        raise ValueError(
            "El índice, los edificios y los municipios no pueden estar vacíos"
        )
    if indice.tile_id.duplicated().any():
        raise ValueError("El índice contiene teselas repetidas")

    # El GeoPackage conserva un fid, pero no una referencia catastral oficial.
    edificios["id_edificio"] = edificios.index.astype(int)
    duplicados = edificios.geometry.to_wkb().duplicated()
    if duplicados.any():
        print(f"Geometrías catastrales repetidas descartadas: {duplicados.sum()}")
        edificios = edificios.loc[~duplicados]
    edificios = edificios.reset_index(drop=True)
    municipios["codigo"] = municipios["codigo"].astype(str)
    return indice, edificios, municipios


def comprobar_ortofotos(indice: pd.DataFrame, crs: object) -> dict[str, object]:
    """Valida las ortofotos y conserva sus extensiones para resolver solapes.

    Args:
        indice (pd.DataFrame): Teselas que se van a procesar.
        crs: Sistema de coordenadas de los edificios.

    Returns:
        dict[str, object]: Extensión de cada ortofoto indexada.
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
                or (filas[["ancho", "alto"]] > TAM_TES).any().any()
            ):
                raise ValueError(f"Hay ventanas fuera de la ortofoto: {ruta}")
            extensiones[nombre] = box(*tif.bounds)
    return extensiones


def asociar_municipios(
    edificios: gpd.GeoDataFrame, municipios: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Asigna cada edificio al municipio de un punto interior.

    Args:
        edificios (gpd.GeoDataFrame): Geometrías catastrales únicas.
        municipios (gpd.GeoDataFrame): Límites oficiales municipales.

    Returns:
        gpd.GeoDataFrame: Edificios con código y nombre municipal.
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
        modelo (nn.Module): U-Net con los pesos del test.
        dispositivo (torch.device): CPU o GPU utilizada.
        imagenes (list[np.ndarray]): Ventanas RGB de hasta 512 píxeles por lado.

    Returns:
        list[np.ndarray]: Máscaras binarias en el tamaño original de cada ventana.
    """
    tamanos = [imagen.shape[:2] for imagen in imagenes]
    preparadas = np.zeros((len(imagenes), TAM_TES, TAM_TES, 3), dtype=np.float32)
    for indice, (imagen, (alto, ancho)) in enumerate(zip(imagenes, tamanos)):
        preparadas[indice, :alto, :ancho] = imagen.astype(np.float32)
    preparadas = (preparadas / 255.0 - MEDIA_IMAGENET) / DESVIACION_IMAGENET
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
        mascara (np.ndarray): Segmentación binaria de una ventana.
        transformacion: Transformación afín de esa ventana.
        zona: Parte de la ventana no cubierta por ortofotos anteriores.
        edificios (gpd.GeoDataFrame): Edificios de Catastro.
        pixeles (np.ndarray): Conteos acumulados por posición de edificio.
        superficie_cubierta (np.ndarray): Superficie observada acumulada por edificio.
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


def procesar_teselas(
    indice: pd.DataFrame,
    edificios: gpd.GeoDataFrame,
    extensiones: dict[str, object],
    modelo: nn.Module,
    dispositivo: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Infiere por lotes sin conservar las máscaras de Cantabria en memoria.

    Args:
        indice (pd.DataFrame): Teselas PNOA indexadas.
        edificios (gpd.GeoDataFrame): Edificios que recibirán indicadores.
        extensiones (dict[str, object]): Extensiones de las ortofotos.
        modelo (nn.Module): U-Net preparada para inferencia.
        dispositivo (torch.device): CPU o GPU utilizada.

    Returns:
        tuple[np.ndarray, np.ndarray]: Píxeles positivos y superficie observada.
    """
    pixeles = np.zeros(len(edificios), dtype=np.int64)
    superficie_cubierta = np.zeros(len(edificios), dtype=np.float64)
    anteriores = []
    procesadas = 0

    for nombre, filas in indice.groupby("tif", sort=True):
        territorio = (
            extensiones[nombre].difference(union_all(anteriores))
            if anteriores
            else extensiones[nombre]
        )
        with rasterio.open(PNOA / nombre) as tif:
            for inicio in range(0, len(filas), LOTE):
                lote = filas.iloc[inicio : inicio + LOTE]
                ventanas = [
                    Window(fila.columna, fila.fila, fila.ancho, fila.alto)
                    for fila in lote.itertuples()
                ]
                zonas = [
                    territorio.intersection(box(*tif.window_bounds(v)))
                    for v in ventanas
                ]
                activas = [i for i, zona in enumerate(zonas) if not zona.is_empty]
                if activas:
                    imagenes = [
                        np.moveaxis(tif.read([1, 2, 3], window=ventanas[i]), 0, -1)
                        for i in activas
                    ]
                    mascaras = predecir_lote(modelo, dispositivo, imagenes)
                    for i, mascara in zip(activas, mascaras):
                        acumular_ventana(
                            mascara,
                            tif.window_transform(ventanas[i]),
                            zonas[i],
                            edificios,
                            pixeles,
                            superficie_cubierta,
                        )
                procesadas += len(lote)
                if procesadas % 1000 < len(lote):
                    print(f"Teselas procesadas: {procesadas}/{len(indice)}", flush=True)
        anteriores.append(extensiones[nombre])
    return pixeles, superficie_cubierta


def calcular_edificios(
    edificios: gpd.GeoDataFrame, pixeles: np.ndarray, superficie_cubierta: np.ndarray
) -> gpd.GeoDataFrame:
    """Conserva los edificios cubiertos y calcula sus superficies planimétricas.

    Args:
        edificios (gpd.GeoDataFrame): Edificios con su municipio.
        pixeles (np.ndarray): Píxeles positivos exclusivos por edificio.
        superficie_cubierta (np.ndarray): Área cubierta por ventanas inferidas.

    Returns:
        gpd.GeoDataFrame: Edificios completamente analizados e indicadores.
    """
    completa = np.isclose(
        superficie_cubierta,
        edificios.superficie_cubierta_m2.to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    ) & (superficie_cubierta > 0)
    print(f"Edificios sin cobertura completa, excluidos: {(~completa).sum()}")
    resultado = edificios.loc[completa].copy()
    resultado["superficie_fv_detectada_m2"] = pixeles[completa] * RESOLUCION_PNOA_M**2
    resultado["superficie_fv_corregida_test_m2"] = (
        resultado.superficie_fv_detectada_m2 / FACTOR_CALIBRACION_TEST
    )
    resultado["porcentaje_cubierta_ocupado"] = (
        100 * resultado.superficie_fv_detectada_m2 / resultado.superficie_cubierta_m2
    )
    resultado["presencia_fotovoltaica"] = pixeles[completa] > 0
    return resultado.drop(columns="index_right", errors="ignore")


def resumir_municipios(
    edificios: gpd.GeoDataFrame, municipios: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Agrega los indicadores por municipio, incluidos los que tienen cero edificios.

    Args:
        edificios (gpd.GeoDataFrame): Edificios completamente analizados.
        municipios (gpd.GeoDataFrame): Límites municipales originales.

    Returns:
        gpd.GeoDataFrame: Resumen georreferenciado por municipio.
    """
    agrupados = edificios.groupby("codigo_municipal").agg(
        edificios_analizados=("id_edificio", "size"),
        edificios_con_deteccion=("presencia_fotovoltaica", "sum"),
        superficie_fv_detectada_total_m2=("superficie_fv_detectada_m2", "sum"),
        superficie_fv_corregida_test_total_m2=(
            "superficie_fv_corregida_test_m2",
            "sum",
        ),
        superficie_cubiertas_analizadas_m2=("superficie_cubierta_m2", "sum"),
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
        edificios (gpd.GeoDataFrame): Indicadores por edificio.
        municipios (gpd.GeoDataFrame): Indicadores municipales.
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


def main() -> None:
    """Aplica la U-Net actual a las teselas de Cantabria y resume sus edificios."""
    indice, edificios, municipios = cargar_datos()
    extensiones = comprobar_ortofotos(indice, edificios.crs)
    edificios = asociar_municipios(edificios, municipios)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = cargar_modelo(PESOS, dispositivo)
    print(
        f"Dispositivo: {dispositivo}; teselas: {len(indice)}; edificios: {len(edificios)}"
    )
    pixeles, superficie_cubierta = procesar_teselas(
        indice, edificios, extensiones, modelo, dispositivo
    )
    resultado_edificios = calcular_edificios(edificios, pixeles, superficie_cubierta)
    resultado_municipios = resumir_municipios(resultado_edificios, municipios)
    guardar_resultados(resultado_edificios, resultado_municipios)


if __name__ == "__main__":
    main()
