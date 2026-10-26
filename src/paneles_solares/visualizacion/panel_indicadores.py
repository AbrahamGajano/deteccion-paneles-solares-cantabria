"""Visor único del inventario histórico y de su muestra temporal comparable."""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import pandas as pd
from shapely import covers, prepare, union_all

from paneles_solares.rutas import (
    INDICADORES_SEGUIMIENTO,
    MUNICIPIOS_CANTABRIA,
    VISUALIZACION,
    huella_archivo,
    ruta_proyecto,
    ruta_registrada,
)
from paneles_solares.visualizacion.estimar_energia import (
    TOTALES_ENERGIA,
    energia_union,
    hipotesis_historicas,
    preparar_energia_historica,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SALIDA = VISUALIZACION
PLANTILLAS = Path(__file__).with_name("plantillas")


def campanias_comparables(serie: pd.DataFrame) -> list[int]:
    """Obtiene las campañas con superficies revisadas que admiten comparación."""

    anios = sorted(serie.loc[serie.superficie_comparable, "anio"].unique().tolist())

    return anios if len(anios) >= 2 else []


def leer_datos(carpeta: Path, limite: int | None = None):
    """Lee resultados terminados; el límite permite probar sin alterarlos."""

    archivo = carpeta / "atribucion.gpkg"
    candidatos = gpd.read_file(archivo, layer="candidatos_instalacion")
    if limite is not None:
        candidatos = candidatos.head(limite).copy()
    paneles = gpd.read_file(archivo, layer="paneles")
    paneles = paneles[paneles.id_instalacion.isin(candidatos.id_instalacion)].copy()
    serie = pd.read_csv(carpeta / "serie_historica.csv")
    serie = serie[serie.id_instalacion.isin(candidatos.id_instalacion)].copy()
    periodos = pd.read_csv(carpeta / "periodos_instalacion.csv")
    candidatos = candidatos.merge(
        periodos, on="id_instalacion", suffixes=("", "_periodo")
    )

    municipios = gpd.read_file(MUNICIPIOS_CANTABRIA, layer="municipios").to_crs(
        paneles.crs
    )
    municipios["codigo"] = municipios.codigo.astype(str)
    municipios.loc[municipios.codigo == "34063953072", "tipo_unidad"] = (
        "territorio_comunal"
    )
    puntos = candidatos[["id_instalacion", "geometry"]].copy()
    puntos.geometry = puntos.representative_point()
    enlace = gpd.sjoin(puntos, municipios[["codigo", "geometry"]], predicate="within")
    candidatos["codigo"] = candidatos.id_instalacion.map(
        enlace.set_index("id_instalacion").codigo
    ).fillna("fuera")
    candidatos["uso"] = candidatos.usos_cartograficos.fillna("").replace(
        "", "Sin dato cartográfico"
    )

    # La cohorte se calcula una sola vez, antes de seleccionar el año o municipio.
    anios = campanias_comparables(serie)
    comparables = (
        serie[serie.anio.isin(anios)]
        .groupby("id_instalacion")
        .agg(n=("anio", "nunique"), completa=("superficie_comparable", "all"))
    )
    cohorte = comparables.index[comparables.completa & (comparables.n == len(anios))]
    candidatos["muestra"] = candidatos.id_instalacion.isin(cohorte)
    contexto = {}
    for capa, columna, lista in (
        ("edificios_2023", "id_edificio", "edificios"),
        ("parcelas_2023", "id_parcela", "parcelas"),
    ):
        datos = gpd.read_file(archivo, layer=capa)
        ids = set(candidatos[lista].fillna("").str.split(";").explode())
        columnas = [columna, "geometry"]
        if capa == "parcelas_2023":
            columnas.append("referencia_catastral")
        contexto[capa] = datos[datos[columna].isin(ids)][columnas]
    return candidatos, paneles, serie, municipios, contexto


def desgloses(tabla: pd.DataFrame):
    """Emite los totales y filtros de uso y revisión sin sumar grupos solapados."""

    for uso in ["*"] + sorted(tabla.uso.unique()):
        parte = tabla if uso == "*" else tabla[tabla.uso == uso]
        for revision in ["*"] + sorted(parte.revision.unique()):
            yield (
                uso,
                revision,
                parte if revision == "*" else parte[parte.revision == revision],
            )


def recortar_paneles(paneles, municipios):
    """Separa el área municipal y exterior conservando la geometría de cada máscara."""

    geometrias = paneles[["id_instalacion", "anio", "geometry"]]
    limites = municipios[["codigo", "geometry"]]
    recortes = gpd.overlay(geometrias, limites, how="intersection", keep_geom_type=True)
    territorio = union_all(limites.geometry)
    prepare(territorio)
    exteriores = geometrias[~covers(territorio, geometrias.geometry)].copy()
    exteriores.geometry = exteriores.geometry.difference(territorio)
    exteriores = exteriores[~exteriores.is_empty].copy()
    exteriores["codigo"] = "fuera"
    return pd.concat([recortes, exteriores], ignore_index=True)


def resumir_mapa(candidatos, paneles, serie, municipios, recortes=None, energia=None):
    """Mide uniones en EPSG:25830, sin duplicar las máscaras solapadas.

    Los recuentos usan el punto interior del candidato. Las áreas se recortan al
    límite municipal; la parte exterior conserva su propia categoría. El total
    incluye ambas y disuelve también solapes entre candidatos distintos.
    """

    atributos = candidatos[["id_instalacion", "codigo", "uso", "revision", "muestra"]]
    estados = serie.merge(atributos, on="id_instalacion")
    geometrias = paneles[["id_instalacion", "anio", "geometry"]].merge(
        atributos.drop(columns="codigo"), on="id_instalacion"
    )
    if recortes is None:
        recortes = recortar_paneles(paneles, municipios)
    recortes = recortes.merge(atributos.drop(columns="codigo"), on="id_instalacion")

    filas = []
    campanias = [
        ("inventario", sorted(serie.anio.unique())),
        ("muestra", campanias_comparables(serie)),
    ]

    for modo, anios in campanias:
        for anio in anios:
            tabla = estados[estados.anio == anio]
            mascaras = recortes[recortes.anio == anio]
            originales = geometrias[geometrias.anio == anio]
            if modo == "muestra":
                tabla = tabla[tabla.muestra]
                mascaras = mascaras[mascaras.muestra]
                originales = originales[originales.muestra]
            for uso, revision, parte in desgloses(tabla):
                ids = parte.id_instalacion
                recorte = mascaras[mascaras.id_instalacion.isin(ids)]
                todas = originales[originales.id_instalacion.isin(ids)]
                con_mascara = set(todas.id_instalacion)
                codigos = sorted(set(parte.codigo) | set(recorte.codigo))
                inicio = len(filas)
                mezcla = (
                    energia[(energia.anio == anio) & energia.id_instalacion.isin(ids)]
                    if energia is not None
                    else None
                )
                for codigo in ["*"] + codigos:
                    grupo = parte if codigo == "*" else parte[parte.codigo == codigo]
                    formas = (
                        todas if codigo == "*" else recorte[recorte.codigo == codigo]
                    )
                    ausentes = int((grupo.estado == "ausente").sum())
                    area = (
                        float(union_all(formas.geometry).area) if len(formas) else None
                    )
                    if area is None and ausentes:
                        area = (
                            0.0  # Cero observado en la muestra, nunca cero territorial.
                        )
                    filas.append(
                        {
                            "modo": modo,
                            "anio": anio,
                            "uso": uso,
                            "revision": revision,
                            "codigo": codigo,
                            "superficie_m2": area,
                            "candidatos": len(grupo),
                            "con_mascara": int(
                                grupo.id_instalacion.isin(con_mascara).sum()
                            ),
                            "presentes": int((grupo.estado == "presente").sum()),
                            "ausentes": ausentes,
                            "sin_estado_resuelto": int(
                                (~grupo.estado.isin(["presente", "ausente"])).sum()
                            ),
                        }
                    )
                    if mezcla is not None and codigo != "*":
                        filas[-1].update(
                            energia_union(area, mezcla[mezcla.codigo == codigo])
                        )
                if mezcla is not None:
                    for columna in TOTALES_ENERGIA:
                        filas[inicio][columna] = (
                            sum(
                                f[columna]
                                for f in filas[inicio + 1 :]
                                if f[columna] is not None
                            )
                            if filas[inicio]["superficie_m2"] is not None
                            else None
                        )
    return pd.DataFrame(filas)


def geojson(tabla, tolerancia=0.02):
    """Transforma la copia de dibujo; las áreas usan la geometría original."""

    dibujo = tabla.copy()
    dibujo.geometry = dibujo.geometry.simplify(tolerancia)
    return json.loads(dibujo.to_crs(4326).to_json(drop_id=True))


def preparar_contenido(candidatos, paneles, serie, municipios, contexto, resumen):
    """Empaqueta las fichas y las capas necesarias para explorar el mapa."""

    puntos = candidatos.to_crs(4326).representative_point()
    fichas = candidatos.drop(columns="geometry").copy()
    fichas["lat"] = puntos.y.to_numpy()
    fichas["lon"] = puntos.x.to_numpy()
    columnas = [
        "id_instalacion",
        "codigo",
        "uso",
        "revision",
        "muestra",
        "lat",
        "lon",
        "edificios",
        "parcelas",
        "estado_periodo",
        "instalacion_desde",
        "instalacion_hasta",
        "ausencia_posterior",
        "instalacion_confirmada",
    ]
    contenido = {
        "candidatos": fichas[columnas].to_dict("records"),
        "serie": serie[
            [
                "id_instalacion",
                "anio",
                "estado",
                "superficie_horizontal_anotada_m2",
                "superficie_horizontal_modelo_m2",
                "conflicto",
                "superficie_comparable",
            ]
        ].to_dict("records"),
        "resumen": resumen.to_dict("records"),
        "municipios": geojson(
            municipios[["codigo", "municipio", "tipo_unidad", "geometry"]], 20
        ),
        "paneles": geojson(
            paneles[
                [
                    "id_panel",
                    "id_instalacion",
                    "anio",
                    "fuente_geometria",
                    "atribucion",
                    "geometry",
                ]
            ]
        ),
        "contexto": {nombre: geojson(tabla) for nombre, tabla in contexto.items()},
        "anios": sorted(serie.anio.unique().tolist()),
        "comparables": campanias_comparables(serie),
        "fuentes": {
            int(anio): grupo.fuente_geometria.unique().tolist()
            for anio, grupo in paneles.groupby("anio")
        },
        "energia": hipotesis_historicas()
        if "produccion_central_mwh" in resumen
        else None,
    }
    # Los valores desconocidos de pandas se publican como null, nunca como NaN.
    return json.loads(
        pd.Series([contenido]).to_json(orient="values", double_precision=7)
    )[0]


def crear_resumen_png(resumen, destino: Path):
    """Exporta la comparación de la muestra común con superficies revisadas."""

    tabla = resumen[
        (resumen.modo == "muestra")
        & (resumen.codigo == "*")
        & (resumen.uso == "*")
        & (resumen.revision == "*")
    ]
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ejes[0].bar(tabla.anio.astype(str), tabla.superficie_m2, color="#157d79")
    ejes[0].set_ylabel("Superficie horizontal anotada (m²; unión de máscaras)")
    ejes[1].bar(tabla.anio.astype(str), tabla.presentes, color="#28679c")
    ejes[1].set_ylabel("Candidatos con presencia confirmada en la imagen")
    n = int(tabla.candidatos.iloc[0]) if len(tabla) else 0
    anios = tabla.anio.tolist()
    periodo = f"{min(anios)}–{max(anios)}" if anios else "Sin campañas comparables"
    figura.suptitle(f"Muestra común de {n:,} candidatos · {periodo}".replace(",", "."))
    figura.supxlabel(
        "Inventario seleccionado; agrupaciones provisionales. No es un censo de Cantabria.",
        fontsize=9,
    )
    figura.savefig(destino, dpi=170)
    plt.close(figura)


def crear_panel(contenido: dict, destino: Path):
    """Compone el HTML con las mismas plantillas para la prueba y el lote completo."""

    html = (PLANTILLAS / "panel_indicadores.html").read_text(encoding="utf-8")
    for etiqueta, texto in (
        ("ESTILOS", (PLANTILLAS / "panel_indicadores.css").read_text(encoding="utf-8")),
        (
            "DATOS",
            json.dumps(
                contenido, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).replace("</", "<\\/"),
        ),
        ("CODIGO", (PLANTILLAS / "panel_indicadores.js").read_text(encoding="utf-8")),
    ):
        html = html.replace("{{" + etiqueta + "}}", texto)

    destino.write_text(html, encoding="utf-8")


def main():
    """Genera un solo HTML, su resumen municipal y una figura exportable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--carpeta", type=ruta_proyecto, default=INDICADORES_SEGUIMIENTO
    )
    parser.add_argument("--salida", type=ruta_proyecto, default=SALIDA)
    parser.add_argument(
        "--limite", type=int, help="Prueba con los primeros N candidatos"
    )
    args = parser.parse_args()
    if args.limite is not None and args.salida == SALIDA:
        parser.error("La prueba necesita --salida para conservar el visor completo")

    print("Leyendo candidatos, observaciones y contexto de 2023…", flush=True)
    candidatos, paneles, serie, municipios, contexto = leer_datos(
        args.carpeta, args.limite
    )
    args.salida.mkdir(parents=True, exist_ok=True)
    print("Preparando radiación y escenarios energéticos…", flush=True)
    recortes = recortar_paneles(paneles, municipios)
    energia = preparar_energia_historica(candidatos, paneles, recortes, args.salida)
    print("Calculando áreas sin solapes y la muestra comparable…", flush=True)
    resumen = resumir_mapa(candidatos, paneles, serie, municipios, recortes, energia)
    contenido = preparar_contenido(
        candidatos, paneles, serie, municipios, contexto, resumen
    )
    crear_panel(contenido, args.salida / "panel_indicadores.html")
    resumen.to_csv(args.salida / "resumen_municipal.csv", index=False)
    columnas = [
        "modo",
        "anio",
        "uso",
        "revision",
        "codigo",
        "superficie_m2",
    ] + TOTALES_ENERGIA
    resumen[columnas].to_csv(args.salida / "produccion_estimada.csv", index=False)
    (args.salida / "hipotesis_energia.json").write_text(
        json.dumps(hipotesis_historicas(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    crear_resumen_png(resumen, args.salida / "resumen_indicadores.png")
    archivos = [
        args.carpeta / n
        for n in ("atribucion.gpkg", "serie_historica.csv", "periodos_instalacion.csv")
    ]
    archivos.append(MUNICIPIOS_CANTABRIA)
    archivos.append(args.salida / "pvgis_anual.json")
    procedencia = {
        "fuentes": {
            ruta_registrada(p): huella_archivo(p) for p in archivos
        },
        "limite_prueba": args.limite,
        "candidatos": len(candidatos),
        "muestra_comparable": int(candidatos.muestra.sum()),
        "superficies": "Horizontales; unión de polígonos, sin corregir inclinación",
        "alcance": "Inventario seleccionado, agrupaciones provisionales; no censo regional",
    }
    (args.salida / "procedencia_panel.json").write_text(
        json.dumps(procedencia, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.salida / "panel_indicadores.html")


if __name__ == "__main__":
    main()
