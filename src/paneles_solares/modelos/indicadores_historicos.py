"""Calcula superficies y candidatos a instalación con anotaciones históricas."""

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid, normalize, union_all
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from paneles_solares.datos.anotaciones import leer_anotacion
from paneles_solares.datos.instalaciones import (
    agrupar_paneles,
    resumir_instalaciones,
    resumir_superficies,
)
from paneles_solares.datos.periodos import (
    comparar_superficies,
    observar_campanias,
    resumir_periodos,
)
from paneles_solares.datos.seguimiento import CARPETA, guardar_csv, recuperar_revision
from paneles_solares.geografia.catastro import relacionar
from paneles_solares.rutas import (
    EDIFICIOS_CATASTRO,
    INDICADORES_SEGUIMIENTO,
    huella_archivo,
    reemplazar_carpeta,
    retirar_panel,
    ruta_proyecto,
)

SALIDA = INDICADORES_SEGUIMIENTO
FRACCION_CLARA = 0.8


def georreferenciar_anotacion(datos: dict, ventana: dict):
    """Convierte los polígonos de píxeles a coordenadas EPSG:25830.

    Args:
        datos: Anotación de LabelMe con las dimensiones de la imagen.
        ventana: Límites geográficos del recorte.

    Returns:
        tuple: Geometrías, marcas de borde y número de polígonos reparados.
    """

    resolucion_x = (ventana["xmax"] - ventana["xmin"]) / datos["imageWidth"]
    resolucion_y = (ventana["ymax"] - ventana["ymin"]) / datos["imageHeight"]

    # En la imagen Y crece hacia abajo; en el mapa crece hacia el norte.
    transformacion = [
        resolucion_x,
        0,
        0,
        -resolucion_y,
        ventana["xmin"],
        ventana["ymax"],
    ]
    formas = []
    bordes = []
    reparadas = 0

    for forma in datos["shapes"]:
        geometria = Polygon(forma["points"])

        if not geometria.is_valid:
            geometria = make_valid(geometria)
            reparadas += 1

        formas.append(affine_transform(geometria, transformacion))
        borde = any(
            x <= 0 or y <= 0 or x >= datos["imageWidth"] or y >= datos["imageHeight"]
            for x, y in forma["points"]
        )
        bordes.append(borde)

    return formas, bordes, reparadas


def leer_paneles(
    manifest: Path, limite: int | None = None, ventanas: list[str] | None = None
):
    """Lee una muestra de JSON sin modificar imágenes ni aceptar trabajo parcial.

    Args:
        manifest: Manifiesto del preparador histórico, con coordenadas EPSG:25830.
        limite: Número máximo de ventanas; None procesa el manifiesto completo.
        ventanas: Identificadores concretos para repetir o ampliar el piloto.

    Returns:
        tuple: Polígonos georreferenciados y estado de cada imagen de la muestra.
            Los JSON guardados siguen siendo provisionales hasta terminar la revisión.
    """

    tabla = pd.read_csv(manifest, dtype={"ventana": str}, keep_default_na=False)
    elegidas = list(dict.fromkeys(tabla.ventana)) if ventanas is None else ventanas
    tabla = tabla[tabla.ventana.isin(elegidas[:limite])]
    paneles, observaciones = {}, []

    for fila in tabla.to_dict("records"):
        ruta = manifest.parent / fila["anotacion"]
        estado = "pendiente"
        formas = []
        bordes = []
        reparadas = 0
        completa = (
            fila.get("revision_completa") in (True, "True")
            and ruta.is_file()
            and huella_archivo(ruta) == fila.get("anotacion_sha256")
        )
        if ruta.exists():
            datos = leer_anotacion(ruta)
            flags = datos.get("flags") or {}
            if flags.get("dudosa") or flags.get("descartar"):
                estado = "no_evaluable"
            elif (flags.get("ausente") or completa) and not datos["shapes"]:
                estado = "ausencia_imagen"
            elif datos["shapes"]:
                estado = (
                    "presencia_anotada"
                    if completa
                    else "poligonos_guardados_sin_confirmar"
                )
                formas, bordes, reparadas = georreferenciar_anotacion(datos, fila)
                if any(
                    not g.is_valid
                    or g.area <= 0
                    or g.geom_type not in ("Polygon", "MultiPolygon")
                    for g in formas
                ):
                    estado, formas = "geometria_por_corregir", []
            else:
                estado = "sin_poligonos_sin_confirmar"

        observaciones.append(
            {
                "ventana": fila["ventana"],
                "anio": fila["anio"],
                "estado": estado,
                "anotacion": fila["anotacion"],
                "selecciones_antiguas": fila.get("selecciones", ""),
                "revision_completa": completa,
                "geometrias_reparadas": reparadas,
                "xmin": fila["xmin"],
                "ymin": fila["ymin"],
                "xmax": fila["xmax"],
                "ymax": fila["ymax"],
                "superficie_visible_m2": union_all(formas).area
                if formas
                else (0.0 if estado == "ausencia_imagen" else None),
            }
        )
        for geometria, borde in zip(formas, bordes):
            firma = str(fila["anio"]).encode() + normalize(geometria).wkb
            identificador = "panel_" + hashlib.sha256(firma).hexdigest()[:20]
            if identificador in paneles:
                anterior = paneles[identificador]
                anterior["ventanas"] = ";".join(
                    sorted(set(anterior["ventanas"].split(";")) | {fila["ventana"]})
                )
                anterior["revision_completa"] |= completa
                anterior["borde_imagen"] &= borde
                continue
            paneles[identificador] = {
                "id_panel": identificador,
                "anio": fila["anio"],
                "ventanas": fila["ventana"],
                "superficie_horizontal_m2": geometria.area,
                "anio_referencia": 2023,
                "revision_completa": completa,
                "borde_imagen": borde,
                "fuente_geometria": "labelme",
                "id_seleccion": "",
                "geometry": geometria,
            }

    columnas = [
        "id_panel",
        "anio",
        "ventanas",
        "superficie_horizontal_m2",
        "anio_referencia",
        "revision_completa",
        "borde_imagen",
        "fuente_geometria",
        "id_seleccion",
        "geometry",
    ]
    return (
        gpd.GeoDataFrame(
            paneles.values(), columns=columnas, geometry="geometry", crs=25830
        ),
        pd.DataFrame(
            observaciones,
            columns=[
                "ventana",
                "anio",
                "estado",
                "anotacion",
                "selecciones_antiguas",
                "revision_completa",
                "geometrias_reparadas",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "superficie_visible_m2",
            ],
        ),
    )


def elegir_relacion(relaciones: pd.DataFrame, campo: str) -> dict:
    """Acepta un soporte dominante sin ocultar solapes con otros candidatos.

    Args:
        relaciones: Intersecciones de paneles con edificios o parcelas.
        campo: Identificador del soporte, id_edificio o id_parcela.

    Returns:
        dict: Identificador claro por panel; las relaciones ambiguas siguen sin asignar.
    """

    elegidas = {}
    for panel, grupo in relaciones.groupby("id_panel"):
        ordenadas = grupo.sort_values("fraccion_objeto", ascending=False)
        primera = ordenadas.iloc[0]
        if (
            primera.fraccion_objeto >= FRACCION_CLARA
            and (grupo.fraccion_objeto >= 0.2).sum() == 1
        ):
            elegidas[panel] = primera[campo]
    return elegidas


def atribuir_paneles(paneles, edificios, parcelas):
    """Conserva todos los solapes y distingue atribución clara, ambigua y exterior.

    Args:
        paneles: Polígonos de una o varias campañas, sin fusionarlos entre años.
        edificios: Huellas CBT23, que no equivalen necesariamente a edificios completos.
        parcelas: Parcelario de 2023; compartir parcela no confirma una instalación.

    Returns:
        tuple: Paneles con atribución provisional y las dos tablas de intersecciones.
    """

    edificios_panel = relacionar(paneles, edificios, "id_panel", "id_edificio")
    parcelas_panel = relacionar(paneles, parcelas, "id_panel", "id_parcela")
    resultado = paneles.copy()
    resultado["id_edificio"] = resultado.id_panel.map(
        elegir_relacion(edificios_panel, "id_edificio")
    ).astype(object)
    resultado["id_parcela"] = resultado.id_panel.map(
        elegir_relacion(parcelas_panel, "id_parcela")
    ).astype(object)
    resultado["atribucion"] = "exterior_edificaciones"
    resultado.loc[resultado.id_panel.isin(edificios_panel.id_panel), "atribucion"] = (
        "ambigua"
    )
    resultado.loc[resultado.id_edificio.notna(), "atribucion"] = "edificio_probable"
    resultado["instalacion_confirmada"] = False
    resultado = resultado.merge(
        edificios[["id_edificio", "uso_del_suelo", "subtipo", "uso_fuente"]],
        on="id_edificio",
        how="left",
    )
    resultado = resultado.merge(
        parcelas[["id_parcela", "referencia_catastral"]],
        on="id_parcela",
        how="left",
    )
    return resultado, edificios_panel, parcelas_panel


def main() -> None:
    """Exporta la revisión histórica o una muestra limitada, sin ejecutar inferencia."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=CARPETA / "labelme/manifest.csv"
    )
    parser.add_argument(
        "--limite", type=int, help="Limitar ventanas para una prueba pequeña"
    )
    parser.add_argument("--ventana", action="append")
    parser.add_argument(
        "--base",
        type=Path,
        help="Recuperar referencia de 2023 y respuestas del visor anterior",
    )
    parser.add_argument(
        "--vinculos", type=Path, help="CSV opcional de agrupaciones revisadas"
    )
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()
    if args.limite is not None and args.limite < 1:
        parser.error("El límite debe ser positivo")
    destino = ruta_proyecto(args.salida)
    if (args.limite is not None or args.ventana) and destino == SALIDA:
        parser.error(
            "Indica --salida para guardar la prueba sin sustituir el lote completo"
        )
    paneles, observaciones = leer_paneles(
        ruta_proyecto(args.manifest), args.limite, args.ventana
    )
    referencia, selecciones, respuestas = None, None, None
    if args.base:
        referencia, selecciones, respuestas = recuperar_revision(
            ruta_proyecto(args.base)
        )
        if args.limite is not None or args.ventana:
            elegidas = set(args.ventana) if args.ventana else set(observaciones.ventana)
            referencia = referencia[referencia.ventanas.isin(elegidas)].copy()
            selecciones = selecciones[selecciones.ventana.isin(elegidas)].copy()
            respuestas = respuestas[
                respuestas.id_seleccion.isin(selecciones.id_seleccion)
            ].copy()
        activas = referencia[referencia.decision_referencia.eq("presente")]
        paneles = gpd.GeoDataFrame(
            pd.concat([paneles, activas], ignore_index=True), crs=25830
        )
    if paneles.empty:
        parser.error("No hay polígonos válidos guardados en las ventanas elegidas")
    print(
        f"Máscaras: {len(paneles)}; imágenes revisadas: {len(observaciones)}",
        flush=True,
    )
    entorno = paneles.geometry.buffer(2)
    contexto = []
    for capa in ("edificios", "parcelas"):
        datos = gpd.read_file(
            EDIFICIOS_CATASTRO, layer=capa, bbox=tuple(entorno.total_bounds)
        )
        posiciones = datos.sindex.query(entorno, predicate="intersects")[1]
        contexto.append(datos.iloc[sorted(set(posiciones))].reset_index(drop=True))
        print(f"Contexto: {len(contexto[-1])} {capa}", flush=True)
    edificios, parcelas = contexto
    paneles, relaciones_edificios, relaciones_parcelas = atribuir_paneles(
        paneles, edificios, parcelas
    )
    vinculos = None
    if args.vinculos:
        vinculos = pd.read_csv(
            ruta_proyecto(args.vinculos),
            dtype={"id_panel": str, "id_instalacion": str},
        )
    paneles, enlaces = agrupar_paneles(paneles, vinculos)
    instalaciones = resumir_instalaciones(paneles)
    serie = observar_campanias(
        paneles, observaciones, selecciones, respuestas, edificios
    )
    periodos = resumir_periodos(serie)
    salida = destino.with_name(f".{destino.name}_preparando")
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "atribucion.gpkg").unlink(missing_ok=True)
    (salida / "revision_anterior.csv").unlink(missing_ok=True)
    guardar_csv(observaciones, salida / "observaciones.csv")
    guardar_csv(
        resumir_superficies(paneles), salida / "superficies_por_instalacion.csv"
    )
    guardar_csv(enlaces, salida / "enlaces_espaciales.csv")
    guardar_csv(serie, salida / "serie_historica.csv")
    guardar_csv(periodos, salida / "periodos_instalacion.csv")
    guardar_csv(comparar_superficies(serie), salida / "cambios_superficie.csv")
    if respuestas is not None:
        guardar_csv(respuestas, salida / "revision_anterior.csv")
    columnas = [
        "id_panel",
        "id_instalacion",
        "anio",
        "ventanas",
        "origen_agrupacion",
        "instalacion_confirmada",
        "fuente_geometria",
    ]
    guardar_csv(paneles[columnas], salida / "miembros_instalaciones.csv")
    guardar_csv(relaciones_edificios, salida / "paneles_edificios.csv")
    guardar_csv(relaciones_parcelas, salida / "paneles_parcelas.csv")
    ruta = salida / "atribucion.gpkg"
    for nombre, capa in (
        ("paneles", paneles),
        ("candidatos_instalacion", instalaciones),
        ("edificios_2023", edificios),
        ("parcelas_2023", parcelas),
    ):
        capa.to_file(ruta, layer=nombre, index=False)
    if referencia is not None:
        referencia.to_file(ruta, layer="referencia_2023", index=False)
        selecciones.to_file(ruta, layer="selecciones_anteriores", index=False)
    resumen = {
        "alcance": "revision_historica_no_censo_ni_instalaciones_confirmadas",
        "ventanas": observaciones.ventana.nunique(),
        "imagenes": len(observaciones),
        "estados": observaciones.estado.value_counts().to_dict(),
        "revisiones_completas": int(observaciones.revision_completa.sum()),
        "geometrias_reparadas": int(observaciones.geometrias_reparadas.sum()),
        "poligonos": len(paneles),
        "fuentes_geometria": paneles.fuente_geometria.value_counts().to_dict(),
        "estados_periodo": periodos.estado_periodo.value_counts().to_dict(),
        "estados_anteriores_superados": int(serie.respuesta_anterior_superada.sum()),
        "ausencias_recuperadas": int(
            (
                serie.estado.eq("ausente")
                & serie.fuente_estado.eq("seleccion_anterior")
            ).sum()
        ),
        "atribuciones": paneles.atribucion.value_counts().to_dict(),
        "candidatos_instalacion": len(instalaciones),
        "instalaciones_confirmadas": int(instalaciones.instalacion_confirmada.sum()),
        "revision_agrupaciones": instalaciones.revision.value_counts().to_dict(),
        "anio_referencia_geografica": 2023,
        "pendiente": "Revisar vínculos dudosos y superficies estimadas de 2023; separar épocas de ampliaciones antes de estimar energía por instalación.",
    }
    (salida / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    reemplazar_carpeta(salida, destino)
    if destino == SALIDA:
        retirar_panel()
    print(json.dumps(resumen, ensure_ascii=False))
    print(destino)


if __name__ == "__main__":
    main()
