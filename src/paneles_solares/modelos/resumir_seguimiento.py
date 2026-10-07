"""Auditoría de conservación y resultados históricos derivados del progreso humano."""

import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely import from_wkb
from shapely.geometry import box

from paneles_solares.datos.seguimiento import (
    RELACIONES,
    SOPORTES,
    derivar_historia,
    guardar_csv,
    guardar_json,
)
from paneles_solares.rutas import ruta_proyecto

SALIDA = ruta_proyecto("runs/evaluacion/seguimiento")


def trazar_geometria(dibujo, geometria, limites, tamano, color, ancho=2):
    """Dibuja contornos métricos sin modificar ninguna geometría original."""
    xmin, ymin, xmax, ymax = limites
    partes = geometria.geoms if hasattr(geometria, "geoms") else [geometria]
    for parte in partes:
        if not hasattr(parte, "exterior"):
            continue
        for anillo in [parte.exterior, *parte.interiors]:
            puntos = [
                ((x - xmin) / (xmax - xmin) * tamano[0], (ymax - y) / (ymax - ymin) * tamano[1])
                for x, y in anillo.coords
            ]
            dibujo.line(puntos, fill=color, width=ancho)


def generar_auditoria(almacen, salida: Path = SALIDA) -> dict:
    """Audita las predicciones completas disponibles, sin fingir cobertura regional.

    Args:
        almacen: Inventario y recursos originales.
        salida: Carpeta fija de resultados.

    Returns:
        Totales, superficies, municipio y límites de recuperación.
    """
    salida.mkdir(parents=True, exist_ok=True)
    componentes = almacen.filas("SELECT * FROM componentes WHERE anio=2023")
    columnas = [
        "id",
        "ventana",
        "anio",
        "area",
        "borde",
        "relacion",
        "fraccion",
        "distancia",
        "municipio",
        "codigo_municipal",
        "origen",
    ]
    tabla = pd.DataFrame([{c: f[c] for c in columnas} for f in componentes], columns=columnas)
    intersectan = tabla.fraccion > 0
    proximos = tabla.relacion == "proximo"
    exteriores = ~intersectan & ~proximos
    grupos = {
        "total_antes_catastro": tabla,
        "intersectan": tabla[intersectan],
        "proximos_sin_interseccion": tabla[proximos & ~intersectan],
        "completamente_exteriores": tabla[exteriores],
        "descartados_antes": tabla[~intersectan],
    }
    cifras = {
        nombre: {"componentes": len(filas), "superficie_m2": round(float(filas.area.sum()), 2)}
        for nombre, filas in grupos.items()
    }
    sin_interseccion = tabla[~intersectan].distancia.dropna()
    cifras["distancias_exteriores_m"] = (
        {str(p): round(float(sin_interseccion.quantile(p)), 2) for p in [0.1, 0.5, 0.9, 0.95]}
        if len(sin_interseccion)
        else {}
    )
    cifras["superficie_exterior_en_componentes_parciales_m2"] = round(
        float((tabla.loc[intersectan, "area"] * (1 - tabla.loc[intersectan, "fraccion"])).sum()), 2
    )
    cifras["cobertura"] = "recortes_preparados; no es auditoría regional completa"
    cifras["registros_anteriores_con_deteccion"] = 3188
    cifras["teselas_indice_anterior"] = 100591
    cifras["ventanas_2023_completas"] = almacen.filas(
        "SELECT COUNT(*) n FROM recursos WHERE anio=2023 AND prediccion<>''"
    )[0]["n"]
    cifras["region"] = {
        modo: almacen.meta("cursor_region_" + modo, {"completo": False, "procesadas": 0})
        for modo in ("indice", "fuera_indice", "todas")
    }
    cifras["mascaras_regionales_originales_disponibles"] = False
    cifras["recuperacion_pendiente"] = (
        "Reinferir el índice original sin recorte y las ventanas locales sin edificios; los resúmenes antiguos no permiten recuperar máscaras."
    )
    cifras["proximidad_m"] = almacen.meta("proximidad_m")
    cifras["diagnostico_catastro"] = almacen.meta("diagnostico_catastro", {})
    cifras["estrategia_catastro"] = (
        "Distancia y fracción originales; próximos sin asignación automática. La relación nunca filtra paneles."
    )
    cifras["por_origen"] = {
        origen: {
            "componentes": len(grupo),
            "exteriores": int((grupo.fraccion == 0).sum()),
            "superficie_m2": round(float(grupo.area.sum()), 2),
        }
        for origen, grupo in tabla.groupby("origen")
    }
    distribucion = []
    for nombre, filas in grupos.items():
        for municipio, grupo in filas.groupby("municipio"):
            distribucion.append(
                {
                    "grupo": nombre,
                    "municipio": municipio,
                    "componentes": len(grupo),
                    "superficie_m2": float(grupo.area.sum()),
                }
            )
    guardar_csv(tabla, salida / "auditoria_componentes.csv")
    guardar_csv(pd.DataFrame(distribucion), salida / "auditoria_municipios.csv")
    guardar_json(cifras, salida / "auditoria.json")
    # Pequeños, grandes, bordes y municipios diversos por relación. No se certifica el soporte mirando Catastro.
    seleccion = []
    for relacion in RELACIONES:
        filas = tabla[tabla.relacion == relacion].sort_values("area")
        if filas.empty:
            continue
        seleccion.extend(filas.head(2).id)
        seleccion.extend(filas.tail(2).id)
        seleccion.extend(filas[filas.borde == 1].head(2).id)
        seleccion.extend(
            filas.drop_duplicates("municipio")
            .iloc[:: max(1, len(filas.drop_duplicates("municipio")) // 3)]
            .head(3)
            .id
        )
    seleccion = list(dict.fromkeys(seleccion))
    almacen.poner_meta("muestra_auditoria", seleccion)
    _galeria_auditoria(almacen, componentes, seleccion, salida)
    almacen.poner_meta("auditoria_generada", cifras)
    return cifras


def _galeria_auditoria(almacen, componentes, seleccion, salida):
    """Crea una muestra visual offline con recortes existentes y sus contornos."""
    edificios = gpd.read_file(ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg"))
    carpeta = salida / "muestra_auditoria"
    carpeta.mkdir(exist_ok=True)
    tarjetas = []
    por_id = {c["id"]: c for c in componentes}
    for cid in seleccion:
        c = por_id[cid]
        recurso = almacen.filas(
            "SELECT r.*,v.xmin,v.ymin,v.xmax,v.ymax FROM recursos r JOIN ventanas v ON v.id=r.ventana WHERE r.ventana=? AND r.anio=2023",
            (c["ventana"],),
        )
        if not recurso or not Path(recurso[0]["imagen"]).exists():
            continue
        r = recurso[0]
        limites = tuple(r[k] for k in ("xmin", "ymin", "xmax", "ymax"))
        imagen = Image.open(r["imagen"]).convert("RGB").resize((512, 512))
        dibujo = ImageDraw.Draw(imagen)
        for g in edificios.geometry.iloc[
            edificios.sindex.query(box(*limites), predicate="intersects")
        ]:
            trazar_geometria(dibujo, g, limites, imagen.size, "#ffd740", 1)
        trazar_geometria(dibujo, from_wkb(c["geometria"]), limites, imagen.size, "#ff395e", 3)
        imagen.save(carpeta / f"{cid}.png")
        distancia = f"{c['distancia']:.2f}" if c["distancia"] is not None else "?"
        tarjetas.append(
            f'<article><img src="muestra_auditoria/{cid}.png"><p>{html.escape(c["municipio"])} · {c["relacion"]}<br>{c["area"]:.2f} m² · distancia {distancia} m · borde {c["borde"]}<br>{cid}</p></article>'
        )
    documento = (
        '<!doctype html><meta charset="utf-8"><title>Auditoría de detecciones</title><style>body{font:16px system-ui;background:#17232e;color:white;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}img{width:100%}article{background:#263746;padding:12px}p{overflow-wrap:anywhere}</style><h1>Auditoría de los recortes preparados</h1><p>Amarillo: Catastro. Rojo: componente original, sin recorte catastral. La existencia y el soporte requieren revisión visual. No representa cobertura regional completa.</p><main>'
        + "".join(tarjetas)
        + "</main>"
    )
    (salida / "auditoria_visual.html").write_text(documento, encoding="utf-8")


def generar_resumen(almacen, salida: Path = SALIDA, geopackage: bool = False) -> dict:
    """Exporta observaciones y conclusiones prudentes desde la base única.

    Args:
        almacen: Progreso persistente.
        salida: Carpeta fija de resultados regenerables.
        geopackage: Incluye las geometrías originales y agrupadas.

    Returns:
        Cifras observadas; sin extrapolación a un censo administrativo.
    """
    salida.mkdir(parents=True, exist_ok=True)
    sitios = almacen.filas("SELECT * FROM emplazamientos")
    activos = [s for s in sitios if s["activo"]]
    observaciones = almacen.filas("SELECT * FROM observaciones")
    anios = sorted(c["anio"] for c in almacen.meta("campanias", []))
    por_id = {}
    for o in observaciones:
        por_id.setdefault(o["emplazamiento"], []).append(o)
    conclusiones = [
        {
            "id": s["id"],
            "municipio": s["municipio"],
            "activo": s["activo"],
            **derivar_historia(por_id.get(s["id"], []), anios),
        }
        for s in sitios
    ]
    actuales = [o for o in observaciones if o["emplazamiento"] in {s["id"] for s in activos}]
    validos = [
        s
        for s in activos
        if s["agrupacion"] == "confirmada"
        and any(
            o["decision"] == "presente" and o["alcance"] == "emplazamiento_completo"
            for o in por_id.get(s["id"], [])
        )
    ]
    presentes_2023 = [
        s
        for s in validos
        if any(o["anio"] == 2023 and o["decision"] == "presente" for o in por_id.get(s["id"], []))
    ]
    historicos = [a for a in anios if a < 2023]
    observaciones_historicas = [o for o in actuales if o["anio"] in historicos]
    referencias = [o for o in actuales if o["anio"] == 2023]
    tiempos = [o["segundos"] for o in observaciones_historicas if o["segundos"] >= 0]
    total = len(activos) * len(anios)
    pendientes = total - len(actuales)
    total_historicos = len(activos) * len(historicos)
    pendientes_historicos = total_historicos - len(observaciones_historicas)
    media = float(np.mean(tiempos)) if tiempos else None
    componentes = almacen.filas("SELECT * FROM componentes")
    c2023 = [c for c in componentes if c["anio"] == 2023]
    resumen = {
        "alcance": "Emplazamientos observados en el inventario preparado; no censo administrativo ni fecha exacta de instalación.",
        "componentes_originales_2023": len(c2023),
        "componentes_todas_campanias": len(componentes),
        "agrupaciones_activas": len(activos),
        "agrupaciones_archivadas": len(sitios) - len(activos),
        "emplazamientos_validos_observados": len(validos),
        "candidatas_con_presencia_observada": sum(
            any(o["decision"] == "presente" for o in por_id.get(s["id"], [])) for s in activos
        ),
        "emplazamientos_confirmados_2023": len(presentes_2023),
        "agrupaciones_pendientes": sum(s["agrupacion"] != "confirmada" for s in activos),
        "soportes_confirmados": {
            tipo: sum(s["soporte"] == tipo for s in validos) for tipo in SOPORTES
        },
        "cubiertas_catastrales_con_fv_confirmadas": len(
            {
                c
                for s in validos
                if s["soporte"] in ("cubierta_catastrada", "mixto")
                for c in json.loads(s["cubiertas"])
            }
        ),
        "cubiertas_no_catastrales": "Sin recuento automático: usa identificadores manuales de cubierta en la asociación.",
        "emplazamientos_sin_asociacion": sum(not json.loads(s["edificios"]) for s in validos),
        "componentes_recuperados_exteriores": sum(c["fraccion"] == 0 for c in c2023),
        "falsos_positivos_descartados": sum(
            o["decision"] == "referencia_incorrecta" for o in actuales
        ),
        "decisiones_guardadas": len(actuales),
        "decisiones_pendientes": pendientes,
        "historicos_guardados": len(observaciones_historicas),
        "historicos_pendientes": pendientes_historicos,
        "referencias_2023_revisadas": len(referencias),
        "referencias_2023_pendientes": len(activos) - len(referencias),
        "porcentaje_historico_completado": round(
            100 * len(observaciones_historicas) / total_historicos, 1
        )
        if total_historicos
        else 0,
        "porcentaje_completado": round(100 * len(actuales) / total, 1) if total else 0,
        "segundos_medios_por_decision": round(media, 1) if media is not None else None,
        "tiempo_medio_corresponde_a": "Respuesta histórica; incluye comprobar la referencia en el primer par y las pausas.",
        "horas_restantes_estimadas": round(pendientes_historicos * media / 3600, 1)
        if media is not None
        else None,
        "tiempo_incluye_pausas": True,
        "secuencias_anomalas": sum(c["secuencia_anomala"] for c in conclusiones if c["activo"]),
        "observaciones_dudosas": sum(o["decision"] == "dudosa" for o in actuales),
        "observaciones_no_evaluables": sum(o["decision"] == "no_evaluable" for o in actuales),
        "ampliaciones": sum("ampliacion" in json.loads(o["etiquetas"]) for o in actuales),
        "reducciones": sum("reduccion" in json.loads(o["etiquetas"]) for o in actuales),
        "region_completa": almacen.meta("cursor_region_todas", {}).get("completo", False)
        or all(
            almacen.meta("cursor_region_" + m, {}).get("completo", False)
            for m in ("indice", "fuera_indice")
        ),
        "auditoria": almacen.meta("auditoria_generada", {}),
        "campanias": {},
    }
    municipal = []
    for anio in anios:
        filas = [o for o in actuales if o["anio"] == anio]
        resumen["campanias"][str(anio)] = {
            d: sum(o["decision"] == d for o in filas)
            for d in ("presente", "ausente", "dudosa", "no_evaluable", "referencia_incorrecta")
        }
        resumen["campanias"][str(anio)]["pendientes"] = len(activos) - len(filas)
        for municipio in sorted({s["municipio"] for s in activos}):
            locales = [s for s in activos if s["municipio"] == municipio]
            ids = {s["id"] for s in locales}
            presentes = [
                s
                for s in validos
                if s["id"] in ids
                and any(
                    o["anio"] == anio and o["decision"] == "presente"
                    for o in por_id.get(s["id"], [])
                )
            ]
            municipal.append(
                {
                    "municipio": municipio,
                    "anio": anio,
                    "candidatos": len(locales),
                    "emplazamientos_presentes_confirmados": len(presentes),
                    "suelo": sum(s["soporte"] == "suelo" for s in presentes),
                    "marquesina": sum(s["soporte"] == "marquesina" for s in presentes),
                    "mixto": sum(s["soporte"] == "mixto" for s in presentes),
                    "cubiertas_catastrales": len(
                        {
                            c
                            for s in presentes
                            if s["soporte"] in ("cubierta_catastrada", "mixto")
                            for c in json.loads(s["cubiertas"])
                        }
                    ),
                    "observaciones": sum(o["emplazamiento"] in ids for o in filas),
                }
            )
    guardar_csv(
        pd.DataFrame(
            observaciones,
            columns=[
                "emplazamiento",
                "anio",
                "decision",
                "alcance",
                "etiquetas",
                "notas",
                "segundos",
                "fecha_utc",
                "original",
            ],
        ),
        salida / "observaciones.csv",
    )
    guardar_csv(
        pd.DataFrame([{k: v for k, v in s.items() if k != "geometria"} for s in sitios]),
        salida / "emplazamientos.csv",
    )
    guardar_csv(pd.DataFrame(conclusiones), salida / "intervalos_aparicion.csv")
    guardar_csv(pd.DataFrame(municipal), salida / "resultados_municipales.csv")
    guardar_csv(pd.DataFrame(almacen.filas("SELECT * FROM miembros")), salida / "agrupaciones.csv")
    guardar_csv(
        pd.DataFrame(almacen.filas("SELECT * FROM acciones")), salida / "historial_acciones.csv"
    )
    guardar_json(resumen, salida / "resumen.json")
    texto = (
        f"Se han registrado {len(actuales):,} observaciones ({resumen['porcentaje_completado']:.1f} % del inventario preparado). "
        f"De ellas, {len(observaciones_historicas):,} son históricas ({resumen['porcentaje_historico_completado']:.1f} %); quedan {pendientes_historicos:,} respuestas históricas. "
        "El recorrido es 2020, 2017 y 2014; 2023 se comprueba dentro de la primera comparación, sin una etapa aparte. "
        f"Hay {resumen['candidatas_con_presencia_observada']:,} candidatas con presencia observada, {len(validos):,} emplazamientos con presencia y agrupación confirmadas y {resumen['agrupaciones_pendientes']:,} agrupaciones pendientes; "
        f"{resumen['observaciones_dudosas']:,} observaciones dudosas y {resumen['observaciones_no_evaluables']:,} no evaluables.\n\n"
        f"Se conservaron {len(c2023):,} componentes de 2023; {resumen['componentes_recuperados_exteriores']:,} quedan completamente fuera de edificios. "
        "Estas detecciones necesitan validación visual. Los recuentos corresponden a emplazamientos observados entre vuelos, no a instalaciones administrativas ni a años exactos de instalación. "
        "La recuperación regional sigue pendiente. Tampoco se detectan automáticamente instalaciones retiradas antes de 2023: pueden añadirse como candidatos históricos al observarlas.\n"
    )
    (salida / "resumen_correo.txt").write_text(texto, encoding="utf-8")
    guardar_csv(
        pd.DataFrame([{"anio": a, **d} for a, d in resumen["campanias"].items()]),
        salida / "resumen.csv",
    )
    guardar_csv(
        pd.DataFrame(conclusiones)
        .groupby("estado_historia", dropna=False)
        .size()
        .rename("emplazamientos")
        .reset_index(),
        salida / "distribucion_aparicion.csv",
    )
    if geopackage:
        _exportar_geopackage(sitios, componentes, conclusiones, salida / "seguimiento.gpkg")
    return resumen


def _exportar_geopackage(sitios, componentes, conclusiones, ruta):
    """Publica geometrías por capas, separadas de observaciones y conclusiones."""
    temporal = ruta.with_name(".seguimiento_exportando.gpkg")
    temporal.unlink(missing_ok=True)
    try:
        for nombre, filas in (("emplazamientos", sitios), ("componentes_originales", componentes)):
            if not filas:
                continue
            datos = [
                {
                    **{k: v for k, v in f.items() if k != "geometria"},
                    "geometry": from_wkb(f["geometria"]),
                }
                for f in filas
            ]
            gpd.GeoDataFrame(datos, crs="EPSG:25830").to_file(
                temporal, layer=nombre, driver="GPKG", index=False
            )
        if temporal.exists():
            from paneles_solares.datos.seguimiento import sustituir_archivo

            sustituir_archivo(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)
