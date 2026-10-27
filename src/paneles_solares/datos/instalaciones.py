"""Agrupaciones físicas provisionales de paneles y correcciones explícitas."""

import hashlib

import geopandas as gpd
import pandas as pd
from shapely import union_all


def agrupar_paneles(paneles: gpd.GeoDataFrame, vinculos: pd.DataFrame | None = None):
    """Propone unidades por edificio y solape, conservando los conflictos.

    Args:
        paneles: Polígonos atribuidos, con identificador, año, edificio y parcela.
        vinculos: Correcciones opcionales con id_panel, id_instalacion y
            instalacion_confirmada. Permiten separar conjuntos o unir cubiertas.

    Returns:
        tuple: Paneles con candidato asignado y tabla de enlaces espaciales.
            Compartir parcela o estar cerca no crea una relación.
            Entre años se exige un solape del 50 % del área menor; dentro del
            mismo año, del 80 % del área mayor. Son reglas provisionales.
    """

    datos = paneles.reset_index(drop=True).copy()
    semillas = [
        f"edificio:{f.id_edificio}"
        if pd.notna(f.id_edificio)
        else f"panel:{f.id_panel}"
        for f in datos.itertuples()
    ]
    vecinos = {s: set() for s in semillas}
    enlaces = []
    izquierdas, derechas = datos.sindex.query(datos.geometry, predicate="intersects")
    for i, j in zip(izquierdas, derechas):
        if i >= j or semillas[i] == semillas[j]:
            continue
        a, b = datos.iloc[i], datos.iloc[j]
        area = a.geometry.intersection(b.geometry).area
        mismo_anio = a.anio == b.anio
        referencia = (
            max(a.geometry.area, b.geometry.area)
            if mismo_anio
            else min(a.geometry.area, b.geometry.area)
        )
        fraccion = area / referencia
        if fraccion < (0.8 if mismo_anio else 0.5):
            continue
        vecinos[semillas[i]].add(semillas[j])
        vecinos[semillas[j]].add(semillas[i])
        enlaces.append(
            {
                "id_panel_1": a.id_panel,
                "id_panel_2": b.id_panel,
                "tipo": "posible_duplicado" if mismo_anio else "solape_entre_campanias",
                "interseccion_m2": area,
                "fraccion_solape": fraccion,
            }
        )

    asignacion, conflictos = {}, set()
    pendientes = set(vecinos)
    while pendientes:
        primero = min(pendientes)
        componente, por_visitar = set(), [primero]
        while por_visitar:
            actual = por_visitar.pop()
            if actual in componente:
                continue
            componente.add(actual)
            por_visitar.extend(vecinos[actual] - componente)
        pendientes.difference_update(componente)
        edificios = sorted(s for s in componente if s.startswith("edificio:"))
        if len(edificios) > 1:
            conflictos.update(componente)
            asignacion.update({s: s for s in componente})
        else:
            raiz = edificios[0] if edificios else min(componente)
            asignacion.update({s: raiz for s in componente})

    datos["id_instalacion"] = [
        "fv_" + hashlib.sha256(asignacion[s].encode()).hexdigest()[:20]
        for s in semillas
    ]
    datos["conflicto_agrupacion"] = [s in conflictos for s in semillas]
    datos["origen_agrupacion"] = [
        "edificio_2023"
        if s.startswith("edificio:")
        else ("solape" if len(vecinos[s]) and s not in conflictos else "sin_resolver")
        for s in semillas
    ]
    datos["instalacion_confirmada"] = False
    if vinculos is not None and not vinculos.empty:
        correcciones = vinculos.dropna(subset=["id_instalacion"])
        if correcciones.id_panel.duplicated().any():
            raise ValueError("Cada panel debe tener una sola corrección de instalación")
        por_panel = correcciones.set_index("id_panel")
        elegidas = datos.id_panel.isin(por_panel.index)
        datos.loc[elegidas, "id_instalacion"] = datos.loc[elegidas, "id_panel"].map(
            por_panel.id_instalacion
        )
        datos.loc[elegidas, "origen_agrupacion"] = "manual"
        if "instalacion_confirmada" in por_panel:
            confirmadas = (
                por_panel.instalacion_confirmada.astype(str)
                .str.lower()
                .isin(("true", "1"))
            )
            datos["instalacion_confirmada"] = datos.id_panel.map(confirmadas).eq(True)
    datos["instalacion_confirmada"] = datos.groupby(
        "id_instalacion"
    ).instalacion_confirmada.transform("all")
    grupos = datos.set_index("id_panel").id_instalacion
    manuales = set(datos.loc[datos.origen_agrupacion.eq("manual"), "id_panel"])
    for enlace in enlaces:
        a, b = enlace["id_panel_1"], enlace["id_panel_2"]
        unidos = grupos[a] == grupos[b]
        enlace["decision"] = (
            "unido_provisional" if unidos else "conflicto_entre_edificios"
        )
        if a in manuales or b in manuales:
            enlace["decision"] = "unido_manual" if unidos else "separado_manual"
    columnas = [
        "id_panel_1",
        "id_panel_2",
        "tipo",
        "interseccion_m2",
        "fraccion_solape",
        "decision",
    ]
    return datos, pd.DataFrame(enlaces, columns=columnas)


def resumir_instalaciones(paneles, municipios) -> gpd.GeoDataFrame:
    """Describe candidatos sin convertir la primera presencia en fecha de instalación.

    Args:
        paneles: Miembros de los grupos automáticos o corregidos.
        municipios: Límites municipales en el mismo sistema de coordenadas.

    Returns:
        gpd.GeoDataFrame: Huella observada, soportes y revisión pendiente por candidato.
    """

    filas = []
    for identificador, grupo in paneles.groupby("id_instalacion"):
        edificios = sorted(grupo.id_edificio.dropna().unique())
        parcelas = sorted(grupo.id_parcela.dropna().unique())
        confirmada = bool(grupo.instalacion_confirmada.all())
        revision = (
            "comprobar_edificio" if edificios else "comprobar_emplazamiento_exterior"
        )
        if grupo.atribucion.eq("ambigua").any():
            revision = "resolver_atribucion"
        if grupo.conflicto_agrupacion.any():
            revision = "conflicto_entre_edificios"
        if confirmada:
            revision = "agrupacion_confirmada"
        filas.append(
            {
                "id_instalacion": identificador,
                "instalacion_confirmada": confirmada,
                "revision": revision,
                "edificios": ";".join(edificios),
                "parcelas": ";".join(parcelas),
                "referencias_catastrales": ";".join(
                    sorted(grupo.referencia_catastral.dropna().unique())
                ),
                "n_conjuntos_observados": len(grupo),
                "primera_campania_con_paneles": int(grupo.anio.min()),
                "ultima_campania_con_paneles": int(grupo.anio.max()),
                "usos_cartograficos": ";".join(
                    sorted(grupo.uso_del_suelo.dropna().unique())
                ),
                "geometry": union_all(grupo.geometry),
            }
        )
    instalaciones = gpd.GeoDataFrame(filas, geometry="geometry", crs=paneles.crs)
    puntos = instalaciones[["id_instalacion", "geometry"]].copy()
    puntos.geometry = puntos.representative_point()
    enlace = gpd.sjoin(
        puntos,
        municipios[["codigo", "municipio", "geometry"]],
        how="left",
        predicate="within",
    )
    instalaciones = instalaciones.merge(
        enlace[["id_instalacion", "codigo", "municipio"]], on="id_instalacion"
    )
    instalaciones["codigo"] = instalaciones.codigo.fillna("fuera")
    instalaciones["municipio"] = instalaciones.municipio.fillna("Fuera de límites")
    coordenadas = puntos.to_crs(4326)
    instalaciones["latitud"] = coordenadas.geometry.y.to_numpy()
    instalaciones["longitud"] = coordenadas.geometry.x.to_numpy()
    return instalaciones


def georreferenciar_serie(paneles, serie) -> gpd.GeoDataFrame:
    """Añade la huella de cada campaña a la tabla de instalaciones.

    Args:
        paneles: Polígonos originales agrupados, con año y fuente de geometría.
        serie: Observaciones y superficies de instalaciones.csv.

    Returns:
        gpd.GeoDataFrame: Una fila por instalación y campaña. Las ausencias y
            observaciones sin superficie disponible conservan geometría nula.
            Cada huella usa la misma fuente que la superficie de esa fila.
    """

    claves = ["id_instalacion", "anio"]
    datos = paneles.merge(serie[claves + ["fuente_superficie"]], on=claves)
    datos = datos[datos.fuente_geometria.eq(datos.fuente_superficie)]
    huellas = datos[claves + ["geometry"]].dissolve(by=claves, as_index=False)

    return gpd.GeoDataFrame(
        serie.merge(huellas, on=claves, how="left"),
        geometry="geometry",
        crs=paneles.crs,
    )


def resumir_superficies(paneles: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calcula superficie visible por candidato y año, sin inventar campañas ausentes.

    Args:
        paneles: Polígonos agrupados, con su campaña y estado de confirmación.

    Returns:
        pd.DataFrame: Áreas horizontales sin duplicar solapes; cobertura aún provisional.
    """

    filas = []
    for (identificador, anio), grupo in paneles.groupby(["id_instalacion", "anio"]):
        superficie = union_all(grupo.geometry).area
        filas.append(
            {
                "id_instalacion": identificador,
                "anio": anio,
                "n_conjuntos": len(grupo),
                "superficie_horizontal_observada_m2": superficie,
                "fuente_superficie": ";".join(sorted(grupo.fuente_geometria.unique()))
                if "fuente_geometria" in grupo
                else "labelme",
                "solape_entre_anotaciones_m2": max(
                    0, grupo.geometry.area.sum() - superficie
                ),
                "instalacion_confirmada": bool(grupo.instalacion_confirmada.all()),
                "anotaciones_completas": bool(grupo.revision_completa.all())
                if "revision_completa" in grupo
                else False,
                "recorte_en_borde": bool(grupo.borde_imagen.any())
                if "borde_imagen" in grupo
                else False,
                "cobertura_completa_confirmada": False,
            }
        )
    return pd.DataFrame(filas)
