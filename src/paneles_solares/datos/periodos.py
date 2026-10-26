"""Observaciones por campaña e intervalos de inicio, sin fechar toda la superficie."""

import geopandas as gpd
import pandas as pd
from shapely import box, union_all


def observar_campanias(
    paneles, observaciones, selecciones=None, respuestas=None, edificios=None
):
    """Relaciona presencias y ausencias con la huella conocida de cada candidato.

    Args:
        paneles: Máscaras agrupadas, con fuente de geometría y año independientes.
        observaciones: Estado y límites de cada imagen revisada en LabelMe.
        selecciones: Geometrías seleccionadas en el visor anterior, opcionales.
        respuestas: Respuestas originales, referidas a esas selecciones.
        edificios: Huellas de referencia para comprobar el soporte completo.

    Returns:
        pd.DataFrame: Serie por candidato y campaña. Una ausencia necesita cubrir
            toda su huella conocida; no certifica el resto del edificio o parcela.
    """

    revisadas = observaciones[
        observaciones.revision_completa
        & observaciones.estado.isin(("presencia_anotada", "ausencia_imagen"))
    ].copy()
    limites = revisadas[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float).T
    imagenes = gpd.GeoDataFrame(revisadas, geometry=box(*limites), crs=paneles.crs)
    anios = set(observaciones.anio) | set(paneles.anio)
    positivas, negativas = set(), {}
    if selecciones is not None:
        anios.update(respuestas.anio)
        referencia = paneles[paneles.fuente_geometria.eq("modelo_2023")]
        correspondencias = {}
        for seleccion, grupo in referencia.groupby("id_seleccion"):
            esperadas = selecciones.loc[
                selecciones.id_seleccion.eq(seleccion), "n_componentes"
            ].item()
            if len(grupo) == esperadas and grupo.id_instalacion.nunique() == 1:
                correspondencias[seleccion] = grupo.id_instalacion.iloc[0]
        for r in respuestas[respuestas.decision.eq("presente")].itertuples():
            if r.id_seleccion in correspondencias:
                positivas.add((correspondencias[r.id_seleccion], r.anio))
        ausencias = respuestas[
            respuestas.decision.isin(("ausente", "referencia_incorrecta"))
            & respuestas.alcance.isin(
                ("emplazamiento_completo", "agrupacion_cubiertas")
            )
        ]
        negativas = {
            anio: grupo
            for anio, grupo in selecciones.merge(
                ausencias[["id_seleccion", "anio"]],
                on="id_seleccion",
            ).groupby("anio")
        }

    fotos = {anio: grupo for anio, grupo in imagenes.groupby("anio")}
    manuales = paneles[paneles.fuente_geometria.eq("labelme")]
    dibujados = {anio: grupo for anio, grupo in manuales.groupby("anio")}
    filas = []
    for identificador, grupo in paneles.groupby("id_instalacion"):
        huella = union_all(grupo.geometry)
        soporte = None
        if edificios is not None:
            geometrias = edificios[
                edificios.id_edificio.isin(grupo.id_edificio.dropna())
            ]
            if not geometrias.empty:
                soporte = union_all([huella, *geometrias.geometry])
        for anio in sorted(anios):
            actuales = grupo[grupo.anio.eq(anio)]
            manual = actuales[actuales.fuente_geometria.eq("labelme")]
            modelo = actuales[actuales.fuente_geometria.eq("modelo_2023")]
            cobertura, fraccion_ausencia, fuentes = 0.0, 0.0, ""
            vacia, soporte_revisado = False, False
            otras = False
            if anio in fotos:
                tabla = fotos[anio]
                posiciones = tabla.sindex.query(
                    soporte if soporte is not None else huella, predicate="intersects"
                )
                cercanas = tabla.iloc[posiciones]
                vista = union_all(cercanas.geometry)
                cobertura = min(1.0, huella.intersection(vista).area / huella.area)
                sin_paneles = union_all(
                    cercanas.loc[cercanas.estado.eq("ausencia_imagen"), "geometry"]
                )
                vacia = huella.difference(sin_paneles).area <= huella.area * 1e-8
                soporte_revisado = (
                    soporte is not None
                    and soporte.difference(vista).area <= soporte.area * 1e-8
                )
            if anio in dibujados:
                tabla = dibujados[anio]
                zona = soporte if soporte is not None else huella
                posiciones = tabla.sindex.query(zona, predicate="intersects")
                otras = any(
                    tabla.iloc[i].id_instalacion != identificador
                    and zona.intersection(tabla.iloc[i].geometry).area > 1e-6
                    for i in posiciones
                )
            if anio in negativas:
                tabla = negativas[anio]
                posiciones = tabla.sindex.query(huella, predicate="intersects")
                zona = union_all(tabla.geometry.iloc[posiciones])
                fraccion_ausencia = min(
                    1.0, huella.intersection(zona).area / huella.area
                )
                fuentes = ";".join(sorted(tabla.id_seleccion.iloc[posiciones].unique()))

            ausencia_manual = (vacia or soporte_revisado) and manual.empty and not otras
            ausencia_anterior = fraccion_ausencia >= 1 - 1e-8
            presencia_anterior = (identificador, anio) in positivas
            contradiccion = (
                ausencia_anterior
                and (not manual.empty or presencia_anterior or otras)
                or ausencia_manual
                and presencia_anterior
            )
            revision_resuelve = ausencia_manual or (
                not manual.empty and manual.revision_completa.all()
            )
            superada = bool(contradiccion and revision_resuelve)
            conflicto = bool(contradiccion and not revision_resuelve)
            estado, fuente, superficie = "desconocido", "", None
            if not manual.empty:
                estado = (
                    "presente"
                    if manual.revision_completa.any()
                    else "pendiente_revision"
                )
                fuente = "labelme"
                if manual.revision_completa.all():
                    superficie = union_all(manual.geometry).area
            elif ausencia_manual:
                estado, fuente, superficie = "ausente", "labelme", 0.0
            elif otras:
                estado, fuente = "vinculo_pendiente", "labelme"
            elif presencia_anterior:
                estado, fuente = "presente", "seleccion_anterior"
            elif ausencia_anterior:
                estado, fuente, superficie = "ausente", "seleccion_anterior", 0.0
            elif not modelo.empty:
                estado, fuente = "detectado_sin_confirmar", "modelo_2023"
            filas.append(
                {
                    "id_instalacion": identificador,
                    "anio": anio,
                    "estado": estado,
                    "fuente_estado": fuente,
                    "superficie_horizontal_anotada_m2": superficie,
                    "superficie_horizontal_modelo_m2": union_all(modelo.geometry).area
                    if not modelo.empty
                    else None,
                    "cobertura_huella_revisada": cobertura,
                    "fraccion_ausencia_anterior": fraccion_ausencia,
                    "selecciones_ausentes": fuentes,
                    "conflicto": bool(conflicto),
                    "respuesta_anterior_superada": superada,
                    "superficie_comparable": bool(
                        superficie is not None
                        and (
                            cobertura >= 1 - 1e-8
                            or estado == "ausente"
                            and ausencia_anterior
                        )
                        and manual.revision_completa.all()
                        and not manual.borde_imagen.any()
                        and not conflicto
                        and not otras
                    ),
                    "instalacion_confirmada": bool(grupo.instalacion_confirmada.all()),
                }
            )
    return pd.DataFrame(filas)


def resumir_periodos(serie: pd.DataFrame) -> pd.DataFrame:
    """Acota el inicio del candidato; sus ampliaciones no heredan esa antigüedad.

    Args:
        serie: Observaciones independientes por candidato y campaña.

    Returns:
        pd.DataFrame: Última ausencia previa y primera presencia. El límite
            inferior es exclusivo; un vacío conserva la antigüedad desconocida.
    """

    filas = []
    for identificador, grupo in serie.groupby("id_instalacion"):
        presentes = grupo[grupo.estado.eq("presente")]
        detectadas = grupo[grupo.estado.eq("detectado_sin_confirmar")]
        inicio = presentes.anio.min() if not presentes.empty else detectadas.anio.min()
        ausencias = grupo[grupo.estado.eq("ausente") & grupo.anio.lt(inicio)]
        desde = ausencias.anio.max() if not ausencias.empty else None
        estado = "acotado" if desde is not None else "sin_ausencia_previa"
        if presentes.empty:
            estado = (
                "presencia_por_confirmar" if not detectadas.empty else "sin_presencia"
            )
        intermitente = bool((grupo.estado.eq("ausente") & grupo.anio.gt(inicio)).any())
        if grupo.conflicto.any() or intermitente:
            estado = "revisar_cambio_o_contradiccion"
        filas.append(
            {
                "id_instalacion": identificador,
                "instalacion_desde": desde,
                "instalacion_hasta": int(inicio) if pd.notna(inicio) else None,
                "estado_periodo": estado,
                "fuente_primera_presencia": grupo.loc[
                    grupo.anio.eq(inicio), "fuente_estado"
                ].iloc[0]
                if pd.notna(inicio)
                else "",
                "ausencia_posterior": intermitente,
                "instalacion_confirmada": bool(grupo.instalacion_confirmada.all()),
            }
        )
    return pd.DataFrame(filas)


def comparar_superficies(serie: pd.DataFrame) -> pd.DataFrame:
    """Compara áreas anotadas en campañas consecutivas con la misma huella revisada.

    Args:
        serie: Áreas manuales y estimadas separadas, con cobertura y conflictos.

    Returns:
        pd.DataFrame: Variaciones medidas; las comparaciones no fiables quedan vacías.
            Un aumento de área no fecha ni identifica automáticamente nuevos módulos.
    """

    filas = []
    for identificador, grupo in serie.groupby("id_instalacion"):
        ordenadas = list(grupo.sort_values("anio").itertuples())
        for antes, despues in zip(ordenadas, ordenadas[1:]):
            comparable = antes.superficie_comparable and despues.superficie_comparable
            diferencia = (
                despues.superficie_horizontal_anotada_m2
                - antes.superficie_horizontal_anotada_m2
                if comparable
                else None
            )
            filas.append(
                {
                    "id_instalacion": identificador,
                    "anio_desde": antes.anio,
                    "anio_hasta": despues.anio,
                    "comparable": comparable,
                    "cambio_horizontal_m2": diferencia,
                    "instalacion_confirmada": antes.instalacion_confirmada
                    and despues.instalacion_confirmada,
                }
            )
    return pd.DataFrame(filas)
