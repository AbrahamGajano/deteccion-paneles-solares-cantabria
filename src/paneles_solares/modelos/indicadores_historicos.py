"""Exporta un avance independiente desde una instantánea de solo lectura del etiquetado."""

import argparse
import html
import json
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import from_wkb, union_all

from paneles_solares.datos.seguimiento import (
    BASE,
    derivar_historia,
    guardar_csv,
    guardar_json,
)
from paneles_solares.rutas import (
    INDICADORES_SEGUIMIENTO,
)

SALIDA = INDICADORES_SEGUIMIENTO
NOTA_ALCANCE = (
    "Avance del inventario estudiado hasta 2023, con revisión todavía en curso. "
    "Los recuentos son de emplazamientos; las agrupaciones pendientes no son instalaciones "
    "administrativas confirmadas. No representa un censo completo de Cantabria ni incluye "
    "sistemáticamente instalaciones retiradas antes de 2023."
)


def tomar_instantanea(base: Path) -> sqlite3.Connection:
    """Copia SQLite en memoria por bloques breves; nunca abre el origen para escribir.

    El visor puede seguir guardando entre bloques. Si la base permanece ocupada, se
    abandona la exportación en lugar de reconfigurar o bloquear el etiquetado.

    Args:
        base: Archivo SQLite de seguimiento que se consulta en modo de lectura.

    Returns:
        sqlite3.Connection: Conexión SQLite en memoria, consistente y de solo lectura.
    """

    inicio = time.monotonic()
    copia = sqlite3.connect(":memory:")

    def comprobar(_estado, _pendientes, _total):
        """Detiene la copia si supera el tiempo máximo de espera.

        Args:
            _estado: Estado de la copia comunicado por SQLite; no se utiliza.
            _pendientes: Páginas pendientes comunicadas por SQLite; no se utilizan.
            _total: Páginas totales comunicadas por SQLite; no se utilizan.

        Raises:
            TimeoutError: La copia de SQLite supera diez segundos de espera.
        """

        if time.monotonic() - inicio > 10:
            raise TimeoutError(
                "La base está ocupada; vuelve a generar el avance más tarde."
            )

    with ExitStack() as recursos:
        recursos.callback(copia.close)
        with closing(
            sqlite3.connect(base.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
        ) as origen:
            origen.backup(copia, pages=64, progress=comprobar, sleep=0.02)
        copia.row_factory = sqlite3.Row
        copia.execute("PRAGMA query_only=ON")
        recursos.pop_all()
        return copia


def calcular_indicadores(db: sqlite3.Connection) -> tuple[list, list, list, dict, dict]:
    """Deriva periodos, cobertura municipal y puntos para un futuro mapa sin inferencia.

    Args:
        db: Instantánea SQLite de solo lectura.

    Returns:
        tuple[list, list, list, dict, dict]: Emplazamientos, indicadores municipales,
            periodos, resumen y puntos GeoJSON.

    Raises:
        ValueError: Las campañas de la base no son únicas o no incluyen la referencia
            2023.
    """

    meta = {
        r["clave"]: json.loads(r["valor"])
        for r in db.execute("SELECT * FROM metadatos")
    }
    anios = sorted(c["anio"] for c in meta.get("campanias", []))
    if len(anios) < 2 or len(set(anios)) != len(anios) or max(anios) != 2023:
        raise ValueError(
            "Se necesitan campañas únicas con 2023 como última referencia."
        )
    historicos = [a for a in anios if a < 2023]
    nombres_periodos = {f"hasta_{anios[0]}": f"Ya presente en {anios[0]} o antes"}
    nombres_periodos.update(
        {f"{a}_{b}": f"Entre vuelos de {a} y {b}" for a, b in pairwise(anios)}
    )
    observaciones = defaultdict(list)
    for fila in db.execute("SELECT * FROM observaciones"):
        if fila["anio"] in anios:
            observaciones[fila["emplazamiento"]].append(dict(fila))
    geometrias = defaultdict(list)
    for fila in db.execute(
        "SELECT m.emplazamiento,c.geometria FROM miembros m JOIN componentes c "
        "ON c.id=m.componente WHERE c.anio=2023"
    ):
        geometrias[fila["emplazamiento"]].append(from_wkb(fila["geometria"]))
    convertir = Transformer.from_crs(25830, 4326, always_xy=True)
    casos, puntos = [], []
    for s in db.execute(
        "SELECT * FROM emplazamientos WHERE activo=1 ORDER BY municipio,id"
    ):
        obs = observaciones[s["id"]]
        por_anio = {o["anio"]: o for o in obs}
        historia = derivar_historia(obs, anios)
        provisional = (
            s["origen"] in ("piloto", "inventario_2023") and not geometrias[s["id"]]
        )
        periodo = "sin_resolver"
        if historia["estado_historia"] == "presente_desde_primera_campania":
            periodo = f"hasta_{anios[0]}"
        elif historia["estado_historia"] == "intervalo_observado":
            periodo = (
                f"{historia['aparicion_despues_de']}_{historia['aparicion_hasta']}"
            )
        completas = all(a in por_anio for a in anios)
        util = (
            not provisional
            and completas
            and historia["referencia_2023"] == "presente"
            and periodo in nombres_periodos
        )
        fila = {
            "id": s["id"],
            "municipio": s["municipio"],
            "codigo_municipal": s["codigo_municipal"],
            "soporte": s["soporte"],
            "origen": s["origen"],
            "zona_provisional": provisional,
            "agrupacion": s["agrupacion"],
            "revision_completa": completas,
            "historicos_guardados": sum(a in por_anio for a in historicos),
            **{
                f"decision_{a}": por_anio.get(a, {}).get("decision", "pendiente")
                for a in anios
            },
            "primera_presencia": historia["primera_presencia"],
            "aparicion_despues_de": historia["aparicion_despues_de"],
            "aparicion_hasta": historia["aparicion_hasta"],
            "estado_historia": historia["estado_historia"],
            "periodo_aparicion": periodo,
            "incluir_en_despliegue": util,
            "alcance_parcial": any(
                o["alcance"] != "emplazamiento_completo" for o in obs
            ),
            "cambio_anotado": any(
                t
                in (
                    "ampliacion",
                    "reduccion",
                    "reinstalacion",
                    "distribucion_diferente",
                )
                for o in obs
                for t in json.loads(o["etiquetas"] or "[]")
            ),
            "superficie_fv_detectada_2023_m2": None,
            "longitud": None,
            "latitud": None,
        }
        if geometrias[s["id"]]:
            geometria = union_all(geometrias[s["id"]])
            fila["superficie_fv_detectada_2023_m2"] = round(geometria.area, 3)
            punto = geometria.representative_point()
            fila["longitud"], fila["latitud"] = convertir.transform(punto.x, punto.y)
            puntos.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [fila["longitud"], fila["latitud"]],
                    },
                    "properties": dict(fila),
                }
            )
        casos.append(fila)

    grupos = defaultdict(list)
    for c in casos:
        grupos[(c["codigo_municipal"], c["municipio"])].append(c)
    municipal, periodos = [], []
    for (codigo, municipio), locales in sorted(grupos.items()):
        principales = [c for c in locales if not c["zona_provisional"]]
        esperados = len(principales) * len(historicos)
        guardados = sum(c["historicos_guardados"] for c in principales)
        presentes = [c for c in principales if c["decision_2023"] == "presente"]
        fechados = [c for c in principales if c["incluir_en_despliegue"]]
        municipal.append(
            {
                "codigo_municipal": codigo,
                "municipio": municipio,
                "candidatos_localizados": len(principales),
                "zonas_provisionales_apartadas": len(locales) - len(principales),
                "historicos_guardados": guardados,
                "historicos_esperados": esperados,
                "revision_historica_pct": round(100 * guardados / esperados, 2)
                if esperados
                else None,
                "referencias_2023_presentes": len(presentes),
                "emplazamientos_con_periodo_util": len(fechados),
                "referencias_presentes_sin_periodo_util": len(presentes)
                - len(fechados),
                "agrupaciones_confirmadas": sum(
                    c["agrupacion"] == "confirmada" for c in principales
                ),
                "agrupaciones_pendientes": sum(
                    c["agrupacion"] != "confirmada" for c in principales
                ),
            }
        )
        for periodo, etiqueta in nombres_periodos.items():
            seleccion = [c for c in fechados if c["periodo_aparicion"] == periodo]
            intervalo = None if periodo.startswith("hasta_") else periodo.split("_")
            duracion = int(intervalo[1]) - int(intervalo[0]) if intervalo else None
            periodos.append(
                {
                    "codigo_municipal": codigo,
                    "municipio": municipio,
                    "periodo": periodo,
                    "descripcion": etiqueta,
                    "emplazamientos_observados": len(seleccion),
                    "duracion_nominal_anios": duracion,
                    "ritmo_medio_observado_por_anio": round(
                        len(seleccion) / duracion, 3
                    )
                    if duracion
                    else None,
                    "agrupaciones_confirmadas": sum(
                        c["agrupacion"] == "confirmada" for c in seleccion
                    ),
                    "agrupaciones_pendientes": sum(
                        c["agrupacion"] != "confirmada" for c in seleccion
                    ),
                    "revision_historica_pct": municipal[-1]["revision_historica_pct"],
                }
            )
    principales = [c for c in casos if not c["zona_provisional"]]
    guardadas = sum(c["historicos_guardados"] for c in principales)
    esperadas = len(principales) * len(historicos)
    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "alcance": NOTA_ALCANCE,
        "campanias": anios,
        "candidatos_localizados": len(principales),
        "zonas_provisionales_apartadas": len(casos) - len(principales),
        "historicos_guardados": guardadas,
        "historicos_esperados": esperadas,
        "revision_historica_pct": round(100 * guardadas / esperadas, 2)
        if esperadas
        else None,
        "casos_revisados_completos": sum(c["revision_completa"] for c in principales),
        "emplazamientos_con_periodo_util": sum(
            c["incluir_en_despliegue"] for c in principales
        ),
        "periodos": {
            p: sum(
                c["incluir_en_despliegue"] and c["periodo_aparicion"] == p
                for c in casos
            )
            for p in nombres_periodos
        },
        "estados_historia": dict(Counter(c["estado_historia"] for c in principales)),
        "agrupaciones_confirmadas": sum(
            c["agrupacion"] == "confirmada" for c in principales
        ),
        "agrupaciones_pendientes": sum(
            c["agrupacion"] != "confirmada" for c in principales
        ),
        "observaciones_en_instantanea": db.execute(
            "SELECT COUNT(*) FROM observaciones"
        ).fetchone()[0],
        "ultima_respuesta_utc": db.execute(
            "SELECT MAX(fecha_utc) FROM observaciones"
        ).fetchone()[0],
        "ultima_accion": db.execute("SELECT MAX(id) FROM acciones").fetchone()[0],
        "potencia_por_epoca": "Pendiente de definir y validar coeficientes; no calculada en este avance.",
    }
    return (
        casos,
        municipal,
        periodos,
        resumen,
        {"type": "FeatureCollection", "features": puntos},
    )


def informe_html(municipios: list[dict], resumen: dict) -> str:
    """Genera un informe autocontenido de progreso, sin servicios web ni scripts remotos.

    Args:
        municipios: Indicadores municipales del avance.
        resumen: Totales, periodos y cobertura de la instantánea.

    Returns:
        str: Documento HTML completo con progreso, periodos y enlaces a los datos.
    """

    barras = []
    maximo = max(resumen["periodos"].values(), default=1) or 1
    for periodo, n in resumen["periodos"].items():
        etiqueta = (
            f"En {periodo[6:]} o antes"
            if periodo.startswith("hasta_")
            else periodo.replace("_", " → ")
        )
        barras.append(
            f'<div class="barra"><span>{etiqueta}</span><meter min="0" '
            f'max="{maximo}" value="{n}">{n}</meter><b>{n}</b></div>'
        )
    filas = []
    for m in municipios:
        revision = (
            f"{m['revision_historica_pct']:.1f} %"
            if m["revision_historica_pct"] is not None
            else "Sin objetivos localizados"
        )
        filas.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    m["municipio"] or "Sin municipio",
                    m["candidatos_localizados"],
                    revision,
                    m["emplazamientos_con_periodo_util"],
                    m["agrupaciones_pendientes"],
                )
            )
            + "</tr>"
        )
    return f"""<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Avance histórico fotovoltaico</title>
<style>body{{font:16px system-ui;color:#18323b;background:#f3f6f5;max-width:1100px;margin:auto;padding:30px}}
section{{background:white;padding:22px;border-radius:12px;margin:18px 0}}h1,h2{{color:#155b55}}
.cifras{{display:flex;flex-wrap:wrap;gap:28px}}.cifras b{{display:block;font-size:30px}}a{{color:#146558}}
.barra{{display:grid;grid-template-columns:180px 1fr 65px;gap:14px;align-items:center;margin:14px 0}}meter{{width:100%;height:26px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde6e1}}
.tabla{{overflow-x:auto}}.nota{{color:#4a6169}}@media(max-width:600px){{body{{padding:12px}}.barra{{grid-template-columns:125px 1fr 40px}}}}</style>
<h1>Avance de la revisión histórica fotovoltaica</h1><p>Instantánea: {html.escape(resumen["generado_utc"])} (UTC).</p>
<section class="cifras"><div><b>{resumen["historicos_guardados"]:,}</b>respuestas históricas guardadas</div>
<div><b>{resumen["casos_revisados_completos"]:,}</b>emplazamientos con las cuatro campañas revisadas</div>
<div><b>{resumen["emplazamientos_con_periodo_util"]:,}</b>emplazamientos presentes en 2023 con periodo utilizable</div></section>
<section><h2>Aparición observada por periodo</h2>{"".join(barras)}
<p class="nota">Solo incluye casos con las cuatro respuestas guardadas, presencia en 2023 e intervalo resuelto.
La barra inicial incluye los ya presentes en el primer vuelo; no permite calcular un ritmo anual anterior.
Los años son campañas de vuelo, no fechas exactas de instalación.</p></section>
<section><h2>Avance por municipio</h2><div class="tabla"><table><thead><tr><th>Municipio</th><th>Candidatos localizados</th>
<th>Revisión histórica</th><th>Con periodo utilizable</th><th>Agrupaciones pendientes</th></tr></thead><tbody>{"".join(filas)}</tbody></table></div>
<p class="nota">La cobertura se refiere a los candidatos localizados de esta instantánea. Puede cambiar con la preparación de más recortes.
Las diferencias entre municipios siguen siendo provisionales mientras continúa la revisión.</p></section>
<section><h2>Datos descargables</h2><p><a href="emplazamientos_historicos.csv">CSV por emplazamiento</a> ·
<a href="indicadores_municipales.csv">CSV municipal</a> · <a href="despliegue_por_periodo.csv">CSV de ritmos por periodo</a> ·
<a href="emplazamientos_mapa.geojson">Puntos GeoJSON para el mapa</a></p>
<p>{html.escape(NOTA_ALCANCE)}</p><p>La superficie exportada es la unión de las predicciones originales de 2023: no es una medición manual validada.
No se atribuye esa superficie a años anteriores ni se estima todavía potencia por época.</p>
<p>Este avance no exige confirmar agrupaciones para mostrar el trabajo realizado; conserva su estado pendiente.
No cambia ninguna respuesta ni el recorrido del visor. Se regenera por separado al avanzar el etiquetado.</p></section></html>"""


def generar_avance(base: Path = BASE, salida: Path | None = None) -> dict:
    """Publica archivos nuevos en una carpeta separada; no toca las salidas del visor.

    Args:
        base: Archivo SQLite de seguimiento que se consulta en modo de lectura.
        salida: Carpeta nueva para el avance; None utiliza el nombre fechado habitual.

    Returns:
        dict: Resumen de la instantánea y carpeta donde se ha publicado.

    Raises:
        ValueError: La salida queda dentro de la carpeta de los datos de etiquetado.
        FileExistsError: Si la carpeta de salida ya existe.
    """

    with closing(tomar_instantanea(base)) as copia:
        casos, municipios, periodos, resumen, puntos = calcular_indicadores(copia)
    salida = salida or SALIDA / datetime.now(timezone.utc).strftime(
        "avance_%Y%m%d_%H%M%S_%f"
    )
    salida = salida.resolve()
    if salida.is_relative_to(base.resolve().parent):
        raise ValueError(
            "El avance debe guardarse fuera de la carpeta de la base de etiquetado."
        )
    salida.mkdir(parents=True, exist_ok=False)
    resumen["base_origen"] = str(base.resolve())
    resumen["carpeta_salida"] = str(salida)
    columnas_casos = [
        "id",
        "municipio",
        "codigo_municipal",
        "soporte",
        "origen",
        "zona_provisional",
        "agrupacion",
        "revision_completa",
        "historicos_guardados",
        *[f"decision_{a}" for a in resumen["campanias"]],
        "primera_presencia",
        "aparicion_despues_de",
        "aparicion_hasta",
        "estado_historia",
        "periodo_aparicion",
        "incluir_en_despliegue",
        "alcance_parcial",
        "cambio_anotado",
        "superficie_fv_detectada_2023_m2",
        "longitud",
        "latitud",
    ]
    columnas_municipios = [
        "codigo_municipal",
        "municipio",
        "candidatos_localizados",
        "zonas_provisionales_apartadas",
        "historicos_guardados",
        "historicos_esperados",
        "revision_historica_pct",
        "referencias_2023_presentes",
        "emplazamientos_con_periodo_util",
        "referencias_presentes_sin_periodo_util",
        "agrupaciones_confirmadas",
        "agrupaciones_pendientes",
    ]
    columnas_periodos = [
        "codigo_municipal",
        "municipio",
        "periodo",
        "descripcion",
        "emplazamientos_observados",
        "duracion_nominal_anios",
        "ritmo_medio_observado_por_anio",
        "agrupaciones_confirmadas",
        "agrupaciones_pendientes",
        "revision_historica_pct",
    ]
    guardar_csv(
        pd.DataFrame(casos, columns=columnas_casos),
        salida / "emplazamientos_historicos.csv",
    )
    guardar_csv(
        pd.DataFrame(municipios, columns=columnas_municipios),
        salida / "indicadores_municipales.csv",
    )
    guardar_csv(
        pd.DataFrame(periodos, columns=columnas_periodos),
        salida / "despliegue_por_periodo.csv",
    )
    guardar_json(puntos, salida / "emplazamientos_mapa.geojson")
    guardar_json(resumen, salida / "resumen.json")
    (salida / "avance_historico.html").write_text(
        informe_html(municipios, resumen), encoding="utf-8"
    )
    return resumen


def main() -> None:
    """Genera un avance puntual sin abrir el visor ni preparar imágenes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--salida", type=Path, help="Carpeta nueva para esta instantánea"
    )
    args = parser.parse_args()
    resumen = generar_avance(args.base, args.salida)
    print(f"Avance: {resumen['carpeta_salida']}")
    print(
        f"Históricos guardados: {resumen['historicos_guardados']}; con periodo "
        f"utilizable: {resumen['emplazamientos_con_periodo_util']}"
    )


if __name__ == "__main__":
    main()
