"""Recupera predicciones completas y prepara recortes históricos por lotes."""

import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import torch
from PIL import Image
from rasterio.windows import Window
from shapely import from_wkb, union_all
from shapely.geometry import Polygon, box

from paneles_solares.datos.seguimiento import (
    URL_WCS,
    ahora,
    geometria_mascara,
    identificador,
    leer_csv,
    sustituir_archivo,
    transformacion,
)
from paneles_solares.etiquetado.preparar_validacion import (
    comprobar_wcs,
    extraer_ventana,
    extraer_wcs,
)
from paneles_solares.modelos.evaluar_unet import predecir
from paneles_solares.modelos.unet import cargar_modelo, normalizar_rgb
from paneles_solares.rutas import INDICADORES_UNET, PESOS_UNET, ruta_proyecto


def guardar_png(imagen: Image.Image, ruta: Path):
    """Publica un PNG verificado, sin TIFF ni temporales permanentes."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".png.part")
    try:
        imagen.save(temporal, format="PNG")
        with Image.open(temporal) as comprobacion:
            comprobacion.verify()
        sustituir_archivo(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)


def importar_piloto(almacen, config: dict):
    """Importa una vez el piloto sin modificar sus archivos ni decisiones.

    Args:
        almacen: Base única de seguimiento.
        config: Campañas independientes configuradas.
    """
    anteriores = {c["anio"]: c for c in almacen.meta("campanias", [])}
    actuales = {c["anio"]: c for c in config["campanias"]}
    if anteriores and any(c != actuales.get(a) for a, c in anteriores.items()):
        raise ValueError(
            "Conserva el origen de las campañas existentes; puedes añadir nuevas sin modificar las observadas"
        )
    almacen.poner_meta("campanias", config["campanias"])
    if almacen.meta("piloto_importado"):
        return
    carpeta = almacen.ruta.parent
    muestra = leer_csv(carpeta / "muestra.csv")
    if muestra.empty:
        almacen.poner_meta("piloto_importado", {"candidatas": 0, "decisiones": 0})
        return
    geometrias = gpd.read_file(carpeta / "muestra.gpkg").set_index("id_instalacion")
    revisiones = leer_csv(carpeta / "revisiones.csv")
    with almacen.db:
        for fila in muestra.to_dict("records"):
            sitio, ventana = fila["id_instalacion"], fila["id_ventana"]
            almacen.db.execute(
                "INSERT OR IGNORE INTO ventanas VALUES (?,?,?,?,?,?)",
                (ventana, *(fila[c] for c in ("xmin", "ymin", "xmax", "ymax")), "piloto"),
            )
            etiquetas = ["agrupacion_pendiente"]
            if fila["fraccion_edificio_en_ventana"] < 0.99:
                etiquetas.append("borde_recorte")
            almacen.db.execute(
                "INSERT OR IGNORE INTO emplazamientos(id,ventana,geometria,municipio,codigo_municipal,edificios,origen,etiquetas) VALUES (?,?,?,?,?,?,?,?)",
                (
                    sitio,
                    ventana,
                    geometrias.loc[sitio].geometry.wkb,
                    fila["municipio"],
                    fila["codigo_municipal"],
                    json.dumps([str(fila["id_edificio"])]),
                    "piloto",
                    json.dumps(etiquetas),
                ),
            )
        for campania in config["campanias"]:
            anio = campania["anio"]
            metadatos = leer_csv(carpeta / f"ventanas_{anio}.csv")
            por_id = {r["id_instalacion"]: r for r in metadatos.to_dict("records")}
            for ventana in muestra.id_ventana.unique():
                imagen = carpeta / "images" / str(anio) / f"{ventana}.png"
                prediccion = carpeta / "predicciones" / str(anio) / f"{ventana}.png"
                if not imagen.exists():
                    continue
                ruta_firma = carpeta / f"modelo_{anio}.json"
                firma = (
                    json.loads(ruta_firma.read_text(encoding="utf-8")).get("sha256", "")
                    if ruta_firma.exists()
                    else ""
                )
                almacen.db.execute(
                    "INSERT OR IGNORE INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                    (
                        ventana,
                        anio,
                        str(imagen),
                        str(prediccion) if anio != 2023 else "",
                        por_id.get(ventana, {}).get("fuente", "piloto"),
                        "",
                        campania["resolucion_m"],
                        firma if anio != 2023 else "",
                    ),
                )
        filas_muestra = muestra.set_index("id_instalacion")
        for fila in revisiones.to_dict("records"):
            sitio = fila["id_instalacion"]
            info = filas_muestra.loc[sitio]
            alcance = fila.get("alcance") or (
                "zona_visible"
                if info.fraccion_edificio_en_ventana < 0.99
                else "emplazamiento_completo"
            )
            if info.get("agrupacion_ambigua", False):
                alcance = "agrupacion_cubiertas"
            if alcance == "cubierta_completa":
                alcance = "emplazamiento_completo"
            almacen._escribir_observacion(
                {
                    "emplazamiento": sitio,
                    "anio": fila["anio"],
                    "decision": "no_evaluable"
                    if fila["decision"] == "imagen_invalida"
                    else fila["decision"],
                    "alcance": alcance,
                    "etiquetas": "[]",
                    "notas": "Importada del piloto",
                    "segundos": fila["segundos"],
                    "fecha_utc": fila["fecha_utc"],
                    "original": json.dumps(fila, ensure_ascii=False),
                }
            )
        datos = {
            "candidatas": len(muestra),
            "decisiones": len(revisiones),
            "sha256_revisiones": hashlib.sha256(
                (carpeta / "revisiones.csv").read_bytes()
            ).hexdigest(),
        }
        almacen.db.execute(
            "INSERT OR REPLACE INTO metadatos VALUES ('piloto_importado',?)", (json.dumps(datos),)
        )


class GeografiaSeguimiento:
    """Catastro describe la detección; su relación no confirma ni excluye paneles."""

    def __init__(self, proximidad: float | None = None):
        self.edificios = gpd.read_file(
            ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg"), fid_as_index=True
        )
        self.edificios = self.edificios[~self.edificios.geometry.to_wkb().duplicated()]
        self.municipios = gpd.read_file(ruta_proyecto("data/geografia/municipios_cantabria.gpkg"))
        self.proximidad = proximidad

    def describir(self, geometria) -> dict:
        """Mide intersección y distancia sin recortar ni desplazar la predicción.

        Args:
            geometria: Componente original en EPSG:25830.

        Returns:
            Municipio, relación, fracción, distancia y asociaciones candidatas.
        """
        posiciones = self.edificios.sindex.query(geometria, predicate="intersects")
        coincidentes = self.edificios.iloc[posiciones]
        fraccion = (
            geometria.intersection(union_all(coincidentes.geometry)).area / geometria.area
            if len(posiciones)
            else 0
        )
        _, distancias = self.edificios.sindex.nearest(
            geometria, return_distance=True, return_all=True
        )
        distancia = float(distancias.min()) if len(distancias) else None
        edificios = [str(i) for i in coincidentes.index]
        cubiertas = []
        for idx, fila in coincidentes.iterrows():
            partes = (
                list(fila.geometry.geoms) if hasattr(fila.geometry, "geoms") else [fila.geometry]
            )
            for n, parte in enumerate(sorted(partes, key=lambda g: g.bounds), 1):
                if parte.intersection(geometria).area > 0:
                    cubiertas.append(f"catastro_{idx}_parte_{n}")
        if fraccion >= 0.8:
            relacion = "interseccion_clara" if len(edificios) == 1 else "ambigua"
        elif fraccion > 0:
            relacion = "parcial_desplazado" if len(edificios) == 1 else "ambigua"
        elif self.proximidad is not None and distancia is not None and distancia <= self.proximidad:
            relacion = "proximo"
        else:
            relacion = "exterior" if distancia is not None else "sin_asociacion"
        municipales = self.municipios.sindex.query(
            geometria.representative_point(), predicate="intersects"
        )
        municipio = self.municipios.iloc[municipales[0]] if len(municipales) else None
        return {
            "relacion": relacion,
            "edificios": edificios,
            "cubiertas": cubiertas,
            "fraccion": float(fraccion),
            "distancia": distancia,
            "municipio": municipio.municipio if municipio is not None else "Sin municipio resuelto",
            "codigo_municipal": str(municipio.codigo) if municipio is not None else "",
        }


def registrar_componentes(
    almacen,
    ventana: dict,
    anio: int,
    probabilidades: np.ndarray,
    firma: str,
    geografia,
    origen: str,
    corte_m: float = 0.6,
):
    """Guarda todos los componentes y propone agrupaciones sin filtro catastral.

    Args:
        almacen: Base persistente.
        ventana: Encuadre original.
        anio: Campaña de la predicción.
        probabilidades: Señal completa sin área mínima.
        firma: SHA256 del modelo congelado.
        geografia: Ayuda catastral y municipal.
        origen: Recorte recuperado o regional.
        corte_m: Máxima separación para pequeños cortes, independiente de Catastro.
    """
    componentes = geometria_mascara(
        probabilidades >= 0.5,
        transformacion(ventana, probabilidades.shape[1], probabilidades.shape[0]),
    )
    limite = box(*(ventana[c] for c in ("xmin", "ymin", "xmax", "ymax")))
    for geometria in componentes:
        cid = identificador(
            "componente", f"{ventana['id']}|{anio}|{firma}|".encode() + geometria.wkb
        )
        if almacen.db.execute("SELECT 1 FROM componentes WHERE id=?", (cid,)).fetchone():
            continue
        info = geografia.describir(geometria)
        borde = geometria.distance(limite.boundary) < 0.16
        candidatos = []
        if anio == 2023:
            revisados = {
                o["emplazamiento"]
                for o in almacen.filas("SELECT DISTINCT emplazamiento FROM observaciones")
            }
            for sitio in almacen.filas("SELECT * FROM emplazamientos WHERE activo=1"):
                grupo = from_wkb(sitio["geometria"])
                distancia = geometria.distance(grupo)
                legado = sitio["origen"] in ("piloto", "inventario_2023")
                if (
                    not legado
                    and sitio["id"] in revisados
                    and geometria.difference(grupo).area > 1e-6
                ):
                    continue
                if (legado and geometria.intersection(grupo).area > 0) or (
                    not legado and distancia <= corte_m
                ):
                    candidatos.append((sitio, geometria.intersection(grupo).area, distancia))
        # La geografía y la búsqueda de vecinos no retienen el bloqueo de escritura.
        # Una transacción breve publica cada componente y sus asociaciones juntos.
        with almacen.db:
            _guardar_componente(
                almacen, ventana, anio, geometria, cid, info, borde, firma, origen, candidatos
            )


def _guardar_componente(
    almacen, ventana, anio, geometria, cid, info, borde, firma, origen, candidatos
):
    """Publica un componente sin cálculos costosos dentro de la transacción."""
    if almacen.db.execute("SELECT 1 FROM componentes WHERE id=?", (cid,)).fetchone():
        return
    almacen.db.execute(
        "INSERT INTO componentes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            cid,
            ventana["id"],
            anio,
            geometria.wkb,
            geometria.area,
            int(borde),
            info["relacion"],
            json.dumps(info["edificios"]),
            json.dumps(info["cubiertas"]),
            info["fraccion"],
            info["distancia"],
            info["municipio"],
            info["codigo_municipal"],
            firma,
            origen,
        ),
    )
    if anio != 2023:
        return
    actuales = []
    for candidato, solape, distancia in candidatos:
        fila = almacen.filas(
            "SELECT * FROM emplazamientos WHERE id=? AND activo=1", (candidato["id"],)
        )
        if not fila or fila[0]["geometria"] != candidato["geometria"]:
            continue
        sitio = fila[0]
        revisado = almacen.db.execute(
            "SELECT 1 FROM observaciones WHERE emplazamiento=?", (sitio["id"],)
        ).fetchone()
        if (
            sitio["origen"] not in ("piloto", "inventario_2023")
            and revisado
            and geometria.difference(from_wkb(sitio["geometria"])).area > 1e-6
        ):
            continue
        actuales.append((sitio, solape, distancia))
    candidatos = actuales
    if candidatos:
        elegido = max(candidatos, key=lambda p: (p[1], -p[2]))[0]
        sid = elegido["id"]
        if elegido["origen"] not in ("piloto", "inventario_2023"):
            grupo = union_all([from_wkb(elegido["geometria"]), geometria])
            almacen.db.execute("UPDATE emplazamientos SET geometria=? WHERE id=?", (grupo.wkb, sid))
        etiquetas = set(json.loads(elegido["etiquetas"]))
        if len(candidatos) > 1:
            etiquetas.add("agrupacion_pendiente")
        if borde:
            etiquetas.add("borde_recorte")
        almacen.db.execute(
            "UPDATE emplazamientos SET etiquetas=? WHERE id=?",
            (json.dumps(sorted(etiquetas)), sid),
        )
    else:
        sid = identificador("emplazamiento", cid.encode())
        etiquetas = ["agrupacion_pendiente"] + (["borde_recorte"] if borde else [])
        almacen.db.execute(
            "INSERT INTO emplazamientos(id,ventana,geometria,municipio,codigo_municipal,relacion,edificios,cubiertas,origen,etiquetas) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                ventana["id"],
                geometria.wkb,
                info["municipio"],
                info["codigo_municipal"],
                info["relacion"],
                json.dumps(info["edificios"]),
                json.dumps(info["cubiertas"]),
                origen,
                json.dumps(etiquetas),
            ),
        )
    almacen.db.execute("INSERT INTO miembros VALUES (?,?)", (cid, sid))
    corregido = almacen.db.execute(
        "SELECT 1 FROM acciones WHERE objetivo=? AND tipo='sitio'", (sid,)
    ).fetchone()
    if not corregido:
        asociados = almacen.filas(
            "SELECT c.* FROM componentes c JOIN miembros m ON m.componente=c.id WHERE m.emplazamiento=?",
            (sid,),
        )
        edificios = sorted({e for c in asociados for e in json.loads(c["edificios"])})
        cubiertas = sorted({e for c in asociados for e in json.loads(c["cubiertas"])})
        relaciones = {c["relacion"] for c in asociados}
        relacion = next(iter(relaciones)) if len(relaciones) == 1 else "ambigua"
        almacen.db.execute(
            "UPDATE emplazamientos SET relacion=?,edificios=?,cubiertas=? WHERE id=?",
            (relacion, json.dumps(edificios), json.dumps(cubiertas), sid),
        )


def preparar_lote(
    almacen,
    config: dict,
    lote: int,
    sitios: list[str] | None = None,
    recuperar: bool = False,
    progreso=None,
    al_completar=None,
    solo_localizados: bool = False,
    detener=None,
):
    """Prepara todas las campañas del lote, incluso tras ausencias o dudas.

    Args:
        almacen: Base con progreso importado.
        config: Campañas oficiales.
        lote: Máximo de ventanas compartidas.
        sitios: Selección opcional para preparar casos especiales.
        recuperar: Solo 2023 de los recortes existentes para la auditoría inicial.
        progreso: Recibe el espacio estimado y el avance para mostrarlos en el visor.
        al_completar: Notifica cada ventana disponible sin esperar al final del lote.
        solo_localizados: Evita históricos de ubicaciones sin objetivo localizado en 2023.
        detener: Evento opcional para detener la precarga al cerrar el visor.
    """
    importar_piloto(almacen, config)
    firma = hashlib.sha256(PESOS_UNET.read_bytes()).hexdigest()
    previos = almacen.filas("SELECT DISTINCT modelo FROM recursos WHERE prediccion<>''")
    if any(f["modelo"] and f["modelo"] != firma for f in previos):
        raise ValueError("El modelo difiere del congelado del inventario")
    campanias = (
        [c for c in config["campanias"] if c["anio"] == 2023] if recuperar else config["campanias"]
    )
    campanias = sorted(campanias, key=lambda c: c["anio"], reverse=True)
    candidatas = almacen.filas(
        "SELECT DISTINCT v.* FROM ventanas v JOIN emplazamientos e ON e.ventana=v.id WHERE e.activo=1 ORDER BY v.id"
    )
    if sitios:
        ids = {
            s["ventana"]
            for s in almacen.filas(
                "SELECT ventana FROM emplazamientos WHERE id IN ("
                + ",".join("?" for _ in sitios)
                + ")",
                tuple(sitios),
            )
        }
        candidatas = [v for v in candidatas if v["id"] in ids]
    elif not recuperar:
        historicos = [c["anio"] for c in config["campanias"] if c["anio"] < 2023]
        pendientes = {
            s["ventana"]
            for s in almacen.filas("SELECT id,ventana FROM emplazamientos WHERE activo=1")
            if any(
                not almacen.db.execute(
                    "SELECT 1 FROM observaciones WHERE emplazamiento=? AND anio=?", (s["id"], a)
                ).fetchone()
                for a in historicos
            )
        }
        candidatas = [v for v in candidatas if v["id"] in pendientes]
    ventanas = []
    for ventana in candidatas:
        if recuperar and ventana["origen"] != "piloto":
            continue
        recursos = {
            r["anio"]: r
            for r in almacen.filas("SELECT * FROM recursos WHERE ventana=?", (ventana["id"],))
        }
        if any(
            not recursos.get(c["anio"], {}).get("prediccion")
            or not Path(recursos[c["anio"]]["prediccion"]).exists()
            or not recursos[c["anio"]]["imagen"]
            or not Path(recursos[c["anio"]]["imagen"]).exists()
            for c in campanias
        ):
            ventanas.append(ventana)
        if len(ventanas) >= lote:
            break
    faltantes = [
        (v, c)
        for v in ventanas
        for c in campanias
        if not any(
            r["imagen"] and Path(r["imagen"]).exists()
            for r in almacen.filas(
                "SELECT imagen FROM recursos WHERE ventana=? AND anio=?", (v["id"], c["anio"])
            )
        )
    ]
    estimados = sum(
        round((v["xmax"] - v["xmin"]) / c["resolucion_m"]) ** 2 * 3 for v, c in faltantes
    )
    libre = shutil.disk_usage(almacen.ruta.parent).free
    estimacion = (
        f"Lote: {len(ventanas)} ventanas; {len(faltantes)} recortes nuevos; "
        f"RGB ≈ {estimados / 1024**2:.1f} MB; libres {libre / 1024**3:.1f} GB."
    )
    print(estimacion, flush=True)
    if progreso:
        progreso(estimacion)
    if estimados * 2 > libre:
        raise OSError("Reduce el lote: falta espacio para imágenes y predicciones")
    modelo = None
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geografia = GeografiaSeguimiento(almacen.meta("proximidad_m"))
    comprobadas = set()
    for numero, ventana in enumerate(ventanas, 1):
        for campania in campanias:
            if detener is not None and detener.is_set():
                return
            anio = campania["anio"]
            if (
                solo_localizados
                and anio < 2023
                and not almacen.db.execute(
                    """SELECT 1 FROM emplazamientos e WHERE e.ventana=? AND e.activo=1
                AND (e.origen NOT IN ('piloto','inventario_2023') OR EXISTS
                (SELECT 1 FROM miembros m JOIN componentes c ON c.id=m.componente
                 WHERE m.emplazamiento=e.id AND c.anio=2023)) LIMIT 1""",
                    (ventana["id"],),
                ).fetchone()
            ):
                continue
            recursos = almacen.filas(
                "SELECT * FROM recursos WHERE ventana=? AND anio=?", (ventana["id"], anio)
            )
            recurso = recursos[0] if recursos else {}
            imagen = Path(
                recurso.get("imagen")
                or almacen.ruta.parent / "images" / str(anio) / f"{ventana['id']}.png"
            )
            prediccion_existente = (
                recurso.get("prediccion") and Path(recurso["prediccion"]).exists()
            )
            if prediccion_existente and imagen.exists():
                continue
            if progreso:
                progreso(f"Ventana {numero}/{len(ventanas)} · campaña {anio}")
            fuente = recurso.get("fuente", "")
            try:
                if not imagen.exists():
                    origen = {
                        "tipo": "wcs",
                        "url": URL_WCS,
                        "bandas": [1, 2, 3],
                        **{c: campania[c] for c in ("cobertura", "etiqueta", "divisor_rgb")},
                    }
                    n = round((ventana["xmax"] - ventana["xmin"]) / campania["resolucion_m"])
                    fila_descarga = {
                        **ventana,
                        "tile_id": ventana["id"],
                        "ancho_nativo": n,
                        "alto_nativo": n,
                    }
                    if anio == 2023:
                        recorte, fuente = leer_referencia_local(fila_descarga)
                    else:
                        if anio not in comprobadas:
                            comprobar_wcs(origen, campania["resolucion_m"])
                            comprobadas.add(anio)
                        recorte, fuente = extraer_wcs(fila_descarga, origen)
                    guardar_png(recorte, imagen)
                if prediccion_existente:
                    with almacen.db:
                        almacen.db.execute(
                            "UPDATE recursos SET imagen=?,fuente=?,error='' WHERE ventana=? AND anio=?",
                            (str(imagen), fuente, ventana["id"], anio),
                        )
                    continue
                with almacen.db:
                    almacen.db.execute(
                        "INSERT OR REPLACE INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                        (
                            ventana["id"],
                            anio,
                            str(imagen),
                            "",
                            fuente,
                            "",
                            campania["resolucion_m"],
                            firma,
                        ),
                    )
                if modelo is None:
                    modelo = cargar_modelo(PESOS_UNET, dispositivo)
                rgb = np.asarray(
                    Image.open(imagen).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
                )
                tensor = torch.from_numpy(
                    np.ascontiguousarray(normalizar_rgb(rgb).transpose(2, 0, 1))
                ).unsqueeze(0)
                probabilidades, _ = predecir(modelo, tensor, dispositivo)
                prediccion = (
                    almacen.ruta.parent
                    / "predicciones"
                    / str(anio)
                    / f"{ventana['id']}_completa.png"
                )
                guardar_png(
                    Image.fromarray(np.rint(probabilidades * 65535).astype("uint16")), prediccion
                )
                registrar_componentes(
                    almacen,
                    ventana,
                    anio,
                    probabilidades,
                    firma,
                    geografia,
                    "piloto_recuperado" if ventana["origen"] == "piloto" else ventana["origen"],
                    config.get("corte_segmentacion_m", 0.6),
                )
                with almacen.db:
                    almacen.db.execute(
                        "UPDATE recursos SET prediccion=?,error='' WHERE ventana=? AND anio=?",
                        (str(prediccion), ventana["id"], anio),
                    )
            except (OSError, ValueError, requests.RequestException) as error:
                with almacen.db:
                    almacen.db.execute(
                        "INSERT OR REPLACE INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                        (
                            ventana["id"],
                            anio,
                            str(imagen) if imagen.exists() else "",
                            "",
                            fuente,
                            str(error),
                            campania["resolucion_m"],
                            firma,
                        ),
                    )
                print(f"{ventana['id']} {anio}: pendiente por {error}", flush=True)
        if numero % 10 == 0 or numero == len(ventanas):
            print(f"Preparación {numero}/{len(ventanas)}: {ventana['id']}", flush=True)
        if al_completar:
            al_completar(ventana["id"])


def incorporar_predicciones_existentes(almacen):
    """Registra probabilidades históricas existentes sin inferencia ni descarga."""
    geografia = GeografiaSeguimiento(almacen.meta("proximidad_m"))
    recursos = almacen.filas(
        "SELECT r.*,v.xmin,v.ymin,v.xmax,v.ymax,v.id FROM recursos r JOIN ventanas v ON v.id=r.ventana WHERE r.prediccion<>''"
    )
    for r in recursos:
        if almacen.db.execute(
            "SELECT 1 FROM componentes WHERE ventana=? AND anio=?", (r["id"], r["anio"])
        ).fetchone():
            continue
        probabilidades = np.array(Image.open(r["prediccion"]), dtype=float) / 65535
        registrar_componentes(
            almacen, r, r["anio"], probabilidades, r["modelo"], geografia, "piloto_probabilidad"
        )


def preparar_region(almacen, config: dict, lote: int, modo: str):
    """Recorre ventanas locales; fuera_indice recupera las nunca procesadas.

    Args:
        almacen: Base con cursor durable por recorrido.
        config: Configuración del seguimiento.
        lote: Máximo de ventanas para esta ejecución.
        modo: fuera_indice, indice o todas; ningún modo descarta paneles por Catastro.
    """
    indice = pd.read_csv(ruta_proyecto("data/manifests/teselas.csv"))
    incluidas = set(zip(indice.tif, indice.fila, indice.columna))
    territorio = union_all(
        gpd.read_file(ruta_proyecto("data/geografia/municipios_cantabria.gpkg")).geometry
    )
    clave = "cursor_region_" + modo
    cursor = almacen.meta(
        clave, {"hoja": 0, "fila": 0, "columna": 0, "completo": False, "procesadas": 0}
    )
    if cursor["completo"]:
        print("Este recorrido regional ya está preparado.")
        return
    modelo = None
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    firma = hashlib.sha256(PESOS_UNET.read_bytes()).hexdigest()
    geografia = GeografiaSeguimiento(almacen.meta("proximidad_m"))
    hojas = sorted(ruta_proyecto("data/pnoa").glob("*.tif"))
    modelos = almacen.filas("SELECT DISTINCT modelo FROM recursos WHERE prediccion<>''")
    if any(m["modelo"] and m["modelo"] != firma for m in modelos):
        raise ValueError("El modelo difiere del congelado del inventario")
    firma_hojas = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in hojas}
    previa = almacen.meta("ortofotos_region")
    if previa and previa != firma_hojas:
        raise ValueError(
            "Las ortofotos cambiaron desde el comienzo del recorrido; comprueba la procedencia antes de continuar"
        )
    almacen.poner_meta("ortofotos_region", firma_hojas)
    terminadas = 0
    print(
        f"Recuperación regional: hasta {lote} ventanas locales; solo se conservan PNG con detecciones. Máximo RGB ≈ {lote * 0.75:.1f} MB. Sin descargas de hojas.",
        flush=True,
    )
    for numero in range(cursor["hoja"], len(hojas)):
        with rasterio.open(hojas[numero]) as tif:
            if (
                tif.crs is None
                or tif.crs.to_epsg() != 25830
                or not np.allclose(tif.res, (0.15, 0.15))
            ):
                raise ValueError(f"Ortofoto incompatible: {hojas[numero]}")
            fila_inicial = cursor["fila"] if numero == cursor["hoja"] else 0
            for fila in range(fila_inicial, tif.height, 512):
                inicio = (
                    cursor["columna"] if numero == cursor["hoja"] and fila == cursor["fila"] else 0
                )
                for columna in range(inicio, tif.width, 512):
                    posicion = (hojas[numero].name, fila, columna)
                    if (
                        modo == "fuera_indice"
                        and posicion in incluidas
                        or modo == "indice"
                        and posicion not in incluidas
                    ):
                        continue
                    v = Window(
                        columna, fila, min(512, tif.width - columna), min(512, tif.height - fila)
                    )
                    limites = tif.window_bounds(v)
                    if not box(*limites).intersects(territorio):
                        continue
                    wid = f"{hojas[numero].stem}_f{fila}_c{columna}"
                    tarea = almacen.filas("SELECT estado FROM tareas WHERE id=?", (wid,))
                    if not tarea or tarea[0]["estado"] != "completa":
                        if modelo is None:
                            modelo = cargar_modelo(PESOS_UNET, dispositivo)
                        rgb = np.moveaxis(tif.read([1, 2, 3], window=v), 0, -1)
                        entrada = np.zeros((512, 512, 3), dtype="uint8")
                        entrada[: rgb.shape[0], : rgb.shape[1]] = rgb
                        tensor = torch.from_numpy(
                            np.ascontiguousarray(normalizar_rgb(entrada).transpose(2, 0, 1))
                        ).unsqueeze(0)
                        probabilidades, _ = predecir(modelo, tensor, dispositivo)
                        probabilidades = probabilidades[: rgb.shape[0], : rgb.shape[1]]
                        registro = dict(zip(("xmin", "ymin", "xmax", "ymax"), limites))
                        with almacen.db:
                            almacen.db.execute(
                                "INSERT OR IGNORE INTO ventanas VALUES (?,?,?,?,?,?)",
                                (wid, *limites, "regional_" + modo),
                            )
                        if (probabilidades >= 0.5).any():
                            imagen = almacen.ruta.parent / "images/2023" / f"{wid}.png"
                            prediccion = (
                                almacen.ruta.parent / "predicciones/2023" / f"{wid}_completa.png"
                            )
                            guardar_png(Image.fromarray(rgb), imagen)
                            guardar_png(
                                Image.fromarray(np.rint(probabilidades * 65535).astype("uint16")),
                                prediccion,
                            )
                            registrar_componentes(
                                almacen,
                                {"id": wid, **registro},
                                2023,
                                probabilidades,
                                firma,
                                geografia,
                                "regional_" + modo,
                            )
                            with almacen.db:
                                almacen.db.execute(
                                    "INSERT OR REPLACE INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                                    (
                                        wid,
                                        2023,
                                        str(imagen),
                                        str(prediccion),
                                        str(hojas[numero]),
                                        "",
                                        0.15,
                                        firma,
                                    ),
                                )
                        with almacen.db:
                            almacen.db.execute(
                                "INSERT OR REPLACE INTO tareas VALUES (?,?,?,?,?,?)",
                                (wid, "regional", json.dumps(registro), "completa", "", ahora()),
                            )
                    terminadas += 1
                    cursor["procesadas"] += 1
                    almacen.poner_meta(
                        clave,
                        {
                            "hoja": numero,
                            "fila": fila,
                            "columna": columna + 512,
                            "completo": False,
                            "procesadas": cursor["procesadas"],
                        },
                    )
                    if terminadas >= lote:
                        print(
                            f"Ventanas regionales preparadas: {terminadas}; cursor guardado.",
                            flush=True,
                        )
                        return
    almacen.poner_meta(clave, {**cursor, "completo": True})


def ampliar_contexto(almacen, config: dict, sitio: str, progreso=None):
    """Añade ventanas contiguas para ver todo el grupo sin alterar sus predicciones.

    Args:
        almacen: Base con emplazamientos y recortes reutilizables.
        config: Campañas oficiales y contexto en metros.
        sitio: Grupo que necesita contexto adicional.
        progreso: Recibe el espacio estimado y cada campaña pendiente.
    """
    s = almacen.filas("SELECT * FROM emplazamientos WHERE id=?", (sitio,))[0]
    componentes = almacen.filas(
        "SELECT c.geometria FROM componentes c JOIN miembros m ON m.componente=c.id WHERE m.emplazamiento=? AND c.anio=2023",
        (sitio,),
    )
    geometria = (
        union_all([from_wkb(c["geometria"]) for c in componentes])
        if componentes
        else from_wkb(s["geometria"])
    )
    vinculadas = almacen.filas(
        "SELECT v.* FROM ventanas v JOIN contextos c ON c.ventana=v.id WHERE c.emplazamiento=?",
        (sitio,),
    )
    if vinculadas:
        geometria = union_all(
            [geometria, *[box(v["xmin"], v["ymin"], v["xmax"], v["ymax"]) for v in vinculadas]]
        )
    xmin, ymin, xmax, ymax = geometria.buffer(config.get("contexto_m", 20)).bounds
    paso = config.get("extension_m", 76.8)
    nx, ny = max(1, int(np.ceil((xmax - xmin) / paso))), max(1, int(np.ceil((ymax - ymin) / paso)))
    centro = geometria.centroid
    xmin, ymin = centro.x - nx * paso / 2, centro.y - ny * paso / 2
    cuadros = [
        (xmin + ix * paso, ymin + iy * paso, xmin + (ix + 1) * paso, ymin + (iy + 1) * paso)
        for ix in range(nx)
        for iy in range(ny)
    ]
    recursos = almacen.filas(
        "SELECT r.* FROM recursos r JOIN contextos c ON c.ventana=r.ventana WHERE c.emplazamiento=?",
        (sitio,),
    )
    if vinculadas and any(
        not any(
            r["ventana"] == v["id"]
            and r["anio"] == c["anio"]
            and r["imagen"]
            and Path(r["imagen"]).exists()
            for r in recursos
        )
        for v in vinculadas
        for c in config["campanias"]
    ):
        cuadros = [tuple(v[k] for k in ("xmin", "ymin", "xmax", "ymax")) for v in vinculadas]
    if len(cuadros) > 64:
        raise ValueError(
            "Contexto muy grande: revisa las partes con sus recortes y conserva alcance zona_visible"
        )
    estimados = len(cuadros) * len(config["campanias"]) * 0.75 * 1024**2
    if estimados * 2 > shutil.disk_usage(almacen.ruta.parent).free:
        raise OSError("No hay espacio para ampliar el contexto")
    estimacion = (
        f"Contexto: hasta {len(cuadros) * len(config['campanias'])} ventanas; "
        f"RGB máximo ≈ {estimados / 1024**2:.1f} MB."
    )
    print(estimacion, flush=True)
    if progreso:
        progreso(estimacion)
    with almacen.db:
        for limites in cuadros:
            wid = identificador("contexto", repr(limites).encode())
            almacen.db.execute(
                "INSERT OR IGNORE INTO ventanas VALUES (?,?,?,?,?,?)",
                (wid, *limites, "contexto_visual"),
            )
            almacen.db.execute("INSERT OR IGNORE INTO contextos VALUES (?,?)", (sitio, wid))
    for numero, limites in enumerate(cuadros, 1):
        wid = identificador("contexto", repr(limites).encode())
        for campania in config["campanias"]:
            anio = campania["anio"]
            recurso = almacen.filas(
                "SELECT * FROM recursos WHERE ventana=? AND anio=?", (wid, anio)
            )
            if recurso and recurso[0]["imagen"] and Path(recurso[0]["imagen"]).exists():
                continue
            if progreso:
                progreso(f"Ventana {numero}/{len(cuadros)} · campaña {anio}")
            origen = {
                "tipo": "wcs",
                "url": URL_WCS,
                "bandas": [1, 2, 3],
                **{k: campania[k] for k in ("cobertura", "etiqueta", "divisor_rgb")},
            }
            n = round(paso / campania["resolucion_m"])
            fila = dict(zip(("xmin", "ymin", "xmax", "ymax"), limites))
            imagen, fuente = extraer_wcs(
                {**fila, "tile_id": wid, "ancho_nativo": n, "alto_nativo": n}, origen
            )
            ruta = almacen.ruta.parent / "images" / str(anio) / f"{wid}.png"
            guardar_png(imagen, ruta)
            with almacen.db:
                almacen.db.execute(
                    "INSERT OR REPLACE INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                    (wid, anio, str(ruta), "", fuente, "", campania["resolucion_m"], ""),
                )


def anadir_candidato(almacen, x: float, y: float, ventana: str | None = None) -> str:
    """Añade una localización humana sin inventar componentes del modelo.

    Args:
        almacen: Base persistente.
        x: Coordenada este en EPSG:25830.
        y: Coordenada norte en EPSG:25830.
        ventana: Recorte visible que contiene el punto; reutiliza sus fotos si se indica.

    Returns:
        Identificador estable para una instalación retirada antes de 2023 o no detectada.
    """
    sid = identificador("manual", f"{x:.3f}|{y:.3f}".encode())
    wid = ventana or "ventana_" + sid
    geometria = box(x - 5, y - 5, x + 5, y + 5)
    info = GeografiaSeguimiento(almacen.meta("proximidad_m")).describir(geometria)
    with almacen.db:
        if not ventana:
            almacen.db.execute(
                "INSERT OR IGNORE INTO ventanas VALUES (?,?,?,?,?,?)",
                (wid, x - 38.4, y - 38.4, x + 38.4, y + 38.4, "candidato_manual"),
            )
        almacen.db.execute(
            "INSERT OR IGNORE INTO emplazamientos(id,ventana,geometria,municipio,codigo_municipal,origen,etiquetas) VALUES (?,?,?,?,?,?,?)",
            (
                sid,
                wid,
                geometria.wkb,
                info["municipio"],
                info["codigo_municipal"],
                "candidato_manual",
                '["agrupacion_pendiente"]',
            ),
        )
    return sid


def analizar_desbordes(almacen):
    """Estima una distancia descriptiva con los desbordes parciales observados.

    Args:
        almacen: Predicciones completas recuperadas.

    Returns:
        Distribución de distancias; no estima una traslación física de Catastro.
    """
    geografia = GeografiaSeguimiento()
    distancias = []
    for c in almacen.filas(
        "SELECT * FROM componentes WHERE anio=2023 AND fraccion>0 AND fraccion<0.99 AND area>=1"
    ):
        geometria = from_wkb(c["geometria"])
        edificios = union_all(
            geografia.edificios.geometry.iloc[
                geografia.edificios.sindex.query(geometria, predicate="intersects")
            ]
        )
        exterior = geometria.difference(edificios)
        if not exterior.is_empty:
            distancias.append(exterior.representative_point().distance(edificios))
    if not distancias:
        return {"estado": "sin_muestra_para_desbordes"}
    radio = float(np.quantile(distancias, 0.95))
    resultado = {
        "componentes_parciales_analizados": len(distancias),
        "mediana_m": float(np.median(distancias)),
        "percentil_95_m": radio,
        "interpretacion": "Distancia del punto interior del desborde al edificio; mezcla segmentación, relieve, posición de paneles y desplazamiento. No es una corrección geométrica.",
        "estrategia": "Próximo si distancia <= percentil 95 observado; clasificación descriptiva, sin asignación ni filtro.",
    }
    almacen.poner_meta("proximidad_m", radio)
    almacen.poner_meta("diagnostico_catastro", resultado)
    with almacen.db:
        almacen.db.execute(
            "UPDATE componentes SET relacion=CASE WHEN distancia<=? THEN 'proximo' ELSE 'exterior' END WHERE fraccion=0",
            (radio,),
        )
        for s in almacen.filas(
            "SELECT * FROM emplazamientos WHERE relacion IN ('exterior','sin_asociacion','proximo')"
        ):
            if almacen.db.execute(
                "SELECT 1 FROM acciones WHERE tipo='sitio' AND objetivo=?", (s["id"],)
            ).fetchone():
                continue
            c = almacen.filas(
                "SELECT c.* FROM componentes c JOIN miembros m ON c.id=m.componente WHERE m.emplazamiento=?",
                (s["id"],),
            )
            relaciones = {r["relacion"] for r in c}
            if len(relaciones) == 1:
                almacen.db.execute(
                    "UPDATE emplazamientos SET relacion=? WHERE id=?",
                    (next(iter(relaciones)), s["id"]),
                )
    return resultado


def incorporar_inventario(almacen, config):
    """Conserva también los registros positivos que no estaban en la muestra piloto.

    Args:
        almacen: Base con la muestra y decisiones ya importadas.
        config: Extensión de contexto inicial.
    """
    if almacen.meta("inventario_importado"):
        return
    edificios = gpd.read_file(INDICADORES_UNET / "edificios.gpkg", where="presencia_fotovoltaica=1")
    existentes = {v["id"] for v in almacen.filas("SELECT id FROM ventanas")}
    cantidad = 0
    with almacen.db:
        for fila in edificios.itertuples():
            wid = f"edificio_{fila.id_edificio}"
            if wid in existentes:
                continue
            punto = fila.geometry.representative_point()
            mitad = config.get("extension_m", 76.8) / 2
            almacen.db.execute(
                "INSERT INTO ventanas VALUES (?,?,?,?,?,?)",
                (
                    wid,
                    punto.x - mitad,
                    punto.y - mitad,
                    punto.x + mitad,
                    punto.y + mitad,
                    "inventario_2023",
                ),
            )
            almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,municipio,codigo_municipal,edificios,origen,etiquetas) VALUES (?,?,?,?,?,?,?,?)",
                (
                    wid,
                    wid,
                    fila.geometry.wkb,
                    fila.municipio,
                    str(fila.codigo_municipal),
                    json.dumps([str(fila.id_edificio)]),
                    "inventario_2023",
                    '["agrupacion_pendiente","localizacion_pendiente"]',
                ),
            )
            cantidad += 1
        almacen.db.execute(
            "INSERT OR REPLACE INTO metadatos VALUES ('inventario_importado',?)",
            (
                json.dumps(
                    {
                        "anadidos": cantidad,
                        "origen": "3188 registros positivos; no cuenta instalaciones",
                    }
                ),
            ),
        )


def leer_referencia_local(fila):
    """Lee solo la ventana de los TIFF de 2023 existentes, incluidos bordes entre hojas."""
    limites = box(*(fila[k] for k in ("xmin", "ymin", "xmax", "ymax")))
    with ExitStack() as pila:
        fuentes = []
        for ruta in sorted(ruta_proyecto("data/pnoa").glob("*.tif")):
            tif = pila.enter_context(rasterio.open(ruta))
            if box(*tif.bounds).intersects(limites):
                fuentes.append(tif)
        return extraer_ventana(fila, fuentes)


def incorporar_muestra_previa(almacen, teselas: list[str]):
    """Reutiliza ejemplos etiquetados en suelo y construcciones auxiliares.

    Args:
        almacen: Base común del seguimiento.
        teselas: Casos deliberadamente elegidos tras inspección visual del pool.
    """
    pool = leer_csv(ruta_proyecto("data/pool/manifest.csv")).set_index("tile_id")
    geografia = GeografiaSeguimiento(almacen.meta("proximidad_m"))
    sitios = []
    for tid in teselas:
        sid = identificador("muestra_previa", tid.encode())
        sitios.append(sid)
        if almacen.db.execute("SELECT 1 FROM emplazamientos WHERE id=?", (sid,)).fetchone():
            continue
        f = pool.loc[tid]
        anotacion = json.loads(
            (ruta_proyecto("data/pool/annotations") / f"{tid}.json").read_text(encoding="utf-8")
        )
        with rasterio.open(ruta_proyecto("data/pnoa") / f.tif) as tif:
            limites = tif.window_bounds(Window(f.columna, f.fila, f.ancho, f.alto))
            poligonos = [
                Polygon([tif.transform * (x + f.columna, y + f.fila) for x, y in sh["points"]])
                for sh in anotacion["shapes"]
                if sh["label"] == "panel_solar" and len(sh["points"]) >= 3
            ]
        geometria = union_all(poligonos)
        info = geografia.describir(geometria)
        with almacen.db:
            almacen.db.execute(
                "INSERT OR IGNORE INTO ventanas VALUES (?,?,?,?,?,?)",
                (tid, *limites, "muestra_previa"),
            )
            almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,municipio,codigo_municipal,origen,etiquetas) VALUES (?,?,?,?,?,?,?)",
                (
                    sid,
                    tid,
                    geometria.wkb,
                    info["municipio"],
                    info["codigo_municipal"],
                    "muestra_previa",
                    '["agrupacion_pendiente","ejemplo_pool"]',
                ),
            )
            almacen.db.execute(
                "INSERT OR IGNORE INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                (
                    tid,
                    2023,
                    str(ruta_proyecto("data/pool/images") / f"{tid}.png"),
                    "",
                    "pool etiquetado; no observación histórica",
                    "",
                    0.15,
                    "",
                ),
            )
    almacen.poner_meta("muestra_previa_sitios", sitios)
