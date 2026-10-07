"""Datos originales, observaciones independientes y progreso histórico en SQLite."""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely import from_wkb, make_valid, union_all
from shapely.geometry import shape

from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/datasets/seguimiento")
BASE = CARPETA / "seguimiento.sqlite"
URL_WCS = (
    "https://geoservicios.cantabria.es/inspire/services/Ortofoto_Series_WCS/MapServer/WCSServer"
)
DECISIONES = ("presente", "ausente", "dudosa", "no_evaluable", "referencia_incorrecta")
SOPORTES = (
    "cubierta",
    "cubierta_catastrada",
    "construccion_no_catastrada",
    "marquesina",
    "suelo",
    "mixto",
    "desconocido",
)
RELACIONES = (
    "interseccion_clara",
    "proximo",
    "parcial_desplazado",
    "exterior",
    "sin_asociacion",
    "ambigua",
)
ETIQUETAS = (
    "ampliacion",
    "reduccion",
    "retirada",
    "reinstalacion",
    "distribucion_diferente",
    "edificio_nuevo",
    "edificio_demolido",
    "edificio_reformado",
    "division",
    "fusion",
    "desplazamiento_ortofoto",
    "mala_resolucion",
    "sombra",
    "nube",
    "sin_cobertura",
    "confusion_claraboyas",
    "cambio_soporte",
    "secuencia_incoherente",
    "borde_recorte",
    "agrupacion_pendiente",
    "asociacion_pendiente",
)


def ahora() -> str:
    """Devuelve la fecha UTC para las acciones persistentes."""
    return datetime.now(timezone.utc).isoformat()


def id_ventana(identificador: str) -> str:
    """Recupera el recorte compartido por las cubiertas del piloto."""
    return str(identificador).split("_cubierta_")[0]


def cargar_config(archivo: str = "seguimiento.json") -> dict:
    """Lee la configuración única.

    Args:
        archivo: Ruta relativa al proyecto o absoluta.

    Returns:
        Campañas y tamaño de los lotes.

    Raises:
        ValueError: Si las campañas se repiten o no incluyen la referencia.
    """
    config = json.loads(ruta_proyecto(archivo).read_text(encoding="utf-8"))
    anios = [c["anio"] for c in config["campanias"]]
    if len(set(anios)) != len(anios) or 2023 not in anios:
        raise ValueError("Las campañas deben ser únicas e incluir 2023")
    return config


def sustituir_archivo(temporal: Path, destino: Path) -> None:
    """Publica un archivo cerrado y reintenta bloqueos breves de Windows."""
    for intento in range(3):
        try:
            temporal.replace(destino)
            return
        except PermissionError:
            if intento == 2:
                raise
            time.sleep(0.1 * (intento + 1))


def guardar_csv(tabla: pd.DataFrame, ruta: Path) -> None:
    """Publica una exportación completa sin retirar antes la anterior."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".csv.part")
    try:
        tabla.to_csv(temporal, index=False)
        sustituir_archivo(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)


def guardar_json(datos: dict, ruta: Path) -> None:
    """Publica un resumen JSON mediante un temporal retirado al terminar."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".json.part")
    try:
        temporal.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        sustituir_archivo(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee los datos del piloto conservando identificadores y campos vacíos."""
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(
        ruta,
        keep_default_na=False,
        dtype={"id_instalacion": str, "id_ventana": str, "codigo_municipal": str},
    )


def transformacion(fila: dict, ancho: int = 512, alto: int = 512):
    """Relaciona píxeles con el encuadre métrico original, sin ajustes a Catastro."""
    return from_bounds(*(fila[c] for c in ("xmin", "ymin", "xmax", "ymax")), ancho, alto)


def geometria_mascara(mascara: np.ndarray, transformacion_pixel) -> list:
    """Conserva todos los componentes de ocho vecinos, incluidos los pequeños."""
    return [
        make_valid(shape(g))
        for g, _ in shapes(
            mascara.astype("uint8"), mask=mascara, connectivity=8, transform=transformacion_pixel
        )
    ]


def identificador(prefijo: str, contenido: bytes) -> str:
    """Crea un identificador reproducible a partir del origen inmutable."""
    return prefijo + "_" + hashlib.sha256(contenido).hexdigest()[:20]


def derivar_historia(observaciones: list[dict], anios: list[int]) -> dict:
    """Deriva visibilidad e intervalos sin convertir predicciones en observaciones.

    Args:
        observaciones: Decisiones humanas actuales de un emplazamiento.
        anios: Campañas esperadas, en cualquier orden.

    Returns:
        Primera presencia, límites e incertidumbres; sin fecha exacta inventada.
    """
    por_anio = {o["anio"]: o for o in observaciones}
    presentes = sorted(a for a, o in por_anio.items() if o["decision"] == "presente")
    ausentes = sorted(
        a
        for a, o in por_anio.items()
        if (o["decision"] == "ausente" or a == 2023 and o["decision"] == "referencia_incorrecta")
        and o.get("alcance", "emplazamiento_completo") == "emplazamiento_completo"
    )
    agrupacion = any(o.get("alcance") == "agrupacion_cubiertas" for o in observaciones)
    anomala = any(a > presentes[0] for a in ausentes) if presentes else False
    primera = presentes[0] if presentes else None
    dudosas = any(o["decision"] in ("dudosa", "no_evaluable") for o in observaciones)
    pendientes = sorted(set(anios) - set(por_anio))
    ref = por_anio.get(2023, {}).get("decision", "pendiente")
    estado = "pendiente"
    inferior = superior = None
    if anomala:
        estado = "secuencia_anomala"
    elif agrupacion:
        estado = "agrupacion_sin_resolver"
    elif primera is not None:
        anteriores = [a for a in anios if a < primera]
        desconocidos = [a for a in anteriores if a not in por_anio or a not in ausentes]
        if desconocidos:
            estado = "primera_presencia_intervalo_no_resuelto"
        elif anteriores:
            inferior, superior = max(anteriores), primera
            estado = "intervalo_observado"
        else:
            superior = primera
            estado = "presente_desde_primera_campania"
    elif not pendientes and not dudosas:
        estado = "sin_presencia_observada"
    if ref == "referencia_incorrecta" and not presentes:
        estado = "referencia_descartada"
    return {
        "primera_presencia": primera,
        "aparicion_despues_de": inferior,
        "aparicion_hasta": superior,
        "estado_historia": estado,
        "secuencia_anomala": anomala,
        "referencia_2023": ref,
        "campanias_pendientes": pendientes,
        "dudoso_no_evaluable": dudosas,
    }


class AlmacenSeguimiento:
    """Único origen de verdad, con transacciones y registro reversible de acciones."""

    def __init__(self, ruta: Path = BASE):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.ruta, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadatos (clave TEXT PRIMARY KEY, valor TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ventanas (
                id TEXT PRIMARY KEY, xmin REAL, ymin REAL, xmax REAL, ymax REAL, origen TEXT);
            CREATE TABLE IF NOT EXISTS recursos (
                ventana TEXT REFERENCES ventanas(id), anio INTEGER, imagen TEXT DEFAULT '',
                prediccion TEXT DEFAULT '', fuente TEXT DEFAULT '', error TEXT DEFAULT '',
                resolucion REAL, modelo TEXT DEFAULT '', PRIMARY KEY(ventana, anio));
            CREATE TABLE IF NOT EXISTS emplazamientos (
                id TEXT PRIMARY KEY, ventana TEXT REFERENCES ventanas(id), geometria BLOB,
                municipio TEXT DEFAULT '', codigo_municipal TEXT DEFAULT '',
                soporte TEXT DEFAULT 'desconocido', relacion TEXT DEFAULT 'sin_asociacion',
                edificios TEXT DEFAULT '[]', cubiertas TEXT DEFAULT '[]',
                agrupacion TEXT DEFAULT 'pendiente', activo INTEGER DEFAULT 1,
                origen TEXT, etiquetas TEXT DEFAULT '[]', notas TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS componentes (
                id TEXT PRIMARY KEY, ventana TEXT REFERENCES ventanas(id), anio INTEGER,
                geometria BLOB NOT NULL, area REAL, borde INTEGER, relacion TEXT,
                edificios TEXT, cubiertas TEXT, fraccion REAL, distancia REAL,
                municipio TEXT, codigo_municipal TEXT, modelo TEXT, origen TEXT);
            CREATE INDEX IF NOT EXISTS componentes_ventana ON componentes(ventana, anio);
            CREATE TABLE IF NOT EXISTS miembros (
                componente TEXT PRIMARY KEY REFERENCES componentes(id),
                emplazamiento TEXT REFERENCES emplazamientos(id));
            CREATE TABLE IF NOT EXISTS observaciones (
                emplazamiento TEXT REFERENCES emplazamientos(id), anio INTEGER,
                decision TEXT NOT NULL, alcance TEXT, etiquetas TEXT, notas TEXT,
                segundos REAL, fecha_utc TEXT, original TEXT,
                PRIMARY KEY(emplazamiento, anio));
            CREATE TABLE IF NOT EXISTS acciones (
                id INTEGER PRIMARY KEY, tipo TEXT, objetivo TEXT, anio INTEGER,
                anterior TEXT, posterior TEXT, fecha_utc TEXT, deshecha INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS tareas (
                id TEXT PRIMARY KEY, tipo TEXT, datos TEXT, estado TEXT DEFAULT 'pendiente',
                error TEXT DEFAULT '', fecha_utc TEXT);
            CREATE TABLE IF NOT EXISTS contextos (
                emplazamiento TEXT REFERENCES emplazamientos(id),
                ventana TEXT REFERENCES ventanas(id), PRIMARY KEY(emplazamiento,ventana));
        """)

    def cerrar(self):
        """Cierra una base cuyos cambios ya se han confirmado después de cada acción."""
        self.db.close()

    def filas(self, sql: str, parametros: tuple = ()) -> list[dict]:
        """Lee filas como diccionarios para la interfaz y las exportaciones."""
        return [dict(f) for f in self.db.execute(sql, parametros)]

    def meta(self, clave: str, defecto=None):
        """Lee un metadato JSON opcional."""
        fila = self.db.execute("SELECT valor FROM metadatos WHERE clave=?", (clave,)).fetchone()
        return json.loads(fila[0]) if fila else defecto

    def poner_meta(self, clave: str, valor):
        """Confirma inmediatamente un metadato de preparación o navegación."""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadatos VALUES (?,?)",
                (clave, json.dumps(valor, ensure_ascii=False)),
            )

    def guardar_observacion(
        self,
        sitio: str,
        anio: int,
        decision: str,
        segundos: float = 0,
        alcance: str = "emplazamiento_completo",
        etiquetas: list | None = None,
        notas: str = "",
        original=None,
        referencia: dict | None = None,
    ):
        """Guarda y registra una decisión en una sola transacción durable.

        Args:
            sitio: Identificador estable del emplazamiento.
            anio: Campaña observada.
            decision: Presencia, ausencia, duda, no evaluable o referencia incorrecta.
            segundos: Tiempo empleado, conservado también al corregir.
            alcance: Completo, zona visible o agrupación sin resolver.
            etiquetas: Indicadores secundarios extensibles.
            notas: Comentario libre.
            original: Fila del piloto, conservada sin reinterpretarla.
            referencia: Respuesta explícita de 2023 incluida en el primer clic histórico.

        Raises:
            ValueError: Si la decisión no es válida para esta campaña.
        """
        if decision not in DECISIONES or (decision == "referencia_incorrecta" and anio != 2023):
            raise ValueError("La referencia incorrecta solo se aplica a 2023")
        anterior = self.filas(
            "SELECT * FROM observaciones WHERE emplazamiento=? AND anio=?", (sitio, anio)
        )
        registro = {
            "emplazamiento": sitio,
            "anio": anio,
            "decision": decision,
            "alcance": alcance,
            "etiquetas": json.dumps(etiquetas or [], ensure_ascii=False),
            "notas": notas,
            "segundos": round(segundos, 2),
            "fecha_utc": ahora(),
            "original": json.dumps(original, ensure_ascii=False)
            if original
            else (anterior[0]["original"] if anterior else None),
        }
        ref_previa = self.filas(
            "SELECT * FROM observaciones WHERE emplazamiento=? AND anio=2023", (sitio,)
        )
        if referencia and anio != 2023 and (not ref_previa or referencia.get("corregir")):
            if referencia["decision"] not in DECISIONES:
                raise ValueError("Respuesta de referencia no válida")
            ref = {
                **registro,
                "anio": 2023,
                "decision": referencia["decision"],
                "alcance": referencia["alcance"],
                "etiquetas": json.dumps(referencia.get("etiquetas", []), ensure_ascii=False),
                "notas": referencia.get("notas", ""),
                "segundos": 0,
                "original": ref_previa[0]["original"] if ref_previa else None,
            }
            with self.db:
                self._escribir_observacion(ref)
                self._escribir_observacion(registro)
                self._accion(
                    "comparacion",
                    sitio,
                    anio,
                    {
                        "referencia": ref_previa[0] if ref_previa else None,
                        "historica": anterior[0] if anterior else None,
                    },
                    {"referencia": ref, "historica": registro},
                )
            return
        with self.db:
            self._escribir_observacion(registro)
            self._accion("observacion", sitio, anio, anterior[0] if anterior else None, registro)

    def _escribir_observacion(self, registro: dict):
        self.db.execute(
            "INSERT OR REPLACE INTO observaciones VALUES (?,?,?,?,?,?,?,?,?)",
            tuple(
                registro[c]
                for c in (
                    "emplazamiento",
                    "anio",
                    "decision",
                    "alcance",
                    "etiquetas",
                    "notas",
                    "segundos",
                    "fecha_utc",
                    "original",
                )
            ),
        )

    def _accion(self, tipo, sitio, anio, anterior, posterior):
        self.db.execute(
            "INSERT INTO acciones(tipo,objetivo,anio,anterior,posterior,fecha_utc) VALUES (?,?,?,?,?,?)",
            (
                tipo,
                sitio,
                anio,
                json.dumps(anterior, ensure_ascii=False),
                json.dumps(posterior, ensure_ascii=False),
                ahora(),
            ),
        )

    def modificar_sitio(self, sitio: str, **cambios):
        """Corrige soporte, asociación o notas sin cambiar las observaciones."""
        permitidos = {
            "soporte",
            "relacion",
            "edificios",
            "cubiertas",
            "agrupacion",
            "etiquetas",
            "notas",
        }
        if not cambios or not set(cambios) <= permitidos:
            raise ValueError("Campos de emplazamiento no permitidos")
        if "soporte" in cambios and cambios["soporte"] not in SOPORTES:
            raise ValueError("Soporte desconocido")
        if "relacion" in cambios and cambios["relacion"] not in RELACIONES:
            raise ValueError("Relación desconocida")
        if "edificios" in cambios and "cubiertas" not in cambios:
            cambios["cubiertas"] = []
        anterior = self.filas("SELECT * FROM emplazamientos WHERE id=?", (sitio,))[0]
        anterior = {c: anterior[c] for c in cambios}
        posterior = {
            c: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
            for c, v in cambios.items()
        }
        with self.db:
            self.db.execute(
                "UPDATE emplazamientos SET "
                + ",".join(c + "=?" for c in posterior)
                + " WHERE id=?",
                (*posterior.values(), sitio),
            )
            self._accion("sitio", sitio, None, anterior, posterior)

    def deshacer(self) -> dict | None:
        """Revierte la última acción humana manteniendo su registro y el resto del trabajo."""
        filas = self.filas(
            "SELECT * FROM acciones WHERE deshecha=0 AND tipo IN ('observacion','comparacion','sitio','agrupacion') ORDER BY id DESC LIMIT 1"
        )
        if not filas:
            return None
        accion = filas[0]
        anterior = json.loads(accion["anterior"])
        with self.db:
            if accion["tipo"] == "comparacion":
                for anio, campo in ((2023, "referencia"), (accion["anio"], "historica")):
                    self.db.execute(
                        "DELETE FROM observaciones WHERE emplazamiento=? AND anio=?",
                        (accion["objetivo"], anio),
                    )
                    if anterior[campo]:
                        self._escribir_observacion(anterior[campo])
            elif accion["tipo"] == "observacion":
                self.db.execute(
                    "DELETE FROM observaciones WHERE emplazamiento=? AND anio=?",
                    (accion["objetivo"], accion["anio"]),
                )
                if anterior:
                    self._escribir_observacion(anterior)
            elif accion["tipo"] == "sitio":
                self.db.execute(
                    "UPDATE emplazamientos SET "
                    + ",".join(c + "=?" for c in anterior)
                    + " WHERE id=?",
                    (*anterior.values(), accion["objetivo"]),
                )
            else:
                for sitio, activo in anterior["activos"].items():
                    self.db.execute(
                        "UPDATE emplazamientos SET activo=? WHERE id=?", (activo, sitio)
                    )
                for componente, sitio in anterior["miembros"].items():
                    self.db.execute(
                        "UPDATE miembros SET emplazamiento=? WHERE componente=?",
                        (sitio, componente),
                    )
                for nuevo in anterior["nuevos"]:
                    self.db.execute("UPDATE emplazamientos SET activo=0 WHERE id=?", (nuevo,))
            self.db.execute("UPDATE acciones SET deshecha=1 WHERE id=?", (accion["id"],))
        return accion

    def reorganizar(self, sitios: list[str], grupos: list[list[str]]) -> list[str]:
        """Divide o fusiona grupos conservando los originales y sus observaciones.

        Args:
            sitios: Emplazamientos que se reorganizan.
            grupos: Componentes para cada nuevo emplazamiento, sin duplicarlos.

        Returns:
            Identificadores nuevos pendientes de validación; los anteriores quedan consultables.

        Raises:
            ValueError: Si la distribución pierde componentes o mezcla campañas.
        """
        interrogantes = ",".join("?" for _ in sitios)
        miembros = self.filas(
            f"SELECT * FROM miembros WHERE emplazamiento IN ({interrogantes})", tuple(sitios)
        )
        originales = {m["componente"]: m["emplazamiento"] for m in miembros}
        propuestos = [c for grupo in grupos for c in grupo]
        if (
            not grupos
            or any(not g for g in grupos)
            or len(propuestos) != len(set(propuestos))
            or set(propuestos) != set(originales)
        ):
            raise ValueError("Reparte todos los componentes una sola vez")
        anteriores = self.filas(
            f"SELECT * FROM emplazamientos WHERE id IN ({interrogantes})", tuple(sitios)
        )
        nuevos = []
        with self.db:
            for grupo in grupos:
                nuevo = identificador(
                    "emplazamiento", ("|".join(sorted(grupo)) + "|manual|" + ahora()).encode()
                )
                datos = self.filas(
                    "SELECT * FROM componentes WHERE id IN (" + ",".join("?" for _ in grupo) + ")",
                    tuple(grupo),
                )
                geometria = union_all([from_wkb(c["geometria"]) for c in datos])
                self.db.execute(
                    "INSERT INTO emplazamientos(id,ventana,geometria,municipio,codigo_municipal,origen,etiquetas) VALUES (?,?,?,?,?,?,?)",
                    (
                        nuevo,
                        datos[0]["ventana"],
                        geometria.wkb,
                        datos[0]["municipio"],
                        datos[0]["codigo_municipal"],
                        "reagrupacion_manual",
                        json.dumps(["agrupacion_pendiente", "observaciones_en_originales"]),
                    ),
                )
                self.db.executemany(
                    "UPDATE miembros SET emplazamiento=? WHERE componente=?",
                    [(nuevo, c) for c in grupo],
                )
                nuevos.append(nuevo)
            self.db.executemany(
                "UPDATE emplazamientos SET activo=0 WHERE id=?", [(s,) for s in sitios]
            )
            self._accion(
                "agrupacion",
                ",".join(sitios),
                None,
                {
                    "activos": {s["id"]: s["activo"] for s in anteriores},
                    "miembros": originales,
                    "nuevos": nuevos,
                },
                {"nuevos": nuevos, "originales": sitios},
            )
        return nuevos
