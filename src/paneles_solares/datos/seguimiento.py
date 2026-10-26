"""Campañas de ortofoto y conservación de las respuestas del seguimiento anterior."""

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely import from_wkb

from paneles_solares.rutas import ruta_proyecto

CARPETA = ruta_proyecto("data/datasets/seguimiento")
BASE = CARPETA / "seguimiento.sqlite"
URL_WCS = "https://geoservicios.cantabria.es/inspire/services/Ortofoto_Series_WCS/MapServer/WCSServer"


CONFIGURACION = {
    "campanias": [
        {
            "anio": 2023,
            "resolucion_m": 0.15,
            "cobertura": "Ortofoto_de_2023",
            "etiqueta": "Ortofoto_de_2023",
            "divisor_rgb": 1.0,
        },
        {
            "anio": 2020,
            "resolucion_m": 0.15,
            "cobertura": "Ortofoto_de_2020",
            "etiqueta": "Ortofoto_de_2020",
            "divisor_rgb": 1.0,
        },
        {
            "anio": 2017,
            "resolucion_m": 0.25,
            "cobertura": "Ortofoto_de_2017",
            "etiqueta": "Ortofoto_de_2017",
            "divisor_rgb": 257.0,
        },
        {
            "anio": 2014,
            "resolucion_m": 0.25,
            "cobertura": "Ortofoto_de_2014",
            "etiqueta": "Ortofoto_de_2014",
            "divisor_rgb": 1.0,
        },
    ],
}


def cargar_config(archivo: str | Path | None = None) -> dict:
    """Carga la configuración del seguimiento o una alternativa en JSON.

    Args:
        archivo: JSON opcional; None utiliza CONFIGURACION de este módulo.

    Returns:
        dict: Configuración de las campañas.

    Raises:
        ValueError: Si las campañas se repiten.
    """

    if archivo is None:
        config = deepcopy(CONFIGURACION)
    else:
        config = json.loads(ruta_proyecto(archivo).read_text(encoding="utf-8"))
    anios = [c["anio"] for c in config["campanias"]]
    if len(set(anios)) != len(anios):
        raise ValueError("Las campañas no pueden repetirse")
    return config


def guardar_csv(tabla: pd.DataFrame, ruta: Path) -> None:
    """Publica una exportación completa sin retirar antes la anterior.

    Args:
        tabla: Tabla completa que se exporta.
        ruta: Ruta del CSV de salida.
    """

    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".csv.part")
    try:
        tabla.to_csv(temporal, index=False)
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)


def recuperar_revision(base: Path = BASE):
    """Recupera máscaras de 2023 y respuestas humanas sin abrir la base para escritura.

    Args:
        base: Archivo heredado del visor; sus agrupaciones no son instalaciones.

    Returns:
        tuple: Máscaras de referencia, selecciones con máscara y respuestas por año.
            No se recuperan las predicciones de las campañas antiguas.
    """

    with closing(sqlite3.connect(f"{base.resolve().as_uri()}?mode=ro", uri=True)) as db:
        referencia = pd.read_sql_query(
            "SELECT c.id AS id_panel,c.ventana AS ventanas,c.anio,c.geometria,"
            "c.borde AS borde_imagen,c.modelo AS modelo_sha256,"
            "m.emplazamiento AS id_seleccion,o.decision AS decision_referencia "
            "FROM componentes c JOIN miembros m ON m.componente=c.id "
            "LEFT JOIN observaciones o ON o.emplazamiento=m.emplazamiento AND o.anio=2023 "
            "WHERE c.anio=2023 ORDER BY c.id",
            db,
        )
        selecciones = pd.read_sql_query(
            "SELECT e.id AS id_seleccion,e.ventana,e.geometria FROM emplazamientos e "
            "WHERE EXISTS(SELECT 1 FROM miembros m JOIN componentes c ON c.id=m.componente "
            "WHERE m.emplazamiento=e.id AND c.anio=2023) ORDER BY e.id",
            db,
        )
        respuestas = pd.read_sql_query(
            "SELECT emplazamiento AS id_seleccion,anio,decision,alcance,fecha_utc "
            "FROM observaciones ORDER BY emplazamiento,anio",
            db,
        )

    capas = []
    for tabla in (referencia, selecciones):
        capas.append(
            gpd.GeoDataFrame(
                tabla.drop(columns="geometria"),
                geometry=from_wkb(tabla.geometria.to_numpy()),
                crs=25830,
            )
        )
    referencia, selecciones = capas
    selecciones["n_componentes"] = selecciones.id_seleccion.map(
        referencia.groupby("id_seleccion").size()
    )
    referencia["superficie_horizontal_m2"] = referencia.geometry.area
    referencia["anio_referencia"] = 2023
    referencia["revision_completa"] = False
    referencia["fuente_geometria"] = "modelo_2023"
    referencia["borde_imagen"] = referencia.borde_imagen.astype(bool)
    return referencia, selecciones, respuestas


class AlmacenSeguimiento:
    """Conserva la base heredada para actualizar únicamente sus asociaciones."""

    def __init__(self, ruta: Path = BASE):
        """Abre la base persistente y crea las tablas que falten.

        Args:
            ruta: Ruta del archivo que se utiliza.
        """

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
        """)

    def cerrar(self):
        """Cierra la base de datos."""

        self.db.close()

    def filas(self, sql: str, parametros: tuple = ()) -> list[dict]:
        """Lee filas como diccionarios.

        Args:
            sql: Consulta SQL de lectura.
            parametros: Valores que se pasan a la consulta.

        Returns:
            list[dict]: Filas de la consulta como diccionarios.
        """

        return [dict(f) for f in self.db.execute(sql, parametros)]

    def poner_meta(self, clave: str, valor):
        """Guarda un metadato de la referencia geográfica.

        Args:
            clave: Nombre del metadato.
            valor: Valor que se conserva en formato JSON.
        """

        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadatos VALUES (?,?)",
                (clave, json.dumps(valor, ensure_ascii=False)),
            )
