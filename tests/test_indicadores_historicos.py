"""Valida indicadores parciales sin modificar la revisión que sirve de fuente."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd
from shapely.geometry import box

from paneles_solares.datos.seguimiento import AlmacenSeguimiento
from paneles_solares.modelos.indicadores_historicos import (
    calcular_indicadores,
    generar_avance,
    tomar_instantanea,
)


class PruebasIndicadoresHistoricos(unittest.TestCase):
    def setUp(self):
        """Prepara datos aislados para cada prueba."""

        self.temporal = tempfile.TemporaryDirectory()
        self.carpeta = Path(self.temporal.name)
        self.almacen = AlmacenSeguimiento(
            self.carpeta / "etiquetado" / "seguimiento.sqlite"
        )
        self.almacen.poner_meta(
            "campanias", [{"anio": a} for a in (2014, 2017, 2020, 2023)]
        )
        self.almacen.poner_meta("cursor_revision", {"sitio": "a", "anio": 2020})
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO ventanas VALUES ('v',430000,4800000,430100,4800100,'prueba')"
            )

    def tearDown(self):
        """Cierra la base y retira los archivos temporales de la prueba."""

        self.almacen.cerrar()
        self.temporal.cleanup()

    def caso(self, sid, decisiones, origen="manual", municipio="Camargo", alcance=None):
        """Añade un emplazamiento de prueba con sus decisiones por campaña.

        Args:
            sid: Identificador del emplazamiento de prueba.
            decisiones: Respuestas por campaña; None deja una campaña pendiente.
            origen: Procedencia del emplazamiento de prueba.
            municipio: Nombre del municipio del caso de prueba.
            alcance: Alcance opcional de las respuestas por campaña.
        """

        with self.almacen.db:
            self.almacen.db.execute(
                (
                    "INSERT INTO "
                    "emplazamientos(id,ventana,geometria,municipio,codigo_municipal,origen)"
                    " VALUES (?,'v',?,?,'39016',?)"
                ),
                (sid, box(430000, 4800000, 430010, 4800010).wkb, municipio, origen),
            )
        for a, decision in zip((2014, 2017, 2020, 2023), decisiones):
            if decision is not None:
                self.almacen.guardar_observacion(
                    sid,
                    a,
                    decision,
                    alcance=(alcance or {}).get(a, "emplazamiento_completo"),
                )

    def calcular(self):
        """Calcula el avance desde una instantánea de la base de prueba.

        Returns:
            tuple: Casos, municipios, periodos, resumen y puntos de la instantánea de
                prueba.
        """

        with closing(tomar_instantanea(self.almacen.ruta)) as copia:
            return calcular_indicadores(copia)

    def test_periodo_ritmo_y_grupos_pendientes_sin_inventar_instalaciones_confirmadas(
        self,
    ):
        """Comprueba periodos y ritmos sin dar por confirmadas las agrupaciones pendientes."""

        self.caso("a", ["ausente", "ausente", "presente", "presente"])
        self.caso("b", ["presente"] * 4)
        casos, municipios, periodos, resumen, _ = self.calcular()
        self.assertEqual(casos[0]["periodo_aparicion"], "2017_2020")
        self.assertEqual(casos[0]["aparicion_despues_de"], 2017)
        self.assertEqual(casos[0]["aparicion_hasta"], 2020)
        self.assertEqual(resumen["emplazamientos_con_periodo_util"], 2)
        self.assertEqual(municipios[0]["agrupaciones_confirmadas"], 0)
        self.assertEqual(municipios[0]["agrupaciones_pendientes"], 2)
        ritmo = next(p for p in periodos if p["periodo"] == "2017_2020")
        self.assertEqual(ritmo["emplazamientos_observados"], 1)
        self.assertAlmostEqual(ritmo["ritmo_medio_observado_por_anio"], 1 / 3, places=3)
        anteriores = next(p for p in periodos if p["periodo"] == "hasta_2014")
        self.assertIsNone(anteriores["ritmo_medio_observado_por_anio"])

    def test_dudas_ausencias_parciales_anomalias_y_respuestas_pendientes_no_se_reparten(
        self,
    ):
        """Comprueba que los casos no resueltos no se asignan a periodos de aparición."""

        self.caso("dudosa", ["ausente", "dudosa", "presente", "presente"])
        self.caso(
            "parcial",
            ["ausente", "ausente", "presente", "presente"],
            alcance={2017: "zona_visible"},
        )
        self.caso("anomala", ["presente", "ausente", "presente", "presente"])
        self.caso("sin_referencia", ["ausente", "ausente", "presente", None])
        self.caso(
            "negativa", ["ausente", "ausente", "ausente", "referencia_incorrecta"]
        )
        self.caso(
            "agrupada",
            ["ausente", "ausente", "presente", "presente"],
            alcance={2020: "agrupacion_cubiertas"},
        )
        casos, _, periodos, resumen, _ = self.calcular()
        self.assertTrue(all(not c["incluir_en_despliegue"] for c in casos))
        self.assertEqual(resumen["emplazamientos_con_periodo_util"], 0)
        self.assertEqual(sum(p["emplazamientos_observados"] for p in periodos), 0)

    def test_provisionales_y_archivados_no_inflan_cobertura_ni_recuentos(self):
        """Comprueba que los casos provisionales y archivados no inflan los indicadores."""

        self.caso("localizado", ["ausente", "ausente", "presente", "presente"])
        self.caso("legado", [], origen="inventario_2023", municipio="Sin revisión")
        self.caso("archivado", ["presente"] * 4)
        with self.almacen.db:
            self.almacen.db.execute(
                "UPDATE emplazamientos SET activo=0 WHERE id='archivado'"
            )
        casos, municipios, _, resumen, _ = self.calcular()
        self.assertEqual(len(casos), 2)
        self.assertEqual(resumen["candidatos_localizados"], 1)
        self.assertEqual(resumen["zonas_provisionales_apartadas"], 1)
        self.assertEqual(resumen["revision_historica_pct"], 100)
        sin_revision = next(m for m in municipios if m["municipio"] == "Sin revisión")
        self.assertIsNone(sin_revision["revision_historica_pct"])

    def test_superficie_usa_union_de_mascaras_sin_area_de_edificio_y_mapa_en_wgs84(
        self,
    ):
        """Comprueba el área de la unión de máscaras y las coordenadas WGS84 del mapa."""

        self.caso(
            "a",
            ["ausente", "ausente", "presente", "presente"],
            origen="inventario_2023",
        )
        with self.almacen.db:
            for cid, geom in (
                ("c1", box(430000, 4800000, 430002, 4800002)),
                ("c2", box(430001, 4800000, 430003, 4800002)),
            ):
                self.almacen.db.execute(
                    (
                        "INSERT INTO componentes(id,ventana,anio,geometria,area) VALUES"
                        " (?,'v',2023,?,?)"
                    ),
                    (cid, geom.wkb, geom.area),
                )
                self.almacen.db.execute("INSERT INTO miembros VALUES (?,'a')", (cid,))
        casos, _, _, _, geojson = self.calcular()
        self.assertEqual(casos[0]["superficie_fv_detectada_2023_m2"], 6)
        self.assertFalse(casos[0]["zona_provisional"])
        lon, lat = geojson["features"][0]["geometry"]["coordinates"]
        self.assertTrue(-5 < lon < -3)
        self.assertTrue(43 < lat < 44)
        self.assertNotIn("potencia_central_kwp", casos[0])

    def test_exportacion_separada_conserva_base_cursor_respuestas_y_archivos_existentes(
        self,
    ):
        """Comprueba que exportar el avance conserva la base, el cursor y las salidas
        anteriores.
        """

        self.caso("a", ["ausente", "ausente", "presente", "presente"])
        firma = hashlib.sha256(self.almacen.ruta.read_bytes()).hexdigest()
        salida_anterior = self.carpeta / "resultado_visor.csv"
        salida_anterior.write_text("original", encoding="utf-8")
        salida = self.carpeta / "avance"
        resumen = generar_avance(self.almacen.ruta, salida)
        self.assertEqual(
            hashlib.sha256(self.almacen.ruta.read_bytes()).hexdigest(), firma
        )
        self.assertEqual(
            self.almacen.meta("cursor_revision"), {"sitio": "a", "anio": 2020}
        )
        self.assertEqual(salida_anterior.read_text(encoding="utf-8"), "original")
        tabla = pd.read_csv(salida / "emplazamientos_historicos.csv")
        self.assertEqual(tabla.loc[0, "periodo_aparicion"], "2017_2020")
        self.assertEqual(resumen["historicos_guardados"], 3)
        contenido = (salida / "avance_historico.html").read_text(encoding="utf-8")
        self.assertNotIn("<script", contenido)
        self.assertNotIn("https://", contenido)
        self.assertEqual(
            json.loads((salida / "resumen.json").read_text())[
                "observaciones_en_instantanea"
            ],
            4,
        )
        with self.assertRaises(FileExistsError):
            generar_avance(self.almacen.ruta, salida)

    def test_instantanea_es_consistente_y_no_admite_escrituras(self):
        """Comprueba que la instantánea es consistente y rechaza escrituras."""

        self.caso("a", ["ausente", "ausente", "presente", "presente"])
        with closing(tomar_instantanea(self.almacen.ruta)) as copia:
            self.almacen.guardar_observacion("a", 2020, "dudosa")
            anteriores, _, _, _, _ = calcular_indicadores(copia)
            self.assertEqual(anteriores[0]["decision_2020"], "presente")
            with self.assertRaises(sqlite3.OperationalError):
                copia.execute("DELETE FROM observaciones")
        nuevos, _, _, _, _ = self.calcular()
        self.assertEqual(nuevos[0]["decision_2020"], "dudosa")

    def test_base_inexistente_no_crea_un_archivo_vacio(self):
        """Comprueba que consultar una base inexistente no crea un archivo vacío."""

        ruta = self.carpeta / "inexistente.sqlite"
        with self.assertRaises(sqlite3.OperationalError):
            tomar_instantanea(ruta)
        self.assertFalse(ruta.exists())

    def test_base_sin_casos_exporta_cabeceras_y_cobertura_desconocida(self):
        """Comprueba la exportación de una base vacía con cabeceras y cobertura sin
        definir.
        """

        salida = self.carpeta / "vacio"
        resumen = generar_avance(self.almacen.ruta, salida)
        self.assertIsNone(resumen["revision_historica_pct"])
        for archivo in (
            "emplazamientos_historicos.csv",
            "indicadores_municipales.csv",
            "despliegue_por_periodo.csv",
        ):
            self.assertTrue(pd.read_csv(salida / archivo).empty)

    def test_cambio_anotado_no_se_convierte_en_superficie_o_potencia_historica(self):
        """Comprueba que un cambio anotado no inventa superficie o potencia histórica."""

        self.caso("a", ["ausente", "ausente", "presente", "presente"])
        self.almacen.guardar_observacion(
            "a", 2023, "presente", etiquetas=["ampliacion"]
        )
        casos, _, _, resumen, _ = self.calcular()
        self.assertTrue(casos[0]["cambio_anotado"])
        self.assertIsNone(casos[0]["superficie_fv_detectada_2023_m2"])
        self.assertIn("no calculada", resumen["potencia_por_epoca"])


if __name__ == "__main__":
    unittest.main()
