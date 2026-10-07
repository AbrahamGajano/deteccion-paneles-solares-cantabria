"""Integración del seguimiento: conservación, persistencia, corrección y cronología."""

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import closing, contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from PIL import Image
from shapely.geometry import box

from paneles_solares.datos.seguimiento import (
    BASE,
    AlmacenSeguimiento,
    cargar_config,
    derivar_historia,
)
from paneles_solares.etiquetado.preparar_seguimiento import (
    GeografiaSeguimiento,
    ampliar_contexto,
    preparar_lote,
    registrar_componentes,
)
from paneles_solares.modelos.resumir_seguimiento import generar_resumen


class PruebasSeguimiento(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.carpeta = Path(self.temporal.name)
        self.almacen = AlmacenSeguimiento(self.carpeta / "seguimiento.sqlite")
        self.almacen.poner_meta("campanias", cargar_config()["campanias"])
        with self.almacen.db:
            self.almacen.db.execute("INSERT INTO ventanas VALUES ('v',0,0,76.8,76.8,'prueba')")
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('a','v',?,'prueba')",
                (box(10, 10, 20, 20).wkb,),
            )

    def tearDown(self):
        self.almacen.cerrar()
        self.temporal.cleanup()

    @contextmanager
    def visor_preparado(self, limite=12, precarga=False):
        import tkinter as tk

        from paneles_solares.visualizacion.revisar_seguimiento import VisorSeguimiento

        for anio in (2023, 2020, 2017, 2014):
            imagen = self.carpeta / f"{anio}.png"
            Image.new("RGB", (32, 32)).save(imagen)
            with self.almacen.db:
                self.almacen.db.execute(
                    "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                    ("v", anio, str(imagen), "", "prueba", "", 0.15, ""),
                )
        raiz = tk.Tk()
        raiz.withdraw()
        try:
            edificios = gpd.GeoDataFrame(geometry=[], crs="EPSG:25830")
            with patch(
                "paneles_solares.visualizacion.revisar_seguimiento.gpd.read_file",
                return_value=edificios,
            ):
                visor = VisorSeguimiento(
                    raiz, self.almacen, {**cargar_config(), "precarga": precarga}, limite
                )
            yield visor
        finally:
            visor.detener_anticipacion.set()
            if visor.evento_anticipacion:
                raiz.after_cancel(visor.evento_anticipacion)
            if hasattr(visor, "hilo_anticipacion"):
                visor.hilo_anticipacion.join(timeout=5)
            raiz.destroy()

    def esperar_preparacion(self, visor):
        limite = time.monotonic() + 5
        while visor.preparando and time.monotonic() < limite:
            visor.raiz.update()
            time.sleep(0.01)
        self.assertFalse(visor.preparando, "La carga no terminó")

    def esperar_hasta(self, visor, condicion):
        limite = time.monotonic() + 5
        while not condicion() and time.monotonic() < limite:
            visor.raiz.update()
            time.sleep(0.01)
        self.assertTrue(condicion(), "El visor no alcanzó el estado esperado")

    def agregar_ventana(self, ventana, sitios, preparada=False):
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO ventanas VALUES (?,0,0,76.8,76.8,'prueba')", (ventana,)
            )
            for sid in sitios:
                self.almacen.db.execute(
                    "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES (?,?,?,'prueba')",
                    (sid, ventana, box(30, 30, 40, 40).wkb),
                )
        if preparada:
            self.guardar_fotos(self.almacen, ventana)

    def guardar_fotos(self, almacen, ventana):
        for anio in (2023, 2020, 2017, 2014):
            imagen = self.carpeta / f"{ventana}_{anio}.png"
            Image.new("RGB", (32, 32)).save(imagen)
            with almacen.db:
                almacen.db.execute(
                    "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                    (ventana, anio, str(imagen), "", "prueba", "", 0.15, ""),
                )

    def test_provisionales_apartados_conservan_respuestas_y_acceso_avanzado(self):
        self.almacen.guardar_observacion("a", 2020, "dudosa", notas="Conservar")
        with self.almacen.db:
            self.almacen.db.execute(
                "UPDATE emplazamientos SET origen='inventario_2023' WHERE id='a'"
            )
        original = self.almacen.filas("SELECT * FROM observaciones")
        with self.visor_preparado() as visor:
            self.assertNotIn("a", visor.visibles)
            self.assertEqual(visor.cola, [])
            self.assertIsNone(visor.actual)
            visor.incluir_provisionales.set(True)
            visor.recargar()
            visor.siguiente(cargar=False)
            self.assertEqual((visor.actual, visor.anio.get()), ("a", "2017"))
            self.assertEqual(original, self.almacen.filas("SELECT * FROM observaciones"))
            self.assertEqual(visor.sitios["a"]["activo"], 1)

    def test_misma_imagen_se_termina_entera_aunque_ids_y_prioridades_se_intercalen(self):
        self.agregar_ventana("w", ["b", "d"], preparada=True)
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('z','v',?,'prueba')",
                (box(40, 40, 50, 50).wkb,),
            )
        self.almacen.poner_meta("muestra_previa_sitios", ["z"])
        with self.visor_preparado(limite=1) as visor:
            self.assertEqual(visor.cola, ["a", "z"])
            visor.mostrar("z", 2020)  # Reanudar en mitad de la imagen también acaba sus vecinos.
            recorrido = []
            for _ in range(12):
                recorrido.append((visor.actual, int(visor.anio.get())))
                visor.decidir("presente")
            self.assertEqual(
                recorrido, [(s, a) for s in ("z", "a", "b", "d") for a in (2020, 2017, 2014)]
            )
            self.assertTrue(visor.lote_terminado)

    def test_precarga_no_bloquea_respuestas_y_salta_automaticamente_sin_esperar_lote_entero(self):
        self.agregar_ventana("w", ["b"])
        self.agregar_ventana("x", ["c"])
        iniciar, publicar, acabar = Event(), Event(), Event()

        def preparar(
            almacen, config, lote, sitios, progreso, al_completar, solo_localizados, detener
        ):
            self.assertEqual((lote, sitios, solo_localizados), (2, ["b", "c"], True))
            iniciar.set()
            publicar.wait(5)
            self.guardar_fotos(almacen, "w")
            al_completar("w")
            acabar.wait(5)
            self.guardar_fotos(almacen, "x")
            al_completar("x")

        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.preparar_lote",
                side_effect=preparar,
            ) as cargar,
            self.visor_preparado(limite=2, precarga=True) as visor,
        ):
            try:
                self.esperar_hasta(visor, iniciar.is_set)
                self.assertTrue(visor.anticipando)
                self.assertFalse(visor.preparando)
                self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))
                visor.decidir("presente")
                self.assertEqual((visor.actual, visor.anio.get()), ("a", "2017"))
                publicar.set()
                self.esperar_hasta(visor, lambda: "b" in visor.candidatas_preparadas())
                self.assertEqual((visor.actual, visor.anio.get()), ("a", "2017"))
                visor.decidir("presente")
                visor.decidir("presente")
                self.assertEqual((visor.actual, visor.anio.get()), ("b", "2020"))
                self.assertTrue(visor.anticipando)  # El resto del lote aún se está preparando.
                for _ in visor.historicos:
                    visor.decidir("ausente")
                self.assertTrue(visor.lote_terminado)
                self.assertTrue(visor.anticipando)
                original = self.almacen.filas(
                    "SELECT * FROM observaciones ORDER BY emplazamiento,anio"
                )
                acabar.set()
                self.esperar_hasta(visor, lambda: visor.actual == "c")
                self.assertEqual(visor.anio.get(), "2020")
                self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))
                self.assertEqual(
                    original,
                    self.almacen.filas("SELECT * FROM observaciones ORDER BY emplazamiento,anio"),
                )
                cargar.assert_called_once()
            finally:
                publicar.set()
                acabar.set()

    def test_reserva_acotada_a_un_lote_y_sin_descargar_otros_municipios(self):
        self.agregar_ventana("w", ["b"], preparada=True)
        self.agregar_ventana("x", ["c"])
        with self.almacen.db:
            self.almacen.db.execute(
                "UPDATE emplazamientos SET municipio='Camargo' WHERE id IN ('a','b')"
            )
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.preparar_lote") as cargar,
            self.visor_preparado(limite=1, precarga=True) as visor,
        ):
            self.esperar_hasta(visor, lambda: visor.evento_anticipacion is None)
            cargar.assert_not_called()  # Hay una imagen en reserva.
            visor.filtros["Municipio"].set("Camargo")
            visor.recargar()
            for _ in range(6):
                visor.decidir("presente")
            visor.raiz.update()
            self.assertEqual(visor.sitios_sin_fotos(), [])
            cargar.assert_not_called()

    def test_precarga_fallida_se_reintenta_sin_dialogos_ni_cambiar_respuestas(self):
        self.agregar_ventana("w", ["b"])

        def preparado(
            almacen, config, lote, sitios, progreso, al_completar, solo_localizados, detener
        ):
            self.guardar_fotos(almacen, "w")
            al_completar("w")

        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.preparar_lote",
                side_effect=[OSError("Sin conexión"), None],
            ) as cargar,
            patch(
                "paneles_solares.visualizacion.revisar_seguimiento.messagebox.showerror"
            ) as dialogo,
            self.visor_preparado(limite=1, precarga=True) as visor,
        ):
            self.esperar_hasta(visor, lambda: "w" in visor.fallos_anticipacion)
            self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))
            for _ in visor.historicos:
                visor.decidir("presente")
            original = self.almacen.filas("SELECT * FROM observaciones")
            self.assertIn("Reintentar", visor.boton_siguiente.cget("text"))
            cargar.side_effect = preparado
            visor.boton_siguiente.invoke()
            self.esperar_hasta(visor, lambda: visor.actual == "b")
            self.assertEqual(original, self.almacen.filas("SELECT * FROM observaciones"))
            self.assertEqual(cargar.call_count, 2)
            dialogo.assert_not_called()

    def test_final_de_lote_continua_automaticamente_con_fotos_existentes(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('b','v',?,'prueba')",
                (box(30, 30, 40, 40).wkb,),
            )
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.preparar_lote") as preparar,
            self.visor_preparado(limite=1) as visor,
        ):
            self.assertEqual(visor.cola, ["a", "b"])
            for anio in (2020, 2017, 2014):
                self.assertEqual((visor.actual, visor.anio.get()), ("a", str(anio)))
                visor.decidir("presente")
            self.assertEqual((visor.actual, visor.anio.get()), ("b", "2020"))
            self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))
            self.assertEqual(len(self.almacen.filas("SELECT * FROM observaciones")), 4)
            preparar.assert_not_called()

    def test_continuar_carga_solo_el_siguiente_lote_sin_reabrir_ni_perder_respuestas(self):
        with self.almacen.db:
            self.almacen.db.execute("INSERT INTO ventanas VALUES ('w',0,0,76.8,76.8,'prueba')")
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('b','w',?,'prueba')",
                (box(30, 30, 40, 40).wkb,),
            )

        def preparar(almacen, config, lote, sitios, progreso):
            self.assertEqual(sitios, ["b"])
            self.assertEqual(lote, 1)
            progreso("Prueba de carga")
            for campania in config["campanias"]:
                anio = campania["anio"]
                imagen = self.carpeta / f"b_{anio}.png"
                Image.new("RGB", (32, 32)).save(imagen)
                with almacen.db:
                    almacen.db.execute(
                        "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                        ("w", anio, str(imagen), "", "prueba", "", 0.15, ""),
                    )

        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.preparar_lote",
                side_effect=preparar,
            ) as cargar,
            self.visor_preparado(limite=1) as visor,
        ):
            for _ in visor.historicos:
                visor.decidir("presente")
            original = self.almacen.filas("SELECT * FROM observaciones ORDER BY anio")
            self.assertTrue(visor.lote_terminado)
            self.assertIn("Continuar revisión", visor.paso.cget("text"))
            self.assertFalse(visor.boton_siguiente.instate(["disabled"]))
            visor.boton_siguiente.invoke()
            self.assertTrue(visor.preparando)
            visor.siguiente()
            visor.decidir("ausente")
            self.esperar_preparacion(visor)
            cargar.assert_called_once()
            self.assertEqual((visor.actual, visor.anio.get()), ("b", "2020"))
            self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))
            self.assertEqual(
                original, self.almacen.filas("SELECT * FROM observaciones ORDER BY anio")
            )

    def test_carga_fallida_permite_reintentar_sin_alterar_el_progreso(self):
        with self.almacen.db:
            self.almacen.db.execute("INSERT INTO ventanas VALUES ('w',0,0,76.8,76.8,'prueba')")
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('b','w',?,'prueba')",
                (box(30, 30, 40, 40).wkb,),
            )
        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.preparar_lote",
                side_effect=OSError("Sin conexión"),
            ),
            patch(
                "paneles_solares.visualizacion.revisar_seguimiento.messagebox.showerror"
            ) as aviso,
            self.visor_preparado(limite=1) as visor,
        ):
            for _ in visor.historicos:
                visor.decidir("presente")
            original = self.almacen.filas("SELECT * FROM observaciones ORDER BY anio")
            visor.boton_siguiente.invoke()
            self.esperar_preparacion(visor)
            aviso.assert_called_once()
            self.assertTrue(visor.lote_terminado)
            self.assertFalse(visor.boton_siguiente.instate(["disabled"]))
            self.assertTrue(visor.botones_decision["presente"].instate(["disabled"]))
            self.assertEqual(
                original, self.almacen.filas("SELECT * FROM observaciones ORDER BY anio")
            )

    def test_fin_con_filtro_indica_como_seguir_y_no_descarga_otros_municipios(self):
        with self.almacen.db:
            self.almacen.db.execute("UPDATE emplazamientos SET municipio='Camargo' WHERE id='a'")
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen,municipio) VALUES ('b','v',?,'prueba','Alfoz de Lloredo')",
                (box(30, 30, 40, 40).wkb,),
            )
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.preparar_lote") as preparar,
            self.visor_preparado(limite=1) as visor,
        ):
            visor.filtros["Municipio"].set("Camargo")
            visor.recargar()
            for _ in visor.historicos:
                visor.decidir("presente")
            self.assertIn("Cambia los filtros", visor.paso.cget("text"))
            self.assertTrue(visor.boton_siguiente.instate(["disabled"]))
            visor.filtros["Municipio"].set("Todos")
            visor.recargar()
            self.assertFalse(visor.boton_siguiente.instate(["disabled"]))
            visor.boton_siguiente.invoke()
            self.assertEqual((visor.actual, visor.anio.get()), ("b", "2020"))
            preparar.assert_not_called()

    def test_un_click_por_campania_guarda_y_avanza_sin_confirmar(self):
        with self.visor_preparado() as visor:
            for indice, anio in enumerate((2020, 2017, 2014), start=1):
                self.assertEqual(int(visor.anio.get()), anio)
                self.assertIn(str(anio), visor.paso.cget("text"))
                visor.botones_decision["presente"].invoke()
                visor.raiz.update()
                o = self.almacen.filas("SELECT * FROM observaciones")
                self.assertEqual(len(o), indice + 1)
                self.assertEqual(len(self.almacen.filas("SELECT * FROM acciones")), indice)
                self.assertTrue(all(f["decision"] == "presente" for f in o))
                self.assertIn(f"{anio} · Presencia clara", visor.aviso.cget("text"))
            self.assertTrue(visor.lote_terminado)
            visor.dibujar()
            self.assertTrue(visor.botones_decision["presente"].instate(["disabled"]))
            visor.decidir("presente")
            self.assertEqual(len(self.almacen.filas("SELECT * FROM acciones")), 3)
            visor.deshacer()
            self.assertEqual(int(visor.anio.get()), 2014)
            self.assertFalse(visor.lote_terminado)
            self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))

    def test_contexto_en_segundo_plano_conserva_caso_anio_y_notas(self):
        liberar = Event()

        def ampliar(almacen, config, sitio, progreso):
            progreso("Prueba de contexto")
            liberar.wait(3)
            with almacen.db:
                almacen.db.execute(
                    "INSERT INTO ventanas VALUES ('contexto',0,-76.8,76.8,0,'contexto_visual')"
                )
                almacen.db.execute("INSERT INTO contextos VALUES ('a','contexto')")
                for c in config["campanias"]:
                    almacen.db.execute(
                        "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                        (
                            "contexto",
                            c["anio"],
                            str(self.carpeta / f"{c['anio']}.png"),
                            "",
                            "prueba",
                            "",
                            0.15,
                            "",
                        ),
                    )

        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.ampliar_contexto",
                side_effect=ampliar,
            ),
            self.visor_preparado() as visor,
        ):
            visor.mostrar("a", 2014)
            visor.notas.set("Nota de la campaña")
            visor.contexto()
            try:
                respuesta = []
                visor.raiz.after(0, lambda: respuesta.append(True))
                visor.raiz.update()
                self.assertTrue(respuesta, "El visor quedó bloqueado durante la carga")
                self.assertTrue(visor.preparando)
            finally:
                liberar.set()
            self.esperar_preparacion(visor)
            self.assertEqual((visor.actual, visor.anio.get()), ("a", "2014"))
            self.assertEqual(visor.notas.get(), "Nota de la campaña")
            self.assertEqual(len(visor._recursos_sitio()), 8)
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))

    def test_contexto_fallido_no_avanza_y_permite_reintentar(self):
        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.ampliar_contexto",
                side_effect=OSError("Corte"),
            ),
            patch(
                "paneles_solares.visualizacion.revisar_seguimiento.messagebox.showerror"
            ) as error,
            self.visor_preparado() as visor,
        ):
            visor.mostrar("a", 2017)
            visor.contexto()
            self.esperar_preparacion(visor)
            error.assert_called_once()
            self.assertEqual((visor.actual, visor.anio.get()), ("a", "2017"))
            self.assertFalse(visor.boton_contexto.instate(["disabled"]))
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])

    def test_candidato_omitido_se_anade_con_click_y_reutiliza_fotos_con_zoom(self):
        geo = SimpleNamespace(describir=lambda _: {"municipio": "Prueba", "codigo_municipal": "x"})
        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.GeografiaSeguimiento",
                return_value=geo,
            ),
            self.visor_preparado() as visor,
        ):
            visor.mostrar("a", 2014)
            visor.zoom, visor.centro = 2.5, [0.42, 0.65]
            visor.dibujar()
            visor.anadir()
            visor.decidir("presente")
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            panel = visor.paneles[1]
            ancho, alto = max(panel.winfo_width(), 100), max(panel.winfo_height(), 100)
            lado = min(ancho, alto - 30)
            evento = SimpleNamespace(
                widget=panel,
                x=(ancho - lado) / 2 + lado * 0.7,
                y=(alto + 30 - lado) / 2 + lado * 0.6,
            )
            visor.iniciar_arrastre(evento)
            visor.elegir_objetivo(evento)
            nuevo = self.almacen.filas(
                "SELECT * FROM emplazamientos WHERE origen='candidato_manual'"
            )[0]
            from shapely import from_wkb

            punto = from_wkb(nuevo["geometria"]).centroid
            self.assertAlmostEqual(punto.x, (0.42 + 0.2 / 2.5) * 76.8)
            self.assertAlmostEqual(punto.y, (1 - 0.65 - 0.1 / 2.5) * 76.8)
            self.assertEqual((visor.actual, visor.anio.get()), (nuevo["id"], "2014"))
            self.assertEqual(nuevo["ventana"], "v")
            self.assertEqual(len(self.almacen.filas("SELECT * FROM ventanas")), 1)
            self.assertEqual(len(visor._recursos_sitio()), 4)
            self.assertEqual(visor.estado_referencia.get(), "Dudoso")
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            self.assertFalse(visor.seleccionando_candidato)

    def test_cancelar_candidato_no_crea_casos_ni_respuestas(self):
        with self.visor_preparado() as visor:
            visor.anadir()
            self.assertTrue(visor.botones_decision["presente"].instate(["disabled"]))
            visor.cancelar_candidato()
            self.assertFalse(visor.botones_decision["presente"].instate(["disabled"]))
            self.assertEqual(len(self.almacen.filas("SELECT * FROM emplazamientos")), 1)
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])

    def test_caso_provisional_no_propone_presencia_2023_sin_localizar_paneles(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "UPDATE emplazamientos SET origen='inventario_2023' WHERE id='a'"
            )
        with self.visor_preparado() as visor:
            self.assertIsNone(visor.actual)
            visor.incluir_provisionales.set(True)
            visor.recargar()
            visor.siguiente(cargar=False)
            self.assertEqual(visor.estado_referencia.get(), "Dudoso")
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            visor.decidir("dudosa")
            self.assertTrue(
                all(
                    o["decision"] == "dudosa"
                    for o in self.almacen.filas("SELECT * FROM observaciones")
                )
            )

    def test_click_para_anadir_sobre_un_caso_existente_evitar_duplicarlo(self):
        with self.visor_preparado() as visor:
            visor.anadir()
            panel = visor.paneles[0]
            ancho, alto = max(panel.winfo_width(), 100), max(panel.winfo_height(), 100)
            lado = min(ancho, alto - 30)
            evento = SimpleNamespace(
                widget=panel,
                x=(ancho - lado) / 2 + lado * 15 / 76.8,
                y=(alto + 30 - lado) / 2 + lado * (1 - 15 / 76.8),
            )
            visor.iniciar_arrastre(evento)
            visor.elegir_objetivo(evento)
            self.assertEqual(visor.actual, "a")
            self.assertFalse(visor.seleccionando_candidato)
            self.assertEqual(len(self.almacen.filas("SELECT * FROM emplazamientos")), 1)
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])

    def test_aviso_antiguo_de_edificio_cortado_no_impone_alcance_parcial_a_paneles(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "UPDATE emplazamientos SET origen='piloto',etiquetas='[\"borde_recorte\"]' WHERE id='a'"
            )
            self.almacen.db.execute(
                "INSERT INTO componentes(id,ventana,anio,geometria,area,borde) VALUES ('panel','v',2023,?,100,0)",
                (box(10, 10, 20, 20).wkb,),
            )
            self.almacen.db.execute("INSERT INTO miembros VALUES ('panel','a')")
        with self.visor_preparado() as visor:
            self.assertEqual(visor.alcance.get(), "emplazamiento_completo")
            visor.alcance.set("zona_visible")
            visor.decidir("ausente")
            visor.mostrar("a", 2020)
            self.assertEqual(visor.alcance.get(), "zona_visible")
            original = self.almacen.filas("SELECT alcance FROM observaciones WHERE anio=2020")[0]
            self.assertEqual(original["alcance"], "zona_visible")

    def test_referencia_falsa_con_presencia_historica_senala_retirada_para_revision(self):
        observaciones = [
            {"anio": a, "decision": d, "alcance": "emplazamiento_completo"}
            for a, d in (
                (2014, "ausente"),
                (2017, "presente"),
                (2020, "presente"),
                (2023, "referencia_incorrecta"),
            )
        ]
        historia = derivar_historia(observaciones, [2014, 2017, 2020, 2023])
        self.assertTrue(historia["secuencia_anomala"])
        self.assertEqual(historia["primera_presencia"], 2017)
        self.assertEqual(historia["estado_historia"], "secuencia_anomala")
        self.assertIsNone(historia["aparicion_despues_de"])

    def test_grupo_con_partes_sin_foto_historica_no_propone_ausencia_completa(self):
        imagen = self.carpeta / "vecina_2023.png"
        Image.new("RGB", (32, 32)).save(imagen)
        with self.almacen.db:
            self.almacen.db.execute("INSERT INTO ventanas VALUES ('w',76.8,0,153.6,76.8,'prueba')")
            self.almacen.db.execute(
                "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                ("w", 2023, str(imagen), "", "prueba", "", 0.15, ""),
            )
            for cid, ventana, geometria in (
                ("panel1", "v", box(10, 10, 20, 20)),
                ("panel2", "w", box(100, 10, 110, 20)),
            ):
                self.almacen.db.execute(
                    "INSERT INTO componentes(id,ventana,anio,geometria,area,borde) VALUES (?,?,2023,?,100,0)",
                    (cid, ventana, geometria.wkb),
                )
                self.almacen.db.execute("INSERT INTO miembros VALUES (?,'a')", (cid,))
        with self.visor_preparado() as visor:
            self.assertEqual(visor.alcance.get(), "zona_visible")
            visor.decidir("ausente")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2020")[0]
            self.assertEqual(o["alcance"], "zona_visible")
            historia = derivar_historia(
                self.almacen.filas("SELECT * FROM observaciones"), [2020, 2023]
            )
            self.assertIsNone(historia["aparicion_despues_de"])

    def test_siguiente_salta_la_campania_sin_inventar_respuesta(self):
        with self.visor_preparado() as visor:
            self.assertEqual(visor.anio.get(), "2020")
            visor.siguiente()
            self.assertEqual(visor.anio.get(), "2017")
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            self.assertFalse(visor.botones_decision["referencia_incorrecta"].instate(["disabled"]))

    def test_filtro_pendiente_no_requiere_repetir_la_respuesta(self):
        with self.visor_preparado() as visor:
            visor.filtros["Estado"].set("Pendiente")
            visor.recargar()
            for anio in (2020, 2017, 2014):
                self.assertEqual(visor.anio.get(), str(anio))
                visor.botones_decision["presente"].invoke()
                visor.raiz.update()
            self.assertEqual(len(self.almacen.filas("SELECT * FROM acciones")), 3)
            self.assertTrue(visor.lote_terminado)

    def test_cambio_del_par_se_guarda_en_2023_sin_repetir_presencia(self):
        self.almacen.guardar_observacion("a", 2023, "presente", notas="Nota previa")
        with self.visor_preparado() as visor:
            visor.notas.set("Nota del histórico")
            visor.marcar_cambio_referencia("ampliacion")
            self.assertEqual(visor.anio.get(), "2020")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertEqual((o["anio"], o["decision"]), (2023, "presente"))
            self.assertIn("ampliacion", json.loads(o["etiquetas"]))
            self.assertIn("Nota previa", o["notas"])
            self.assertIn("Comparación 2020→2023", o["notas"])
            self.assertEqual(visor.notas.get(), "Nota del histórico")
            visor.botones_decision["presente"].invoke()
            o2020 = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2020")[0]
            self.assertEqual(o2020["notas"], "Nota del histórico")
            self.assertNotIn("ampliacion", json.loads(o2020["etiquetas"]))
            visor.deshacer()
            visor.deshacer()
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertEqual(o["decision"], "presente")
            self.assertNotIn("ampliacion", json.loads(o["etiquetas"]))

    def test_cambio_no_clasifica_automaticamente_la_referencia_pendiente(self):
        with self.visor_preparado() as visor:
            visor.mostrar("a", 2017)
            visor.marcar_cambio_referencia("reduccion")
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            borrador = self.almacen.meta("borrador_a_2023")
            self.assertIn("reduccion", borrador["etiquetas"])
            self.assertIn("Comparación 2017→2023", borrador["notas"])
            visor.decidir("presente")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertIn("reduccion", json.loads(o["etiquetas"]))

    def test_dos_sitios_en_orden_y_zoom_conservado_entre_campanias(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('b','v',?,'prueba')",
                (box(30, 30, 40, 40).wkb,),
            )
        with self.visor_preparado() as visor:
            visor.zoom, visor.centro = 2.5, [0.35, 0.42]
            for sitio in ("a", "b"):
                for anio in (2020, 2017, 2014):
                    self.assertEqual((visor.actual, int(visor.anio.get())), (sitio, anio))
                    if sitio == "a":
                        self.assertEqual((visor.zoom, visor.centro), (2.5, [0.35, 0.42]))
                    else:
                        self.assertEqual((visor.zoom, visor.centro), (1.0, [0.5, 0.5]))
                    visor.decidir("presente")
                    visor.raiz.update()
            self.assertEqual(len(self.almacen.filas("SELECT * FROM observaciones")), 8)
            self.assertEqual(len(self.almacen.filas("SELECT * FROM acciones")), 6)
            self.assertTrue(visor.lote_terminado)

    def test_falso_positivo_se_corrige_sin_cambiar_el_historico(self):
        with self.visor_preparado() as visor:
            visor.corregir_referencia("referencia_incorrecta")
            self.assertEqual(visor.anio.get(), "2020")
            o = self.almacen.filas("SELECT * FROM observaciones")
            self.assertEqual(len(o), 1)
            self.assertEqual((o[0]["anio"], o[0]["decision"]), (2023, "referencia_incorrecta"))
            visor.decidir("presente")
            self.assertEqual(visor.anio.get(), "2017")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertEqual(o["decision"], "referencia_incorrecta")
            visor.estado_referencia.set("Dudoso")
            visor.corregir_referencia()
            self.assertEqual(visor.anio.get(), "2017")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertEqual(o["decision"], "dudosa")

    def test_ninguno_en_ambas_descarta_referencia_y_avanza_en_tres_clicks(self):
        with self.visor_preparado() as visor:
            visor.botones_decision["referencia_incorrecta"].invoke()
            self.assertEqual(visor.anio.get(), "2017")
            observaciones = self.almacen.filas("SELECT * FROM observaciones")
            self.assertEqual(len(observaciones), 2)
            self.assertEqual(
                {o["anio"]: o["decision"] for o in observaciones},
                {2023: "referencia_incorrecta", 2020: "ausente"},
            )
            visor.filtros["Estado"].set("referencia_incorrecta")
            visor.recargar()
            self.assertEqual(visor.visibles, ["a"])
            visor.filtros["Estado"].set("Todos")
            visor.recargar()
            visor.decidir("ausente")
            visor.decidir("ausente")
            self.assertEqual(len(self.almacen.filas("SELECT * FROM acciones")), 3)
            self.assertTrue(visor.lote_terminado)

    def test_ninguno_en_ambas_corrige_y_deshace_preservando_importadas(self):
        self.almacen.guardar_observacion(
            "a",
            2023,
            "presente",
            original={"anio": 2023, "decision": "presente"},
            notas="2023 original",
        )
        self.almacen.guardar_observacion("a", 2020, "presente", notas="2020 original")
        with self.visor_preparado() as visor:
            visor.mostrar("a", 2020)
            visor.decidir("referencia_incorrecta")
            self.assertEqual(visor.anio.get(), "2017")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertEqual(o["decision"], "referencia_incorrecta")
            self.assertEqual(json.loads(o["original"])["decision"], "presente")
            visor.deshacer()
            observaciones = self.almacen.filas("SELECT * FROM observaciones")
            self.assertTrue(all(o["decision"] == "presente" for o in observaciones))
            self.assertEqual(
                {o["notas"] for o in observaciones}, {"2023 original", "2020 original"}
            )

    def test_primera_comparacion_se_deshace_junta_y_no_sobrescribe_2023(self):
        with self.visor_preparado() as visor:
            visor.decidir("ausente")
            visor.deshacer()
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            self.assertEqual(visor.anio.get(), "2020")
            self.almacen.guardar_observacion(
                "a", 2023, "ausente", original={"decision": "ausente"}, notas="Original"
            )
            visor.recargar()
            visor.dibujar()
            visor.decidir("presente")
            o = self.almacen.filas("SELECT * FROM observaciones WHERE anio=2023")[0]
            self.assertEqual((o["decision"], o["notas"]), ("ausente", "Original"))
            self.assertEqual(json.loads(o["original"]), {"decision": "ausente"})

    def test_comparacion_atomica_no_deja_media_respuesta_tras_un_error(self):
        referencia = {"decision": "presente", "alcance": "emplazamiento_completo"}
        escribir = self.almacen._escribir_observacion

        def fallar_en_historico(o):
            if o["anio"] == 2020:
                raise OSError("Corte")
            escribir(o)

        with (
            patch.object(
                self.almacen,
                "_escribir_observacion",
                side_effect=fallar_en_historico,
            ),
            self.assertRaises(OSError),
        ):
            self.almacen.guardar_observacion("a", 2020, "ausente", referencia=referencia)
        self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
        self.assertEqual(self.almacen.filas("SELECT * FROM acciones"), [])

    def test_cursor_antiguo_2023_reanuda_en_historico_pendiente(self):
        self.almacen.poner_meta("cursor_revision", {"sitio": "a", "anio": 2023})
        self.almacen.guardar_observacion("a", 2020, "ausente")
        with self.visor_preparado() as visor:
            self.assertEqual((visor.actual, visor.anio.get()), ("a", "2017"))

    def test_cambio_manual_de_anio_conserva_zoom_y_no_mezcla_notas(self):
        with self.visor_preparado() as visor:
            visor.notas.set("Nota exclusiva de 2020")
            visor.guardar_borrador()
            visor.zoom, visor.centro = 2.0, [0.4, 0.6]
            visor.anio.set("2017")
            visor.cambiar_anio()
            self.assertEqual(visor.notas.get(), "")
            self.assertEqual((visor.zoom, visor.centro), (2.0, [0.4, 0.6]))
            visor.anio.set("2020")
            visor.cambiar_anio()
            self.assertEqual(visor.notas.get(), "Nota exclusiva de 2020")

    def test_resumen_separa_tres_historicos_y_referencia_integrada(self):
        self.almacen.guardar_observacion("a", 2020, "ausente", 12)
        resumen = generar_resumen(self.almacen, self.carpeta / "resumen")
        self.assertEqual(resumen["historicos_pendientes"], 2)
        self.assertEqual(resumen["referencias_2023_pendientes"], 1)
        self.assertEqual(resumen["porcentaje_historico_completado"], 33.3)
        self.assertEqual(resumen["segundos_medios_por_decision"], 12)

    def test_objetivo_visual_usa_componentes_sin_cambiar_geometria_original(self):
        from paneles_solares.visualizacion.revisar_seguimiento import objetivos_visuales

        original = self.almacen.filas("SELECT geometria FROM emplazamientos WHERE id='a'")[0]
        paneles = box(50, 2, 57, 8)
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO componentes(id,ventana,anio,geometria,area) VALUES ('panel','v',2023,?,?)",
                (paneles.wkb, paneles.area),
            )
            self.almacen.db.execute("INSERT INTO miembros VALUES ('panel','a')")
        objetivo = objetivos_visuales(self.almacen, "a", ["v"])[0]
        self.assertTrue(objetivo["geometria"].equals(paneles))
        self.assertFalse(objetivo["provisional"])
        self.assertEqual(
            original, self.almacen.filas("SELECT geometria FROM emplazamientos WHERE id='a'")[0]
        )
        self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])

    def test_marco_objetivo_sigue_visible_sin_capas_y_en_ambas_campanias(self):
        with self.visor_preparado() as visor:
            visor.capas["Agrupación"].set(False)
            visor.capas["Predicción"].set(False)
            visor.dibujar()
            for anio in (2020, 2017, 2014):
                visor.mostrar("a", anio)
                izquierdo = visor.marcas_objetivos[visor.paneles[0]]
                derecho = visor.marcas_objetivos[visor.paneles[1]]
                self.assertEqual(izquierdo[0]["rectangulo"], derecho[0]["rectangulo"])
                self.assertEqual(izquierdo[0]["id"], "a")
                for panel in visor.paneles:
                    self.assertTrue(panel.find_withtag("objetivo"))
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])

    def test_click_en_otro_caso_cambia_objetivo_sin_copiar_respuestas(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('b','v',?,'prueba')",
                (box(50, 10, 60, 20).wkb,),
            )
        with self.visor_preparado() as visor:
            panel = visor.paneles[0]
            marca = next(m for m in visor.marcas_objetivos[panel] if m["id"] == "b")
            x0, y0, x1, y1 = marca["rectangulo"]
            evento = SimpleNamespace(widget=panel, x=(x0 + x1) / 2, y=(y0 + y1) / 2)
            visor.iniciar_arrastre(evento)
            visor.elegir_objetivo(evento)
            self.assertEqual((visor.actual, visor.anio.get()), ("b", "2020"))
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
            visor.decidir("presente")
            self.assertEqual(visor.anio.get(), "2017")
            self.assertTrue(
                all(
                    o["emplazamiento"] == "b"
                    for o in self.almacen.filas("SELECT * FROM observaciones")
                )
            )

    def test_arrastrar_la_foto_no_selecciona_un_vecino(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO emplazamientos(id,ventana,geometria,origen) VALUES ('b','v',?,'prueba')",
                (box(50, 10, 60, 20).wkb,),
            )
        with self.visor_preparado() as visor:
            panel = visor.paneles[0]
            marca = next(m for m in visor.marcas_objetivos[panel] if m["id"] == "b")
            x0, y0, x1, y1 = marca["rectangulo"]
            evento = SimpleNamespace(widget=panel, x=(x0 + x1) / 2, y=(y0 + y1) / 2)
            visor.iniciar_arrastre(evento)
            movimiento = SimpleNamespace(widget=panel, x=evento.x + 12, y=evento.y)
            visor.desplazar(movimiento)
            visor.elegir_objetivo(movimiento)
            self.assertEqual(visor.actual, "a")
            self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])

    def test_centrar_recupera_un_objetivo_fuera_de_vista(self):
        with self.visor_preparado() as visor:
            visor.zoom, visor.centro = 8.0, [0.98, 0.01]
            visor.dibujar()
            self.assertFalse(
                any(m["seleccionado"] for m in visor.marcas_objetivos[visor.paneles[0]])
            )
            visor.centrar_objetivo()
            self.assertTrue(
                any(m["seleccionado"] for m in visor.marcas_objetivos[visor.paneles[0]])
            )

    def test_guardado_sobrevive_a_cierre_abrupto(self):
        codigo = "from paneles_solares.datos.seguimiento import AlmacenSeguimiento; import os; s=AlmacenSeguimiento(__import__('pathlib').Path(__import__('sys').argv[1])); s.guardar_observacion('a',2023,'presente',4); os._exit(0)"
        subprocess.run([sys.executable, "-c", codigo, str(self.almacen.ruta)], check=True)
        o = self.almacen.filas("SELECT * FROM observaciones")
        self.assertEqual(o[0]["decision"], "presente")
        self.assertEqual(self.almacen.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_correccion_deshacer_y_reapertura_no_borran_original(self):
        self.almacen.guardar_observacion(
            "a", 2020, "presente", 4, original={"decision": "presente", "anio": 2020}
        )
        self.almacen.guardar_observacion(
            "a", 2020, "dudosa", 7, etiquetas=["nueva_etiqueta"], notas="Sombra"
        )
        self.almacen.cerrar()
        self.almacen = AlmacenSeguimiento(self.carpeta / "seguimiento.sqlite")
        self.assertEqual(
            self.almacen.filas("SELECT decision FROM observaciones")[0]["decision"], "dudosa"
        )
        self.almacen.deshacer()
        o = self.almacen.filas("SELECT * FROM observaciones")[0]
        self.assertEqual(o["decision"], "presente")
        self.assertEqual(json.loads(o["original"])["decision"], "presente")
        self.assertEqual(len(self.almacen.filas("SELECT * FROM acciones")), 2)

    def test_soportes_y_asociacion_no_cambian_cronologia(self):
        self.almacen.guardar_observacion(
            "a", 2020, "presente", etiquetas=["ampliacion", "edificio_nuevo"]
        )
        for soporte in (
            "suelo",
            "marquesina",
            "construccion_no_catastrada",
            "mixto",
            "cubierta_catastrada",
        ):
            self.almacen.modificar_sitio(
                "a",
                soporte=soporte,
                relacion="exterior",
                edificios=[],
                cubiertas=["cubierta_manual_jardin"],
            )
        self.assertEqual(
            self.almacen.filas("SELECT decision FROM observaciones")[0]["decision"], "presente"
        )
        self.assertEqual(
            json.loads(self.almacen.filas("SELECT edificios FROM emplazamientos")[0]["edificios"]),
            [],
        )

    def test_intervalos_solo_con_observaciones_no_con_predicciones(self):
        anios = [2014, 2017, 2020, 2023]

        def obs(secuencia):
            return [
                {"anio": a, "decision": d, "alcance": "emplazamiento_completo"}
                for a, d in zip(anios, secuencia)
            ]

        resultado = derivar_historia(obs(["ausente", "ausente", "presente", "presente"]), anios)
        self.assertEqual(
            (resultado["aparicion_despues_de"], resultado["aparicion_hasta"]), (2017, 2020)
        )
        resultado = derivar_historia(obs(["presente", "ausente", "presente", "presente"]), anios)
        self.assertTrue(resultado["secuencia_anomala"])
        self.assertIsNone(resultado["aparicion_despues_de"])
        self.assertEqual(
            derivar_historia(obs(["ausente", "no_evaluable", "presente", "presente"]), anios)[
                "estado_historia"
            ],
            "primera_presencia_intervalo_no_resuelto",
        )
        self.assertEqual(
            derivar_historia(obs(["ausente"] * 4), anios)["estado_historia"],
            "sin_presencia_observada",
        )
        self.assertEqual(
            derivar_historia(
                obs(["ausente", "ausente", "ausente", "referencia_incorrecta"]), anios
            )["estado_historia"],
            "referencia_descartada",
        )
        parciales = obs(["ausente", "ausente", "presente", "presente"])
        parciales[1]["alcance"] = "zona_visible"
        self.assertIsNone(derivar_historia(parciales, anios)["aparicion_despues_de"])

    def test_componentes_exteriores_pequenos_y_bordes_se_conservan(self):
        geo = GeografiaSeguimiento.__new__(GeografiaSeguimiento)
        geo.edificios = gpd.GeoDataFrame(geometry=[box(20, 20, 30, 30)], crs="EPSG:25830")
        geo.municipios = gpd.GeoDataFrame(
            [{"codigo": "x", "municipio": "Prueba", "geometry": box(0, 0, 200, 200)}],
            crs="EPSG:25830",
        )
        geo.proximidad = 1.3
        p = np.zeros((512, 512), dtype="float32")
        p[0:10, 0:10] = 0.9
        p[40, 40] = 0.9
        ventana = {"id": "v", "xmin": 0, "ymin": 0, "xmax": 76.8, "ymax": 76.8}
        registrar_componentes(self.almacen, ventana, 2023, p, "modelo", geo, "prueba")
        c = self.almacen.filas("SELECT * FROM componentes")
        self.assertEqual(len(c), 2)
        self.assertAlmostEqual(sum(f["area"] for f in c), 101 * 0.15**2)
        self.assertEqual(sum(f["borde"] for f in c), 1)
        self.assertTrue(all(f["fraccion"] == 0 for f in c))
        geometria = c[0]["geometria"]
        registrar_componentes(self.almacen, ventana, 2023, p, "modelo", geo, "prueba")
        self.assertEqual(len(self.almacen.filas("SELECT * FROM componentes")), 2)
        self.assertEqual(geometria, self.almacen.filas("SELECT * FROM componentes")[0]["geometria"])

    def test_relacion_desplazada_y_proxima_no_afirma_paneles(self):
        geo = GeografiaSeguimiento.__new__(GeografiaSeguimiento)
        geo.edificios = gpd.GeoDataFrame(geometry=[box(20, 20, 30, 30)], crs="EPSG:25830")
        geo.municipios = gpd.GeoDataFrame(
            [{"codigo": "x", "municipio": "Prueba", "geometry": box(0, 0, 100, 100)}],
            crs="EPSG:25830",
        )
        geo.proximidad = 1.3
        self.assertEqual(geo.describir(box(29, 20, 32, 25))["relacion"], "parcial_desplazado")
        cercano = geo.describir(box(30.5, 20, 33, 25))
        self.assertEqual(cercano["relacion"], "proximo")
        self.assertEqual(cercano["edificios"], [])
        self.assertEqual(len(self.almacen.filas("SELECT * FROM observaciones")), 0)

    def test_reagrupacion_preserva_revisiones_y_originales(self):
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO componentes(id,ventana,anio,geometria,area,municipio,codigo_municipal) VALUES ('c1','v',2023,?,4,'Prueba','x')",
                (box(10, 10, 12, 12).wkb,),
            )
            self.almacen.db.execute(
                "INSERT INTO componentes(id,ventana,anio,geometria,area,municipio,codigo_municipal) VALUES ('c2','v',2023,?,4,'Prueba','x')",
                (box(12, 10, 14, 12).wkb,),
            )
            self.almacen.db.executemany("INSERT INTO miembros VALUES (?,'a')", [("c1",), ("c2",)])
        self.almacen.guardar_observacion("a", 2020, "presente", etiquetas=["ampliacion"])
        nuevos = self.almacen.reorganizar(["a"], [["c1"], ["c2"]])
        self.assertEqual(len(nuevos), 2)
        self.assertEqual(
            self.almacen.filas("SELECT decision FROM observaciones")[0]["decision"], "presente"
        )
        self.assertEqual(
            self.almacen.filas("SELECT activo FROM emplazamientos WHERE id='a'")[0]["activo"], 0
        )
        self.almacen.deshacer()
        self.assertTrue(
            all(m["emplazamiento"] == "a" for m in self.almacen.filas("SELECT * FROM miembros"))
        )
        self.assertEqual(
            self.almacen.filas("SELECT activo FROM emplazamientos WHERE id='a'")[0]["activo"], 1
        )

    def test_preparacion_independiente_reutiliza_imagenes_tras_ausencia(self):
        config = cargar_config()
        self.almacen.poner_meta("piloto_importado", True)
        self.almacen.guardar_observacion("a", 2023, "referencia_incorrecta")
        self.almacen.guardar_observacion("a", 2020, "ausente")
        pesos = self.carpeta / "modelo.pt"
        pesos.write_bytes(b"modelo_de_prueba")
        firma = hashlib.sha256(pesos.read_bytes()).hexdigest()
        for anio in (2023, 2020):
            im = self.carpeta / f"{anio}.png"
            Image.new("RGB", (512, 512)).save(im)
            pred = self.carpeta / f"pred_{anio}.png"
            Image.fromarray(np.zeros((512, 512), dtype="uint16")).save(pred)
            with self.almacen.db:
                self.almacen.db.execute(
                    "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                    ("v", anio, str(im), str(pred), "prueba", "", 0.15, firma),
                )
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.PESOS_UNET", pesos),
            patch("paneles_solares.etiquetado.preparar_seguimiento.GeografiaSeguimiento"),
            patch("paneles_solares.etiquetado.preparar_seguimiento.comprobar_wcs"),
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.extraer_wcs",
                return_value=(Image.new("RGB", (307, 307)), "prueba"),
            ) as descargar,
            patch("paneles_solares.etiquetado.preparar_seguimiento.cargar_modelo"),
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.predecir",
                return_value=(np.zeros((512, 512)), None),
            ),
        ):
            preparar_lote(self.almacen, config, 1)
            self.assertEqual(descargar.call_count, 2)
            preparar_lote(self.almacen, config, 1)
            self.assertEqual(descargar.call_count, 2)
        self.assertEqual(len(self.almacen.filas("SELECT * FROM observaciones")), 2)

    def test_precarga_no_descarga_historicos_de_zona_provisional_sin_detecciones(self):
        self.almacen.poner_meta("piloto_importado", True)
        with self.almacen.db:
            self.almacen.db.execute(
                "UPDATE emplazamientos SET origen='inventario_2023' WHERE id='a'"
            )
        pesos = self.carpeta / "modelo.pt"
        pesos.write_bytes(b"modelo_de_prueba")
        completas = []
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.PESOS_UNET", pesos),
            patch("paneles_solares.etiquetado.preparar_seguimiento.GeografiaSeguimiento"),
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.leer_referencia_local",
                return_value=(Image.new("RGB", (512, 512)), "prueba"),
            ),
            patch("paneles_solares.etiquetado.preparar_seguimiento.extraer_wcs") as descargar,
            patch("paneles_solares.etiquetado.preparar_seguimiento.cargar_modelo"),
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.predecir",
                return_value=(np.zeros((512, 512)), None),
            ),
        ):
            preparar_lote(
                self.almacen,
                cargar_config(),
                1,
                solo_localizados=True,
                al_completar=completas.append,
            )
        descargar.assert_not_called()
        self.assertEqual(completas, ["v"])
        self.assertEqual([r["anio"] for r in self.almacen.filas("SELECT * FROM recursos")], [2023])
        self.assertEqual(self.almacen.filas("SELECT * FROM observaciones"), [])
        import tkinter as tk

        from paneles_solares.visualizacion.revisar_seguimiento import VisorSeguimiento

        raiz = tk.Tk()
        raiz.withdraw()
        try:
            with patch(
                "paneles_solares.visualizacion.revisar_seguimiento.gpd.read_file",
                return_value=gpd.GeoDataFrame(geometry=[], crs="EPSG:25830"),
            ):
                visor = VisorSeguimiento(raiz, self.almacen, {**cargar_config(), "precarga": False})
            self.assertEqual(
                visor.sitios_sin_fotos(), []
            )  # No se reintenta un negativo ya procesado.
            self.assertEqual(visor.cola, [])
            self.assertEqual(visor.sitios["a"]["activo"], 1)
        finally:
            raiz.destroy()

    def test_registrar_predicciones_no_bloquea_el_guardado_humano_durante_geografia(self):
        segundo, liberar = Event(), Event()
        errores = []
        llamadas = 0

        def describir(geometria):
            nonlocal llamadas
            llamadas += 1
            if llamadas == 2:
                segundo.set()
                liberar.wait(5)
            return {
                "relacion": "exterior",
                "edificios": [],
                "cubiertas": [],
                "fraccion": 0,
                "distancia": 50,
                "municipio": "Prueba",
                "codigo_municipal": "x",
            }

        def registrar():
            almacen = AlmacenSeguimiento(self.almacen.ruta)
            try:
                ventana = almacen.filas("SELECT * FROM ventanas WHERE id='v'")[0]
                probabilidades = np.zeros((512, 512))
                probabilidades[50:70, 50:70] = 0.9
                probabilidades[200:220, 200:220] = 0.9
                registrar_componentes(
                    almacen,
                    ventana,
                    2023,
                    probabilidades,
                    "modelo",
                    SimpleNamespace(describir=describir),
                    "prueba",
                )
            except Exception as error:  # noqa: BLE001 -- Enviar los fallos del hilo a la prueba.
                errores.append(error)
            finally:
                almacen.cerrar()

        self.almacen.db.execute("PRAGMA busy_timeout=500")
        hilo = Thread(target=registrar)
        hilo.start()
        try:
            self.assertTrue(segundo.wait(5))
            self.almacen.guardar_observacion(
                "a", 2020, "presente", notas="Guardado mientras se prepara"
            )
            self.assertEqual(
                self.almacen.filas("SELECT notas FROM observaciones")[0]["notas"],
                "Guardado mientras se prepara",
            )
            self.assertEqual(len(self.almacen.filas("SELECT * FROM componentes")), 1)
        finally:
            liberar.set()
            hilo.join(timeout=5)
        self.assertEqual(errores, [])
        self.assertFalse(hilo.is_alive())
        self.assertEqual(len(self.almacen.filas("SELECT * FROM componentes")), 2)

    def test_preparacion_recupera_foto_perdida_sin_repetir_prediccion(self):
        config = cargar_config()
        self.almacen.poner_meta("piloto_importado", True)
        pesos = self.carpeta / "modelo.pt"
        pesos.write_bytes(b"modelo_de_prueba")
        firma = hashlib.sha256(pesos.read_bytes()).hexdigest()
        for anio in (2023, 2020, 2017, 2014):
            imagen = self.carpeta / f"{anio}.png"
            if anio != 2014:
                Image.new("RGB", (32, 32)).save(imagen)
            prediccion = self.carpeta / f"pred_{anio}.png"
            Image.fromarray(np.zeros((32, 32), dtype="uint16")).save(prediccion)
            with self.almacen.db:
                self.almacen.db.execute(
                    "INSERT INTO recursos VALUES (?,?,?,?,?,?,?,?)",
                    ("v", anio, str(imagen), str(prediccion), "prueba", "", 0.15, firma),
                )
        self.almacen.guardar_observacion("a", 2020, "ausente", notas="Conservar")
        original = self.almacen.filas("SELECT * FROM observaciones")
        pred_original = (self.carpeta / "pred_2014.png").read_bytes()
        avances = []
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.PESOS_UNET", pesos),
            patch("paneles_solares.etiquetado.preparar_seguimiento.GeografiaSeguimiento"),
            patch("paneles_solares.etiquetado.preparar_seguimiento.comprobar_wcs"),
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.extraer_wcs",
                return_value=(Image.new("RGB", (32, 32)), "prueba"),
            ) as descargar,
            patch("paneles_solares.etiquetado.preparar_seguimiento.predecir") as predecir,
        ):
            preparar_lote(self.almacen, config, 1, progreso=avances.append)
            descargar.assert_called_once()
            predecir.assert_not_called()
        self.assertTrue((self.carpeta / "2014.png").exists())
        self.assertEqual((self.carpeta / "pred_2014.png").read_bytes(), pred_original)
        self.assertEqual(original, self.almacen.filas("SELECT * FROM observaciones"))
        self.assertIn("RGB", avances[0])
        self.assertIn("2014", avances[-1])

    def test_preparacion_predeterminada_no_vuelve_a_casos_terminados(self):
        self.almacen.poner_meta("piloto_importado", True)
        for anio in (2020, 2017, 2014):
            self.almacen.guardar_observacion("a", anio, "ausente")
        pesos = self.carpeta / "modelo.pt"
        pesos.write_bytes(b"modelo_de_prueba")
        with (
            patch("paneles_solares.etiquetado.preparar_seguimiento.PESOS_UNET", pesos),
            patch("paneles_solares.etiquetado.preparar_seguimiento.GeografiaSeguimiento"),
            patch("paneles_solares.etiquetado.preparar_seguimiento.extraer_wcs") as descargar,
            patch("paneles_solares.etiquetado.preparar_seguimiento.predecir") as predecir,
        ):
            preparar_lote(self.almacen, cargar_config(), 1)
            descargar.assert_not_called()
            predecir.assert_not_called()
        self.assertEqual(len(self.almacen.filas("SELECT * FROM observaciones")), 3)

    def test_exportacion_no_cuenta_agrupaciones_dudosas(self):
        self.almacen.guardar_observacion("a", 2023, "presente")
        s = generar_resumen(self.almacen, self.carpeta / "resultados", True)
        self.assertEqual(s["emplazamientos_validos_observados"], 0)
        self.almacen.modificar_sitio("a", agrupacion="confirmada", soporte="suelo")
        s = generar_resumen(self.almacen, self.carpeta / "resultados", True)
        self.assertEqual(s["soportes_confirmados"]["suelo"], 1)
        self.assertFalse(s["region_completa"])
        self.assertEqual(
            gpd.read_file(
                self.carpeta / "resultados/seguimiento.gpkg", layer="emplazamientos"
            ).id.iloc[0],
            "a",
        )

    def test_cubierta_sin_comprobar_catastro_conserva_cronologia_y_se_exporta(self):
        self.almacen.guardar_observacion("a", 2023, "presente")
        self.almacen.guardar_observacion("a", 2020, "ausente")
        original = self.almacen.filas("SELECT * FROM observaciones ORDER BY anio")
        self.almacen.modificar_sitio("a", soporte="cubierta", agrupacion="confirmada")
        resumen = generar_resumen(self.almacen, self.carpeta / "resumen")
        self.assertEqual(resumen["soportes_confirmados"]["cubierta"], 1)
        self.assertEqual(resumen["cubiertas_catastrales_con_fv_confirmadas"], 0)
        self.assertEqual(original, self.almacen.filas("SELECT * FROM observaciones ORDER BY anio"))

    def test_contexto_interrumpido_reanuda_sin_crear_otras_ventanas(self):
        config = cargar_config()
        with (
            patch(
                "paneles_solares.etiquetado.preparar_seguimiento.extraer_wcs",
                side_effect=[(Image.new("RGB", (512, 512)), "prueba"), OSError("Corte")],
            ),
            self.assertRaises(OSError),
        ):
            ampliar_contexto(self.almacen, config, "a")
        ventanas = self.almacen.filas("SELECT * FROM contextos")
        self.assertEqual(len(ventanas), 1)
        with patch(
            "paneles_solares.etiquetado.preparar_seguimiento.extraer_wcs",
            return_value=(Image.new("RGB", (512, 512)), "prueba"),
        ) as descargar:
            ampliar_contexto(self.almacen, config, "a")
            self.assertEqual(descargar.call_count, 3)
        self.assertEqual(self.almacen.filas("SELECT * FROM contextos"), ventanas)

    def test_agrupacion_cruza_borde_de_tesela_sin_duplicar_sitio(self):
        geo = GeografiaSeguimiento.__new__(GeografiaSeguimiento)
        geo.edificios = gpd.GeoDataFrame(geometry=[box(20, 20, 30, 30)], crs="EPSG:25830")
        geo.municipios = gpd.GeoDataFrame(
            [{"codigo": "x", "municipio": "Prueba", "geometry": box(0, 0, 200, 200)}],
            crs="EPSG:25830",
        )
        geo.proximidad = 1.3
        v1 = {"id": "v", "xmin": 0, "ymin": 0, "xmax": 76.8, "ymax": 76.8}
        v2 = {"id": "vecina", "xmin": 76.8, "ymin": 0, "xmax": 153.6, "ymax": 76.8}
        with self.almacen.db:
            self.almacen.db.execute(
                "INSERT INTO ventanas VALUES ('vecina',76.8,0,153.6,76.8,'prueba')"
            )
        a = np.zeros((512, 512))
        a[30:50, 500:512] = 0.9
        b = np.zeros((512, 512))
        b[30:50, 0:12] = 0.9
        registrar_componentes(self.almacen, v1, 2023, a, "modelo", geo, "prueba")
        registrar_componentes(self.almacen, v2, 2023, b, "modelo", geo, "prueba")
        miembros = self.almacen.filas("SELECT * FROM miembros")
        self.assertEqual(len(miembros), 2)
        self.assertEqual(len({m["emplazamiento"] for m in miembros}), 1)


class PruebasDatosReales(unittest.TestCase):
    @unittest.skipUnless(BASE.exists(), "Requiere el piloto local preparado")
    def test_depuradora_distingue_geometria_antigua_de_paneles_en_suelo(self):
        import tkinter as tk

        from paneles_solares.visualizacion.revisar_seguimiento import VisorSeguimiento

        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "seguimiento.sqlite"
            with (
                closing(sqlite3.connect(BASE)) as origen,
                closing(sqlite3.connect(ruta)) as destino,
            ):
                origen.backup(destino)
            almacen = AlmacenSeguimiento(ruta)
            raiz = tk.Tk()
            raiz.withdraw()
            try:
                visor = VisorSeguimiento(raiz, almacen, {**cargar_config(), "precarga": False})
                original = almacen.filas("SELECT * FROM observaciones ORDER BY emplazamiento,anio")
                visor.mostrar("edificio_13532_cubierta_0", 2014)
                objetivo = next(o for o in visor.objetivos if o["seleccionado"])
                self.assertTrue(objetivo["provisional"])
                self.assertEqual(objetivo["componentes"], 0)
                self.assertIn("UBICACIÓN PROVISIONAL", visor.indicacion_objetivo.cget("text"))
                panel = visor.paneles[0]
                marca = next(
                    m
                    for m in visor.marcas_objetivos[panel]
                    if m["id"] == "emplazamiento_63ca8740230cce8d14f7"
                )
                x0, y0, x1, y1 = marca["rectangulo"]
                evento = SimpleNamespace(widget=panel, x=(x0 + x1) / 2, y=(y0 + y1) / 2)
                visor.iniciar_arrastre(evento)
                visor.elegir_objetivo(evento)
                self.assertEqual(visor.actual, "emplazamiento_63ca8740230cce8d14f7")
                self.assertFalse(
                    next(o for o in visor.objetivos if o["seleccionado"])["provisional"]
                )
                self.assertEqual(
                    original,
                    almacen.filas("SELECT * FROM observaciones ORDER BY emplazamiento,anio"),
                )
            finally:
                raiz.destroy()
                almacen.cerrar()

    @unittest.skipUnless(BASE.exists(), "Requiere el piloto local preparado")
    def test_visor_real_guarda_notas_soporte_y_correccion(self):
        import tkinter as tk

        from paneles_solares.visualizacion.revisar_seguimiento import VisorSeguimiento

        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "seguimiento.sqlite"
            with (
                closing(sqlite3.connect(BASE)) as origen,
                closing(sqlite3.connect(ruta)) as destino,
            ):
                origen.backup(destino)
            s = AlmacenSeguimiento(ruta)
            raiz = tk.Tk()
            raiz.withdraw()
            try:
                visor = VisorSeguimiento(raiz, s, {**cargar_config(), "precarga": False}, 12)
                sitio = s.meta("muestra_previa_sitios")[0]
                visor.mostrar(sitio, 2020)
                visor.notas.set("Paneles en terreno: prueba aislada")
                visor.marcar_cambio_referencia("ampliacion")
                visor.soporte.set("suelo")
                visor.guardar_campos()
                visor.confirmar_grupo()
                visor.decidir("presente")
                visor.mostrar(sitio, 2020)
                visor.corregir_referencia("dudosa")
                o = s.filas(
                    "SELECT * FROM observaciones WHERE emplazamiento=? AND anio=2023", (sitio,)
                )[0]
                self.assertEqual(o["decision"], "dudosa")
                self.assertIn("Comparación 2020→2023", o["notas"])
                self.assertIn("ampliacion", json.loads(o["etiquetas"]))
                o2020 = s.filas(
                    "SELECT * FROM observaciones WHERE emplazamiento=? AND anio=2020", (sitio,)
                )[0]
                self.assertEqual(o2020["notas"], "Paneles en terreno: prueba aislada")
                self.assertEqual(
                    s.filas("SELECT soporte FROM emplazamientos WHERE id=?", (sitio,))[0][
                        "soporte"
                    ],
                    "suelo",
                )
                self.assertEqual(len(visor.fotos), 2)
            finally:
                raiz.destroy()
                s.cerrar()

    @unittest.skipUnless(BASE.exists(), "Requiere el piloto local preparado")
    def test_flujo_real_en_copia_transaccional_sin_tocar_progreso(self):
        original = BASE.parent / "revisiones.csv"
        firma = hashlib.sha256(original.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "seguimiento.sqlite"
            with (
                closing(sqlite3.connect(BASE)) as origen,
                closing(sqlite3.connect(ruta)) as destino,
            ):
                origen.backup(destino)
            s = AlmacenSeguimiento(ruta)
            importadas = s.filas("SELECT * FROM observaciones WHERE original IS NOT NULL")
            self.assertEqual(len(importadas), 44)
            c = s.filas(
                "SELECT c.*,m.emplazamiento FROM componentes c JOIN miembros m ON m.componente=c.id WHERE c.anio=2023 AND c.fraccion=0 ORDER BY c.area DESC LIMIT 1"
            )[0]
            sid = c["emplazamiento"]
            geom = c["geometria"]
            s.guardar_observacion(sid, 2023, "presente", 2, etiquetas=["desplazamiento_ortofoto"])
            s.modificar_sitio(
                sid, soporte="construccion_no_catastrada", agrupacion="confirmada", edificios=[]
            )
            s.guardar_observacion(sid, 2014, "presente")
            s.guardar_observacion(sid, 2017, "ausente")
            s.guardar_observacion(sid, 2020, "presente", etiquetas=["reinstalacion", "ampliacion"])
            s.cerrar()
            s = AlmacenSeguimiento(ruta)
            o = s.filas("SELECT * FROM observaciones WHERE emplazamiento=?", (sid,))
            self.assertTrue(derivar_historia(o, [2014, 2017, 2020, 2023])["secuencia_anomala"])
            s.guardar_observacion(sid, 2017, "no_evaluable", notas="Sombra")
            s.deshacer()
            self.assertEqual(
                s.filas("SELECT geometria FROM componentes WHERE id=?", (c["id"],))[0]["geometria"],
                geom,
            )
            for soporte in ("suelo", "marquesina", "mixto"):
                s.modificar_sitio(sid, soporte=soporte)
            generar_resumen(s, Path(carpeta) / "resultados", True)
            self.assertEqual(s.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            s.cerrar()
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), firma)


if __name__ == "__main__":
    unittest.main()
