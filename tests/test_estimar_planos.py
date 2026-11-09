"""Comprueba ajustes y apoyo con geometrías conocidas y puntos ajenos al plano."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import laspy
import numpy as np
import pandas as pd
from shapely import intersects_xy
from shapely.geometry import box

from paneles_solares.lidar.ajuste_planos import (
    CRITERIOS,
    angulo_entre_planos,
    buscar_plano,
    calcular_inclinacion,
    decidir_inclinacion,
    evaluar_distribucion,
)
from paneles_solares.lidar.estimar_planos import (
    elegir_caso,
    evaluar_apoyo,
    evaluar_entorno,
    procesar_caso,
)


class PruebasPlanos(unittest.TestCase):
    def escena(self, inclinacion=25., inclinacion_exterior=25., separacion=0.):
        interior = np.array([(x, y) for x in np.linspace(0.2, 2.8, 9)
                             for y in np.linspace(0.2, 2.8, 9)])
        exterior = np.array([(x, y) for x in np.linspace(3.2, 4.2, 8)
                             for y in np.linspace(0.2, 2.8, 9)])
        z_interior = 10 + np.tan(np.radians(inclinacion)) * (interior[:, 0] - 3) + separacion
        z_exterior = 10 + np.tan(np.radians(inclinacion_exterior)) * (exterior[:, 0] - 3)
        return np.vstack([np.column_stack([interior, z_interior]),
                          np.column_stack([exterior, z_exterior])])

    def decidir(self, contexto, ventana=None, criterios=None):
        poligono = box(0, 0, 3, 3)
        dentro = intersects_xy(poligono, contexto[:, 0], contexto[:, 1])
        return decidir_inclinacion(contexto, poligono, dentro, 0.1, 0.1, 2.,
                                   200, 42, CRITERIOS | (criterios or {}), ventana)

    def test_coplanar_con_cubierta_a_un_solo_lado(self):
        resultado = self.decidir(self.escena())

        self.assertEqual(resultado["decision"]["origen"], "estimada")
        self.assertAlmostEqual(resultado["decision"]["inclinacion_estimada"], 25., places=5)
        self.assertEqual(len(resultado["decision"]["intentos_exteriores"]), 1)

    def test_paralelo_con_separacion_pequena(self):
        resultado = self.decidir(self.escena(separacion=0.2))
        decision = resultado["decision"]

        self.assertEqual(decision["origen"], "estimada")
        self.assertAlmostEqual(decision["cubierta"]["separacion_p90_m"], 0.2 * np.cos(np.radians(25)))

    def test_outliers_lejanos_se_toleran_sin_borrarlos_del_porcentaje(self):
        contexto = self.escena()
        rng = np.random.default_rng(3)
        suelo = np.column_stack([rng.uniform(0.2, 2.8, (30, 2)), np.zeros(30)])
        pocos = self.decidir(np.vstack([contexto, suelo[:4]]))["decision"]
        muchos = self.decidir(np.vstack([contexto, suelo]))["decision"]

        self.assertEqual(pocos["origen"], "estimada")
        self.assertAlmostEqual(pocos["calidad_interior"]["porcentaje_compatible"], 100 * 81 / 85)
        self.assertEqual(muchos["motivo"], "apoyo_interior_insuficiente")
        self.assertAlmostEqual(muchos["calidad_interior"]["porcentaje_compatible"], 100 * 81 / 111)

    def test_vertientes_opuestas_no_son_paralelas(self):
        resultado = self.decidir(self.escena(inclinacion_exterior=-25.))
        decision = resultado["decision"]

        self.assertEqual(decision["motivo"], "orientaciones_distintas")
        self.assertIsNone(decision["inclinacion_estimada"])
        self.assertAlmostEqual(decision["cubierta"]["angulo_entre_planos_grados"], 50., places=5)

    def test_paralelo_demasiado_separado(self):
        resultado = self.decidir(self.escena(inclinacion=0., inclinacion_exterior=0., separacion=0.5))

        self.assertEqual(resultado["decision"]["motivo"], "superficies_separadas")

    def test_dos_superficies_proximas_son_ambiguas(self):
        contexto = self.escena(inclinacion=0., inclinacion_exterior=0.)
        otro = contexto[81:].copy()
        otro[:, 0] -= 4.4
        otro[:, 2] += 0.5
        resultado = self.decidir(np.vstack([contexto, otro]))

        self.assertEqual(resultado["decision"]["motivo"], "cubierta_ambigua")

    def test_suelo_exterior_no_impide_encontrar_techo(self):
        contexto = self.escena()
        rng = np.random.default_rng(5)
        xy = rng.uniform([-1.5, -1.5], [4.5, -0.2], (180, 2))
        suelo = np.column_stack([xy, np.zeros(len(xy))])
        resultado = self.decidir(np.vstack([contexto, suelo]))

        self.assertEqual(resultado["decision"]["origen"], "estimada")
        self.assertAlmostEqual(resultado["decision"]["inclinacion_estimada"], 25., places=5)

    def test_parche_lejano_no_completa_un_apoyo_casi_lineal(self):
        interior = self.escena(inclinacion=0.)[:81]
        xy = np.array([(x, y) for x in [3.2, 3.3, 5.5, 5.8, 6.1, 6.4]
                       for y in np.linspace(0.2, 2.8, 9)])
        exterior = np.column_stack([xy, np.full(len(xy), 10.)])
        resultado = self.decidir(np.vstack([interior, exterior]))

        self.assertEqual(resultado["decision"]["motivo"], "sin_cubierta_local")
        self.assertEqual(len(resultado["decision"]["intentos_exteriores"]), 2)

    def test_cubierta_sin_apoyo_y_contexto_truncado(self):
        sin_cubierta = self.decidir(self.escena()[:81])
        truncado = self.decidir(self.escena(), ventana=box(-0.5, -0.5, 3.5, 3.5))

        self.assertEqual(sin_cubierta["decision"]["motivo"], "sin_cubierta_local")
        self.assertEqual(truncado["decision"]["motivo"], "contexto_recortado")

    def test_ampliacion_solo_cuando_falta_apoyo(self):
        interior = self.escena(inclinacion=0.)[:81]
        xy = np.array([(x, y) for x in np.linspace(3.2, 5.8, 14) for y in [1.4, 1.6]]
                      + [(x, y) for x in np.linspace(5.3, 6.3, 8)
                         for y in np.linspace(0.2, 2.8, 9)])
        exterior = np.column_stack([xy, np.full(len(xy), 10.)])
        decision = self.decidir(np.vstack([interior, exterior]))["decision"]

        self.assertEqual(decision["origen"], "estimada")
        self.assertEqual([intento["anillo_m"] for intento in decision["intentos_exteriores"]], [2., 4.])

    def test_falta_de_puntos_conserva_el_valor_predeterminado(self):
        resultado = self.decidir(self.escena()[:2], criterios={"inclinacion_predeterminada": 28.})

        self.assertIsNone(resultado["ajuste"])
        self.assertIsNone(resultado["decision"]["inclinacion_estimada"])
        self.assertEqual(resultado["decision"]["inclinacion_utilizada"], 28.)
        self.assertEqual(resultado["decision"]["origen"], "predeterminada")
        json.dumps(resultado["decision"], allow_nan=False)

    def test_pocos_puntos_y_apoyo_casi_lineal(self):
        puntos = np.array([[0.2, 0.2, 10], [2.8, 0.2, 10], [0.2, 2.8, 10], [2.8, 2.8, 10]])
        resultado = self.decidir(puntos)
        linea = np.column_stack([np.linspace(0, 3, 100), np.zeros(100), np.full(100, 10.)])

        self.assertEqual(resultado["decision"]["motivo"], "pocos_inliers")
        self.assertEqual(evaluar_distribucion(linea)["dispersion_menor_m"], 0.)
        self.assertAlmostEqual(angulo_entre_planos(np.array([0., 0., 1.]), np.array([0., 0., -1.])), 0.)

    def test_ransac_separa_suelo_y_es_reproducible(self):
        generador = np.random.default_rng(7)
        xy = generador.uniform(-5, 5, (100, 2))
        z = np.tan(np.radians(25)) * xy[:, 0] + 10
        cubierta = np.column_stack([xy, z])
        suelo = np.column_stack([generador.uniform(-5, 5, (4, 2)), np.zeros(4)])
        puntos = np.vstack([cubierta, suelo]) + [420000, 4800000, 100]
        ajuste = buscar_plano(puntos, umbral=0.05)
        repetido = buscar_plano(puntos, umbral=0.05)

        self.assertAlmostEqual(calcular_inclinacion(ajuste["normal"]), 25, places=6)
        self.assertEqual(ajuste["compatibles"].sum(), 100)
        self.assertTrue((ajuste["distancias_con_signo"][-4:] < -1).all())
        np.testing.assert_array_equal(ajuste["compatibles"], repetido["compatibles"])
        np.testing.assert_array_equal(ajuste["normal"], repetido["normal"])

    def test_extremos_siguen_en_el_porcentaje(self):
        apoyo = evaluar_apoyo(np.array([0., 0.02, -0.04, -5., 8.]), 0.1)

        self.assertEqual(apoyo["n_puntos"], 5)
        self.assertEqual(apoyo["n_compatibles"], 3)
        self.assertEqual(apoyo["porcentaje_compatible"], 60)
        self.assertEqual(apoyo["n_por_debajo"], 1)
        self.assertEqual(apoyo["n_por_encima"], 1)
        self.assertLess(apoyo["rmse_compatibles_m"], 0.03)
        self.assertGreater(apoyo["rmse_todos_m"], 4)

    def test_puntos_insuficientes_o_alineados(self):
        for puntos in [np.zeros((2, 3)), np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])]:
            with self.assertRaises(ValueError):
                buscar_plano(puntos)

    def test_franja_exterior_y_cuadrantes(self):
        poligono = box(0, 0, 1, 1)
        contexto = np.array([
            [0.25, 0.25, 10], [0.75, 0.25, 10], [0.25, 0.75, 10],
            [1.5, 1.5, 10], [-0.5, 1.5, 10],
            [-0.5, -0.5, 10], [1.5, -0.5, 10],
            [1.5, 0.5, 0], [0.5, 1.5, 15], [5, 5, 10],
        ], dtype=float)
        dentro = intersects_xy(poligono, contexto[:, 0], contexto[:, 1])
        ajuste = {"centro": np.array([0.5, 0.5, 10]), "normal": np.array([0., 0., 1.])}
        entorno = evaluar_entorno(contexto, poligono, dentro, ajuste, ancho=1)

        self.assertEqual(entorno["resumen"]["n_puntos"], 6)
        self.assertEqual(entorno["resumen"]["n_compatibles"], 4)
        self.assertEqual(entorno["resumen"]["n_por_debajo"], 1)
        self.assertEqual(entorno["resumen"]["n_por_encima"], 1)
        self.assertFalse(entorno["exterior"][-1])
        self.assertTrue(all(v["n_compatibles"] == 1 for v in entorno["cuadrantes"].values()))

    def test_superficie_paralela_separada_no_da_apoyo(self):
        apoyo = evaluar_apoyo(np.full(20, -0.4), 0.1)

        self.assertEqual(apoyo["n_compatibles"], 0)
        self.assertEqual(apoyo["n_por_debajo"], 20)
        self.assertAlmostEqual(apoyo["mediana_con_signo_m"], -0.4)
        self.assertIsNone(apoyo["rmse_compatibles_m"])

    def test_ausencia_de_puntos_no_equivale_a_error_cero(self):
        apoyo = evaluar_apoyo(np.array([]), 0.1)

        self.assertEqual(apoyo["n_puntos"], 0)
        self.assertIsNone(apoyo["porcentaje_compatible"])
        self.assertIsNone(apoyo["rmse_todos_m"])
        self.assertIsNone(apoyo["rmse_compatibles_m"])
        json.dumps(apoyo, allow_nan=False)

    def test_las_etiquetas_no_deciden_el_ajuste(self):
        with TemporaryDirectory(prefix="prueba_apoyo_lidar_") as temporal:
            carpeta = Path(temporal)
            xy = np.array([(x, y) for x in [0.2, 0.4, 0.6, 0.8]
                           for y in [0.2, 0.4, 0.6, 0.8]])
            puntos = np.vstack([
                np.column_stack([xy, np.full(len(xy), 10.)]),
                [0.3, 0.3, 0], [0.7, 0.7, 0],
                [1.5, 0.5, 10], [-0.5, 0.5, 10],
            ])
            nube = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
            nube.x, nube.y, nube.z = puntos.T
            nube.write(carpeta / "caso.laz")
            caso = pd.Series({
                "id_panel": "caso", "laz": "caso.laz", "poligono": box(0, 0, 1, 1).wkt,
                "formulario": '{"montaje": "Estructura independiente"}',
            })
            a = procesar_caso(caso, carpeta, carpeta / "primero", guardar_html=False)
            caso.formulario = '{"montaje": "Aparentemente coplanar con cubierta"}'
            b = procesar_caso(caso, carpeta, carpeta / "segundo", guardar_html=False)

            self.assertEqual(a["normal"], b["normal"])
            self.assertEqual(a["interior"], b["interior"])
            self.assertEqual(a["exterior"], b["exterior"])
            self.assertEqual(a["origen"], b["origen"])
            self.assertEqual(a["motivo"], b["motivo"])
            self.assertEqual(a["n_compatibles"], 16)
            tabla = pd.read_csv(carpeta / "primero/caso/puntos.csv")
            self.assertEqual(len(tabla), len(puntos))
            self.assertEqual((tabla.zona == "interior").sum(), 18)
            self.assertEqual((tabla.distancia_con_signo_m < -1).sum(), 2)
            elegido = elegir_caso(pd.DataFrame([caso, caso.copy()]))
            self.assertEqual(elegido.id_panel, "caso")


if __name__ == "__main__":
    unittest.main()
