"""Comprueba la reanudación de PVGIS tras un fallo de conexión."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box

from paneles_solares.datos.inclinaciones import incorporar_inclinaciones
from paneles_solares.visualizacion.estimar_energia import (
    energia_union,
    factores_historicos,
    factores_paneles,
    preparar_energia_historica,
    rendimientos_historicos,
)


class PruebasPvgis(unittest.TestCase):
    def test_reanudar_conserva_consultas_terminadas(self):
        recortes = gpd.GeoDataFrame(
            {"codigo": ["39001"]},
            geometry=[box(420000, 4790000, 420010, 4790010)],
            crs=25830,
        )
        respuesta = {"outputs": {"totals": {"fixed": {"E_y": 1200}}}}
        with TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            carpeta = Path(temporal)
            with (
                patch(
                    "paneles_solares.visualizacion.estimar_energia.solicitar_pvgis",
                    side_effect=[respuesta, requests.ConnectionError("Sin conexión")],
                ),
                self.assertRaises(requests.ConnectionError),
            ):
                rendimientos_historicos(recortes, carpeta)

            with patch(
                "paneles_solares.visualizacion.estimar_energia.solicitar_pvgis",
                return_value=respuesta,
            ) as consultar:
                tabla = rendimientos_historicos(recortes, carpeta)

            self.assertEqual(consultar.call_count, 2)
            self.assertEqual(tabla.rendimiento_min.iloc[0], 1200)
            self.assertEqual(tabla.rendimiento_central.iloc[0], 1200)
            self.assertEqual(tabla.rendimiento_max.iloc[0], 1200)

    def test_reutiliza_mismo_angulo_y_orientacion_entre_escenarios(self):
        recortes = gpd.GeoDataFrame(
            {"codigo": ["39001", "39001"],
             **{f"inclinacion_pvgis_{n}_grados": [10, 15] for n in ["min", "central", "max"]}},
            geometry=[box(420000, 4790000, 420010, 4790010),
                      box(420010, 4790000, 420020, 4790010)], crs=25830,
        )
        respuesta = {"outputs": {"totals": {"fixed": {"E_y": 1200}}}}
        with TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            with patch(
                "paneles_solares.visualizacion.estimar_energia.solicitar_pvgis",
                return_value=respuesta,
            ) as consultar:
                rendimientos_historicos(recortes, Path(temporal))
                self.assertEqual(consultar.call_count, 4)
                rendimientos_historicos(recortes, Path(temporal))
                self.assertEqual(consultar.call_count, 4)


class PruebasInclinaciones(unittest.TestCase):
    def setUp(self):
        self.paneles = gpd.GeoDataFrame(
            {"id_panel": ["a", "b", "c", "a"], "id_instalacion": ["i"] * 4,
             "anio": [2023, 2023, 2023, 2020]},
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10),
                      box(20, 0, 30, 10), box(0, 0, 10, 10)], crs=25830,
        )
        self.candidatos = pd.DataFrame({
            "id_instalacion": ["i"], "instalacion_desde": [2017],
            "instalacion_hasta": [2020], "estado_periodo": ["acotado"],
        })

    def incorporar(self, carpeta):
        archivo = Path(carpeta) / "lidar.csv"
        pd.DataFrame({
            "id_panel": ["a", "b"], "inclinacion_estimada": [0., 60.],
            "origen": ["estimada", "estimada"], "motivo": ["compatible"] * 2,
        }).to_csv(archivo, index=False)
        return incorporar_inclinaciones(self.paneles, archivo)

    def test_no_propaga_angulos_a_otras_campanias_y_respeta_cero(self):
        with TemporaryDirectory() as carpeta:
            datos = self.incorporar(carpeta)
        self.assertEqual(datos.inclinacion_estimada.iloc[0], 0.)
        self.assertTrue(pd.isna(datos.inclinacion_estimada.iloc[2]))
        self.assertTrue(pd.isna(datos.inclinacion_estimada.iloc[3]))
        self.assertEqual(datos.origen_inclinacion.tolist(),
                         ["estimada", "estimada", "sin_resultado", "otra_campania"])

    def test_factores_corrigen_superficie_y_conservan_supuestos(self):
        with TemporaryDirectory() as carpeta:
            datos = self.incorporar(carpeta)
        factores = factores_paneles(self.candidatos, datos)
        anteriores = factores_historicos(self.candidatos, self.paneles).set_index("anio")
        central = anteriores.loc[2023, "factor_central"]
        self.assertAlmostEqual(factores.factor_central.iloc[0], central * np.cos(np.radians(30)))
        self.assertAlmostEqual(factores.factor_central.iloc[1], central * np.cos(np.radians(30)) * 2)
        for nombre in ["min", "central", "max"]:
            self.assertEqual(factores[f"factor_{nombre}"].iloc[2], anteriores.loc[2023, f"factor_{nombre}"])
            self.assertEqual(factores[f"factor_{nombre}"].iloc[3], anteriores.loc[2020, f"factor_{nombre}"])

    def test_suma_energia_por_panel_y_mantiene_area_horizontal(self):
        with TemporaryDirectory() as carpeta, redirect_stdout(io.StringIO()):
            datos = self.incorporar(carpeta).iloc[:2]
            recortes = datos[["id_panel", "id_instalacion", "anio", "geometry"]].copy()
            recortes["codigo"] = "39001"
            def responder(sesion, servicio, parametros):
                return {"outputs": {"totals": {"fixed": {"E_y": 1000 + parametros["angle"]}}}}
            with patch(
                "paneles_solares.visualizacion.estimar_energia.solicitar_pvgis",
                side_effect=responder,
            ):
                energia = preparar_energia_historica(self.candidatos, datos, recortes, Path(carpeta))
            factores = factores_paneles(self.candidatos, datos)
            esperado = (100 * factores.factor_central.iloc[0] * 1000
                        + 100 * factores.factor_central.iloc[1] * 1060) / 1000
            total = energia_union(200, energia)
            self.assertAlmostEqual(total["produccion_central_mwh"], esperado)
            self.assertEqual(energia.area_m2.sum(), 200)
            self.assertTrue((Path(carpeta) / "energia_paneles.csv").exists())

    def test_descarta_candidatas_no_aceptadas(self):
        with TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "lidar.csv"
            pd.DataFrame({"id_panel": ["a"], "inclinacion_estimada": [12.],
                          "origen": ["predeterminada"], "motivo": ["sin_cubierta"]}).to_csv(archivo, index=False)
            datos = incorporar_inclinaciones(self.paneles, archivo)
        self.assertTrue(datos.inclinacion_estimada.isna().all())
        sin_lidar = incorporar_inclinaciones(self.paneles, None)
        self.assertTrue(sin_lidar.inclinacion_estimada.isna().all())

    def test_reparte_solapes_antes_de_sumar_instalaciones(self):
        datos = self.paneles.iloc[:3].copy()
        datos.loc[1, "geometry"] = datos.geometry.iloc[0]
        datos.loc[2, "id_instalacion"] = "j"
        datos = incorporar_inclinaciones(datos, None)
        candidatos = pd.concat([self.candidatos, self.candidatos.assign(id_instalacion="j")])
        recortes = datos[["id_panel", "id_instalacion", "anio", "geometry"]].copy()
        recortes["codigo"] = "39001"
        respuesta = {"outputs": {"totals": {"fixed": {"E_y": 1200}}}}
        with TemporaryDirectory() as carpeta, redirect_stdout(io.StringIO()):
            with patch(
                "paneles_solares.visualizacion.estimar_energia.solicitar_pvgis",
                return_value=respuesta,
            ):
                energia = preparar_energia_historica(candidatos, datos, recortes, Path(carpeta))
        self.assertEqual(energia.superficie_horizontal_m2.sum(), 300)
        self.assertEqual(energia.area_m2.sum(), 200)
        por_instalacion = energia.groupby("id_instalacion").area_m2.sum()
        self.assertEqual(por_instalacion.to_dict(), {"i": 100, "j": 100})


if __name__ == "__main__":
    unittest.main()
