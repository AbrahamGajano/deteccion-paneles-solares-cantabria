"""Comprueba unidades, fechas, factores y reanudación de las series horarias."""

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from paneles_solares.visualizacion.series_temporales import (
    calcular_serie_instalacion,
    normalizar_serie_horaria,
    obtener_serie_horaria,
    parametros_horarios,
    preparar_relaciones,
    preparar_series_temporales,
)


def respuesta_horaria(anio=2023, potencia=500, minuto=10):
    fechas = pd.date_range(f"{anio}-01-01", f"{anio + 1}-01-01", freq="h", inclusive="left")
    fechas += pd.Timedelta(minutes=minuto)
    return {"outputs": {"hourly": [
        {"time": fecha.strftime("%Y%m%d:%H%M"), "P": potencia} for fecha in fechas
    ]}}


class PruebasFechasYUnidades(unittest.TestCase):
    def test_alinea_muestras_y_convierte_w_a_kw(self):
        serie = normalizar_serie_horaria(respuesta_horaria(), 2023)
        self.assertEqual(len(serie), 8760)
        self.assertEqual(serie.iloc[0], 0.5)
        self.assertEqual(serie.index[0], pd.Timestamp("2023-01-01", tz="UTC"))
        self.assertEqual(serie.index[-1], pd.Timestamp("2023-12-31 23:00", tz="UTC"))
        self.assertEqual(serie.sum(), 4380)
        otra = normalizar_serie_horaria(respuesta_horaria(minuto=12), 2023)
        self.assertTrue(serie.index.equals(otra.index))

    def test_conserva_anio_bisiesto_y_cambios_de_horario(self):
        serie = normalizar_serie_horaria(respuesta_horaria(2024), 2024)
        self.assertEqual(len(serie), 8784)
        local = serie.tz_convert("Europe/Madrid")
        self.assertEqual(len(local), len(serie))
        self.assertEqual(local.sum(), serie.sum())
        self.assertEqual(local.resample("D").size().loc["2024-03-31"], 23)
        self.assertEqual(local.resample("D").size().loc["2024-10-27"], 25)

    def test_rechaza_huecos_y_duplicados(self):
        respuesta = respuesta_horaria()
        respuesta["outputs"]["hourly"].pop(100)
        with self.assertRaisesRegex(ValueError, "completo"):
            normalizar_serie_horaria(respuesta, 2023)
        respuesta["outputs"]["hourly"].append(respuesta["outputs"]["hourly"][0])
        with self.assertRaisesRegex(ValueError, "duplicados"):
            normalizar_serie_horaria(respuesta, 2023)

    def test_rechaza_potencia_no_valida(self):
        for potencia in [-1, np.nan]:
            respuesta = respuesta_horaria()
            respuesta["outputs"]["hourly"][0]["P"] = potencia
            with self.assertRaisesRegex(ValueError, "potencias"):
                normalizar_serie_horaria(respuesta, 2023)


class PruebasCacheHoraria(unittest.TestCase):
    def test_reanuda_sin_repetir_consultas_y_distingue_parametros(self):
        fila = SimpleNamespace(latitud=43.4, longitud=-3.8,
                               inclinacion_pvgis_grados=10, orientacion_grados=0)
        parametros = parametros_horarios(fila, 2023)
        otros = {**parametros, "angle": 15}
        with TemporaryDirectory() as temporal, requests.Session() as sesion:
            cache = Path(temporal)
            with patch(
                "paneles_solares.visualizacion.series_temporales.solicitar_pvgis",
                side_effect=[respuesta_horaria(), requests.ConnectionError("Sin conexión")],
            ):
                archivo, original = obtener_serie_horaria(sesion, parametros, cache)
                with self.assertRaises(requests.ConnectionError):
                    obtener_serie_horaria(sesion, otros, cache)
            self.assertEqual(len(json.loads((cache / "consultas.json").read_text())), 1)
            with patch(
                "paneles_solares.visualizacion.series_temporales.solicitar_pvgis",
                return_value=respuesta_horaria(),
            ) as consultar:
                misma_ruta, reutilizada = obtener_serie_horaria(sesion, parametros, cache)
                otra_ruta, _ = obtener_serie_horaria(sesion, otros, cache)
                self.assertEqual(consultar.call_count, 1)
            self.assertEqual(archivo, misma_ruta)
            self.assertNotEqual(archivo, otra_ruta)
            pd.testing.assert_series_equal(original, reutilizada)


class PruebasProduccionHoraria(unittest.TestCase):
    def energia(self):
        datos = pd.DataFrame({
            "id_panel": ["a", "b"], "id_instalacion": ["i", "i"],
            "anio": [2023, 2023], "codigo": ["39001", "39001"],
            "latitud": [43.4, 43.4], "longitud": [-3.8, -3.8],
            "area_m2": [10., 20.], "origen_inclinacion": ["estimada", "estimada"],
            "inclinacion_estimada": [9., 29.], "antiguedad_supuesta": [False, False],
        })
        for nombre in ["min", "central", "max"]:
            datos[f"factor_{nombre}"] = [0.2, 0.3]
            datos[f"inclinacion_{nombre}_grados"] = [9., 29.]
            datos[f"inclinacion_pvgis_{nombre}_grados"] = [10, 30]
        return datos

    def test_filtra_campania_y_no_reaplica_eficiencia_o_perdidas(self):
        energia = pd.concat([self.energia(), self.energia().assign(anio=2020)])
        energia["orientacion_central_grados"] = [-86.0, 93.0] * 2
        energia["orientacion_pvgis_central_grados"] = [-90, 90] * 2
        relaciones = preparar_relaciones(energia, 2023, ["central"])
        self.assertEqual(len(relaciones), 2)
        self.assertEqual(relaciones.escala_referencia.tolist(), [2., 6.])
        self.assertEqual(relaciones.orientacion_grados.tolist(), [-90, 90])
        self.assertEqual(parametros_horarios(next(relaciones.itertuples()), 2023)["aspect"], -90)
        with self.assertRaisesRegex(ValueError, "No hay detecciones"):
            preparar_relaciones(energia, 2014, ["central"])

    def test_curvas_resumenes_y_exportacion_coinciden_sin_promediar_angulos(self):
        def consultar(sesion, servicio, parametros):
            self.assertEqual(servicio, "seriescalc")
            self.assertEqual(parametros["loss"], 0)
            self.assertEqual(parametros["peakpower"], 1)
            return respuesta_horaria(potencia=100 + parametros["angle"])

        with TemporaryDirectory() as temporal, redirect_stdout(io.StringIO()):
            carpeta = Path(temporal)
            with patch(
                "paneles_solares.visualizacion.series_temporales.solicitar_pvgis",
                side_effect=consultar,
            ) as peticion:
                relaciones = preparar_series_temporales(
                    self.energia(), carpeta / "salida", cache=carpeta / "cache", exportar_horarias=True,
                )
            self.assertEqual(peticion.call_count, 4)
            serie = calcular_serie_instalacion(relaciones)
            esperado = 2 * 0.110 + 6 * 0.130
            np.testing.assert_allclose(serie.potencia_central_kw, esperado)
            np.testing.assert_allclose(serie.energia_central_kwh, esperado)
            anual = pd.read_csv(carpeta / "salida/anual_instalaciones.csv")
            central = anual[anual.escenario.eq("central")].energia_kwh.iloc[0]
            self.assertAlmostEqual(central, esperado * 8760)
            self.assertAlmostEqual(central, serie.energia_central_kwh.sum())
            mensual = pd.read_csv(carpeta / "salida/mensual_instalaciones.csv")
            self.assertAlmostEqual(mensual[mensual.escenario.eq("central")].energia_kwh.sum(), central)
            exportada = pd.read_csv(carpeta / "salida/instalaciones/i.csv.gz")
            self.assertEqual(len(exportada), 8760)
            self.assertAlmostEqual(exportada.energia_central_kwh.sum(), central)


if __name__ == "__main__":
    unittest.main()
