"""Comprueba la reanudación de PVGIS tras un fallo de conexión."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import geopandas as gpd
import requests
from shapely.geometry import box

from paneles_solares.visualizacion.estimar_energia import rendimientos_historicos


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


if __name__ == "__main__":
    unittest.main()
