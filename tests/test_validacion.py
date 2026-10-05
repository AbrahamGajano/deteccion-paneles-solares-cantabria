"""Comprueba ventanas, revisión de etiquetas y métricas sin descargas ni GPU."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from paneles_solares.etiquetado.preparar_validacion import (
    comprobar_wcs, extraer_ventana, extraer_wcs, preparar_seleccion,
)
from paneles_solares.modelos.evaluar_unet import calcular_indicadores
from paneles_solares.modelos.barrer_umbrales import comparar_presencia, resultados_umbrales, resumir_barrido
from paneles_solares.modelos.evaluar_validacion import (
    evaluar_campania, intervalos_bootstrap, resumir, revisar_etiquetas,
)


def raster_memoria(pila, ancho=384, xmin=0, valor=100, resolucion=0.20):
    memoria = pila.enter_context(MemoryFile())
    tif = pila.enter_context(memoria.open(driver="GTiff", width=ancho, height=384,
                                         count=3, dtype="uint8", crs="EPSG:25830",
                                         transform=from_origin(xmin, 76.8, resolucion, resolucion)))
    tif.write(np.full((3, 384, ancho), valor, dtype=np.uint8))
    return tif


class PruebasValidacion(unittest.TestCase):
    def setUp(self):
        self.fila = {"tile_id": "ejemplo", "grupo": "aleatorias", "xmin": 0,
                     "ymin": 0, "xmax": 76.8, "ymax": 76.8,
                     "ancho_nativo": 384, "alto_nativo": 384}

    def test_ventana_nativa_y_cobertura(self):
        with ExitStack() as pila:
            tif = raster_memoria(pila)
            imagen, fuente = extraer_ventana(self.fila, [tif])
            self.assertEqual(imagen.size, (384, 384))
            self.assertTrue(np.all(np.asarray(imagen) == 100))
            self.assertEqual(fuente, tif.name)
            with self.assertRaises(ValueError):
                extraer_ventana({**self.fila, "xmax": 77}, [tif])

    def test_borde_entre_hojas_sin_mosaico_completo(self):
        with ExitStack() as pila:
            izquierda = raster_memoria(pila, ancho=192, valor=0)
            derecha = raster_memoria(pila, ancho=192, xmin=38.4, valor=200)
            imagen, fuente = extraer_ventana(self.fila, [izquierda, derecha])
            datos = np.asarray(imagen)
            self.assertEqual(imagen.size, (384, 384))
            self.assertTrue(np.all(datos[:, :192] == 0))
            self.assertTrue(np.all(datos[:, 192:] == 200))
            self.assertIn(" | ", fuente)

    def test_extension_fisica_a_25cm(self):
        with ExitStack() as pila:
            tif = raster_memoria(pila, ancho=384, resolucion=0.25)
            imagen, _ = extraer_ventana({**self.fila, "ancho_nativo": 307, "alto_nativo": 307}, [tif])
            self.assertEqual(imagen.size, (307, 307))

    def test_metricas_agregadas_como_evaluar_unet(self):
        real = np.array([[True, True], [False, False]])
        predicha = np.array([[True, False], [True, False]])
        fila = calcular_indicadores(real, predicha, "a")
        tabla = pd.DataFrame([{**fila, "categoria": "error_mixto"}])
        resumen = resumir(tabla)
        for metrica in ("dice", "iou", "precision", "recall", "sesgo_superficie", "fp", "fn"):
            self.assertEqual(resumen[metrica], fila[metrica])
        self.assertEqual(resumen["imagenes_positivas"], 1)
        self.assertEqual(resumen["imagenes_falso_negativo"], 0)
        self.assertEqual(resumen["superficie_real_m2"], 2 * 0.15 ** 2)

    def test_errores_de_imagen_y_metricas_indefinidas(self):
        vacia = np.zeros((2, 2), dtype=bool)
        llena = ~vacia
        tabla = pd.DataFrame([
            {**calcular_indicadores(vacia, llena, "fp"), "categoria": "falso_positivo"},
            {**calcular_indicadores(llena, vacia, "fn"), "categoria": "falso_negativo"},
        ])
        resumen = resumir(tabla)
        self.assertEqual(resumen["imagenes_falso_positivo"], 1)
        self.assertEqual(resumen["imagenes_falso_negativo"], 1)
        self.assertEqual(resumen["dice"], 0)
        negativa = pd.DataFrame([{**calcular_indicadores(vacia, vacia, "n"), "categoria": "aceptable"}])
        self.assertTrue(np.isnan(resumir(negativa)["sesgo_superficie"]))

    def test_bootstrap_pareado_reproducible(self):
        filas = []
        for i, grupo in enumerate(("aleatorias", "aleatorias", "positivos")):
            real = np.ones((2, 2), dtype=bool)
            predicha = real.copy()
            predicha[0, 0] = i == 0
            filas.append({**calcular_indicadores(real, predicha, str(i)),
                          "grupo": grupo, "categoria": "aceptable"})
        tabla = pd.DataFrame(filas)
        intervalos = intervalos_bootstrap(tabla, tabla.iloc[::-1], 20, 42)
        pd.testing.assert_frame_equal(intervalos, intervalos_bootstrap(tabla, tabla.iloc[::-1], 20, 42))
        deltas = intervalos[intervalos["metrica"].str.startswith("delta_")]
        self.assertTrue((deltas[["ic95_inferior", "ic95_superior"]] == 0).all().all())

    def test_revision_explicita_y_evaluacion_de_negativo(self):
        with tempfile.TemporaryDirectory() as nombre:
            carpeta = Path(nombre)
            imagenes = carpeta / "images" / "aleatorias"
            anotaciones = carpeta / "annotations" / "aleatorias"
            imagenes.mkdir(parents=True)
            anotaciones.mkdir(parents=True)
            Image.new("RGB", (384, 384), "black").save(imagenes / "ejemplo.png")
            seleccion = pd.DataFrame([self.fila])
            with self.assertRaises(ValueError):
                revisar_etiquetas(carpeta, seleccion)

            datos = {"imageWidth": 384, "imageHeight": 384, "shapes": [], "flags": {}}
            ruta = anotaciones / "ejemplo.json"
            ruta.write_text(json.dumps(datos), encoding="utf-8")
            with self.assertRaises(ValueError):
                revisar_etiquetas(carpeta, seleccion)
            datos["flags"]["revisada"] = True
            ruta.write_text(json.dumps(datos), encoding="utf-8")
            validas, excluidas = revisar_etiquetas(carpeta, seleccion)
            self.assertEqual(len(validas), 1)
            self.assertTrue(excluidas.empty)

            class ModeloVacio(torch.nn.Module):
                def forward(self, tensor):
                    return torch.full((1, 1, 512, 512), -20.0)

            tabla = evaluar_campania(ModeloVacio(), torch.device("cpu"), carpeta, validas)
            self.assertEqual(resumir(tabla)["imagenes_negativas"], 1)
            self.assertEqual(tabla.iloc[0]["dice"], 1)
            self.assertEqual(tabla.iloc[0]["superficie_predicha_m2"], 0)
            datos["flags"]["dudosa"] = True
            ruta.write_text(json.dumps(datos), encoding="utf-8")
            with self.assertRaises(ValueError):
                revisar_etiquetas(carpeta, seleccion)

    def test_completar_origen_sin_invalidar_imagenes(self):
        with tempfile.TemporaryDirectory() as nombre:
            carpeta = Path(nombre)
            inicial = {"anio": None, "procedencia": "", "fuentes": [], "resolucion_m": 0.2,
                       "aleatorias": 1, "positivas": 0, "semilla": 42}
            destino = {**inicial, "anio": 2024, "procedencia": "producto", "fuentes": ["hoja.tif"]}
            seleccion = pd.DataFrame([{**self.fila, "anio": None, "procedencia": ""}])
            (carpeta / "config.json").write_text(json.dumps(inicial), encoding="utf-8")
            seleccion.to_csv(carpeta / "manifest.csv", index=False)
            with patch("paneles_solares.etiquetado.preparar_validacion.carpeta_campania", return_value=carpeta):
                actual = preparar_seleccion("ejemplo", destino)
                self.assertEqual(actual.iloc[0]["anio"], 2024)
                self.assertEqual(actual.iloc[0]["procedencia"], "producto")
                imagenes = carpeta / "images" / "aleatorias"
                imagenes.mkdir(parents=True)
                Image.new("RGB", (384, 384)).save(imagenes / "ejemplo.png")
                with self.assertRaises(ValueError):
                    preparar_seleccion("ejemplo", {**destino, "anio": 2025})

    def test_wcs_recorte_y_conversion_fija_de_16bits(self):
        with MemoryFile() as memoria:
            with memoria.open(driver="GTiff", width=384, height=384, count=3,
                              dtype="uint16", crs="EPSG:25830",
                              transform=from_origin(0, 76.8, 0.2, 0.2)) as tif:
                datos = np.empty((3, 384, 384), dtype=np.uint16)
                datos[0], datos[1], datos[2] = 0, 32768, 65535
                tif.write(datos)
            contenido = memoria.read()
        respuesta = MagicMock()
        respuesta.__enter__.return_value = respuesta
        respuesta.iter_content.return_value = [contenido]
        respuesta.url = "https://ejemplo.test/wcs?REQUEST=GetCoverage"
        fuente = {"url": "https://ejemplo.test/wcs", "cobertura": "16",
                  "bandas": [1, 2, 3], "divisor_rgb": 257.0}
        with patch("paneles_solares.etiquetado.preparar_validacion.requests.get", return_value=respuesta) as peticion:
            imagen, url = extraer_wcs(self.fila, fuente)
            self.assertEqual(imagen.size, (384, 384))
            self.assertEqual(imagen.getpixel((0, 0)), (0, 128, 255))
            self.assertEqual(url, respuesta.url)
            parametros = peticion.call_args.kwargs["params"]
            self.assertEqual(parametros["Band"], "1,2,3")
            self.assertEqual(parametros["WIDTH"], 384)

    def test_wcs_rechaza_resolucion_nativa_distinta(self):
        xml = '''<CoverageDescription xmlns="http://www.opengis.net/wcs"
                 xmlns:g="http://www.opengis.net/gml"><CoverageOffering>
                 <label>Ortofoto_de_2024_True_Ortho</label>
                 <g:RectifiedGrid srsName="EPSG:25830">
                 <g:offsetVector>0.15 0</g:offsetVector>
                 <g:offsetVector>0 -0.15</g:offsetVector>
                 </g:RectifiedGrid></CoverageOffering></CoverageDescription>'''
        fuente = {"tipo": "wcs", "url": "https://ejemplo.test/wcs",
                  "cobertura": "16", "etiqueta": "Ortofoto_de_2024_True_Ortho"}
        respuesta = MagicMock(content=xml.encode())
        with patch("paneles_solares.etiquetado.preparar_validacion.requests.get", return_value=respuesta):
            with self.assertRaises(ValueError):
                comprobar_wcs(fuente, 0.2)

    def test_barrido_recupera_paneles_y_aumenta_falsos_positivos(self):
        real = np.array([[True, True, False, False]])
        probabilidades = np.array([[0.8, 0.3, 0.2, 0.01]])
        filas = resultados_umbrales(real, probabilidades, "ejemplo", "aleatorias", (0.5, 0.25, 0.1))
        self.assertEqual([(f["tp"], f["fp"], f["fn"]) for f in filas], [(1, 0, 1), (2, 0, 0), (2, 1, 0)])
        self.assertEqual(filas[1]["dice"], 1)
        tabla = pd.DataFrame(filas)
        resumen = resumir_barrido(tabla, tabla, (0.5, 0.25, 0.1))
        self.assertTrue((resumen["delta_dice"] == 0).all())
        self.assertTrue(np.array_equal(probabilidades, np.array([[0.8, 0.3, 0.2, 0.01]])))

    def test_presencia_no_confunde_omision_antigua_con_instalacion_nueva(self):
        positiva = np.array([[True]])
        negativa = ~positiva
        referencia = pd.DataFrame([
            {**calcular_indicadores(positiva, negativa, "omitida"), "umbral": 0.5},
            {**calcular_indicadores(negativa, negativa, "nueva"), "umbral": 0.5},
        ])
        actual = pd.DataFrame([
            {**calcular_indicadores(positiva, positiva, tile_id), "umbral": 0.5}
            for tile_id in ("nueva", "omitida")
        ])
        fila = comparar_presencia(actual, referencia, (0.5,)).iloc[0]
        self.assertEqual(fila["apariciones_predichas"], 2)
        self.assertEqual(fila["apariciones_en_teselas_ya_positivas_2023"], 1)
        self.assertEqual(fila["cambios_de_etiqueta_detectados"], 1)


if __name__ == "__main__":
    unittest.main()
