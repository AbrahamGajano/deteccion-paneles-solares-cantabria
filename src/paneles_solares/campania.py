"""Ejecuta las etapas de una campaña a partir de TIFF locales o de un inventario."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from paneles_solares.rutas import (
    EDIFICIOS_CATASTRO,
    MUNICIPIOS_CANTABRIA,
    PESOS_UNET,
    RAIZ,
    VISUALIZACION,
    guardar_json_atomico,
    ruta_proyecto,
)


EJEMPLOS = """Ejemplos de ejecución

  1. Generar el HTML de 2023 desde el inventario revisado y sus ángulos.
     python -m paneles_solares.campania --inventario runs/indicadores/seguimiento_historico --lidar runs/lidar/cantabria_20261112_v2/resumen.csv --salida runs/campanias/2023

  2. Probar TIFF de otra campaña en una zona pequeña (coordenadas EPSG:25830).
     python -m paneles_solares.campania --anio 2026 --ortofotos data/pnoa_2026 --salida runs/campanias/prueba_2026 --bbox 387515.6 4799608.4 387692.4 4799785.2

  Repetir el comando retoma las etapas pendientes. Los registros quedan en la salida.
  Para rehacer 2023 desde los TIFF, combinar --ortofotos con --manifest
  y proporcionar el manifiesto del lote de LabelMe cerrado.
"""


def preparar_pasos(args):
    """Define los módulos que se ejecutan y los archivos que deja cada etapa."""
    salida = args.salida
    inventario = args.inventario or salida / "inventario"
    pasos = []
    if args.ortofotos:
        indice = salida / "teselas.csv"
        detecciones = salida / "detecciones"
        bbox = ["--bbox", *map(str, args.bbox)] if args.bbox else []
        entrada = ["--anio", str(args.anio), "--ortofotos", str(args.ortofotos), *bbox]
        pasos += [
            ("teselas", ["-m", "paneles_solares.geografia.teselas", *entrada,
                         "--salida", str(indice)], [indice]),
            ("detecciones", ["-m", "paneles_solares.modelos.indicadores_unet", *entrada,
                             "--indice", str(indice), "--pesos", str(args.pesos),
                             "--salida", str(detecciones)],
             [detecciones / "ejecucion.json", detecciones / "detecciones.gpkg"]),
        ]
        if not args.manifest:
            return pasos
        pasos.append((
            "inventario", ["-m", "paneles_solares.modelos.indicadores_historicos",
                           "--manifest", str(args.manifest), "--salida", str(inventario),
                           "--detecciones", str(detecciones / "detecciones.gpkg")],
            [inventario / "instalaciones.csv", inventario / "atribucion.gpkg"],
        ))

    lidar = args.lidar
    if args.recalcular_lidar:
        preparacion = salida / "lidar" / "preparacion"
        recortes = salida / "lidar" / "recortes"
        informes = salida / "lidar" / "informes"
        planos = salida / "lidar" / "planos"
        lidar = planos / "resumen.csv"
        pasos += [
            ("preparar_lidar", ["-m", "paneles_solares.lidar.preparar_recortes",
                               "--paneles", str(inventario / "atribucion.gpkg"),
                               "--salida", str(preparacion)],
             [preparacion / nombre for nombre in ("recortes.gpkg", "hojas_descarga.csv", "relaciones.csv")]),
            ("recortar_lidar", ["-m", "paneles_solares.lidar.descargar_recortar",
                               "--limite", "0", "--preparacion", str(preparacion),
                               "--salida", str(recortes), "--temporales", str(salida / "lidar" / "tmp"),
                               "--informes", str(informes)], [informes / "errores.csv"]),
            ("planos", ["-m", "paneles_solares.lidar.estimar_planos", "--cantabria",
                        "--solo-resumen", "--reanudar", "--preparacion", str(preparacion),
                        "--recortes", str(recortes), "--salida", str(planos)], [lidar]),
        ]

    panel = salida / "panel"
    figuras = salida / "mapas"
    opcion_lidar = ["--lidar", str(lidar)] if lidar else ["--sin-lidar"]
    pasos += [
        ("panel", ["-m", "paneles_solares.visualizacion.panel_indicadores",
                   "--carpeta", str(inventario), "--salida", str(panel), *opcion_lidar],
         [panel / nombre for nombre in ("panel_indicadores.html", "energia_paneles.csv",
                                       "resumen_municipal.csv", "produccion_estimada.csv",
                                       "procedencia_panel.json")]),
        ("mapas", [str(RAIZ / "docs/generar_mapas.py"), "--anio", str(args.anio),
                   "--carpeta", str(panel), "--salida", str(figuras)],
         [figuras / "mapa_detecciones.png", figuras / "mapa_produccion.png"]),
    ]
    if args.horarias:
        series = panel / "series_temporales" / f"inventario_{args.anio}_meteo_{args.anio_meteo}"
        pasos.append((
            "horarias", ["-m", "paneles_solares.visualizacion.series_temporales",
                         "--carpeta", str(panel), "--anio-inventario", str(args.anio),
                         "--anio-meteo", str(args.anio_meteo), "--exportar-horarias"],
            [series / "procedencia.json", series / "anual_instalaciones.csv"],
        ))
    return pasos


def ejecutar(pasos, salida: Path, entradas: list[Path]) -> None:
    """Reutiliza etapas terminadas y escribe un registro por módulo."""
    salida.mkdir(parents=True, exist_ok=True)
    archivo = salida / "progreso.json"
    firmas = {}
    for ruta in entradas:
        datos = ruta.stat()
        firmas[str(ruta)] = [datos.st_size, datos.st_mtime_ns]
    plan = {"comandos": [comando for _, comando, _ in pasos], "entradas": firmas}
    estado = {"plan": plan, "completadas": []}
    if archivo.is_file():
        estado = json.loads(archivo.read_text(encoding="utf-8"))
        anterior = estado["plan"]
        if (
            anterior["entradas"] != firmas
            or anterior["comandos"] != plan["comandos"][:len(anterior["comandos"])]
        ):
            raise SystemExit("Cambiaron los datos de entrada o los argumentos. Indica otra --salida.")
        estado["plan"] = plan

    if any(nombre == "panel" for nombre, _, _ in pasos):
        cache = VISUALIZACION / "pvgis_anual.json"
        destino = salida / "panel" / cache.name
        if cache.is_file() and not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            temporal = destino.with_suffix(".tmp")
            shutil.copyfile(cache, temporal)
            temporal.replace(destino)

    for i, (nombre, comando, resultados) in enumerate(pasos):
        if nombre in estado["completadas"] and all(ruta.is_file() for ruta in resultados):
            print(f"{nombre}: reutilizado.", flush=True)
            continue
        registro = salida / f"{nombre}.log"
        estado.update(completadas=estado["completadas"][:i], etapa=nombre)
        guardar_json_atomico(archivo, estado, indent=2)
        print(f"{nombre}: {registro}", flush=True)
        with registro.open("ab") as log:
            proceso = subprocess.run(
                [sys.executable, "-X", "utf8", "-u", *comando],
                cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT,
            )
        if proceso.returncode or not all(ruta.is_file() for ruta in resultados):
            raise SystemExit(f"Falló {nombre}. Consulta {registro} y repite el comando para continuar.")
        estado["completadas"].append(nombre)
        estado["etapa"] = None
        guardar_json_atomico(archivo, estado, indent=2)
    print(f"Terminado. Resultados en {salida}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=EJEMPLOS, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--anio", type=int, default=2023, help="Año de las ortofotos.")
    origen = parser.add_mutually_exclusive_group(required=True)
    origen.add_argument("--ortofotos", type=ruta_proyecto, help="Carpeta de TIFF locales.")
    origen.add_argument("--inventario", type=ruta_proyecto, help="Carpeta con atribucion.gpkg e instalaciones.csv.")
    parser.add_argument("--salida", type=ruta_proyecto, required=True)
    parser.add_argument("--pesos", type=ruta_proyecto, default=PESOS_UNET)
    parser.add_argument("--bbox", type=float, nargs=4, help="Zona de prueba de la detección en EPSG:25830.")
    parser.add_argument("--manifest", type=ruta_proyecto, help="Seguimiento cerrado de LabelMe para rehacer el inventario de 2023.")
    lidar = parser.add_mutually_exclusive_group()
    lidar.add_argument("--lidar", type=ruta_proyecto, help="Resumen CSV de ángulos ya calculados.")
    lidar.add_argument("--recalcular-lidar", action="store_true", help="Descarga, recorta y ajusta planos con el LiDAR de 2023.")
    parser.add_argument("--horarias", action="store_true", help="Exporta también las series por instalación.")
    parser.add_argument("--anio-meteo", type=int, default=2023, help="Año meteorológico de las series.")
    args = parser.parse_args()

    publicar = args.inventario or args.manifest
    if publicar and args.anio != 2023:
        parser.error("La publicación está preparada para 2023. Para probar otra campaña, usa --ortofotos.")
    if args.manifest and not args.ortofotos:
        parser.error("--manifest se usa con --ortofotos.")
    if args.bbox and publicar:
        parser.error("--bbox limita solo la prueba de detección. Para publicar, usa el ámbito del inventario o del manifiesto.")
    if (args.lidar or args.recalcular_lidar or args.horarias) and not publicar:
        parser.error("LiDAR y series requieren un --inventario o un --manifest.")

    entradas = [EDIFICIOS_CATASTRO, MUNICIPIOS_CANTABRIA] if args.ortofotos else [MUNICIPIOS_CANTABRIA]
    if args.ortofotos:
        if not args.salida.resolve().is_relative_to(RAIZ / "runs"):
            parser.error("La detección necesita una --salida dentro de runs/.")
        if not args.ortofotos.is_dir():
            parser.error(f"No existe la carpeta de ortofotos {args.ortofotos}")
        tiffs = sorted(r for r in args.ortofotos.iterdir() if r.is_file() and r.suffix.lower() in (".tif", ".tiff"))
        if not tiffs:
            parser.error(f"No hay TIFF en {args.ortofotos}")
        entradas += [args.pesos, *tiffs]
    if args.inventario:
        entradas += [args.inventario / nombre for nombre in ("atribucion.gpkg", "instalaciones.csv")]
    entradas += [ruta for ruta in (args.manifest, args.lidar) if ruta is not None]
    for ruta in entradas:
        if not ruta.is_file():
            parser.error(f"Falta {ruta}")

    try:
        ejecutar(preparar_pasos(args), args.salida, entradas)
    except KeyboardInterrupt:
        raise SystemExit("Campaña interrumpida. Repite el comando para continuar.") from None


if __name__ == "__main__":
    main()
