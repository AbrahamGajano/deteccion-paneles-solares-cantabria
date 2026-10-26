"""Prepara recortes históricos para etiquetarlos directamente en LabelMe."""

import argparse
import json
import sqlite3
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from PIL import Image

from paneles_solares.datos.anotaciones import excluida, revisar_par
from paneles_solares.datos.seguimiento import (
    BASE,
    CARPETA,
    URL_WCS,
    cargar_config,
    guardar_csv,
)
from paneles_solares.etiquetado.preparar_validacion import comprobar_wcs, extraer_wcs
from paneles_solares.rutas import huella_archivo, ruta_proyecto, ruta_registrada

SALIDA = CARPETA / "labelme"
LIMITES = ("xmin", "ymin", "xmax", "ymax")


def recuperar_positivos(base: Path = BASE) -> pd.DataFrame:
    """Recupera las respuestas positivas del visor anterior, sin alterar la base.

    Args:
        base: SQLite con las selecciones, ventanas y respuestas originales.

    Returns:
        pd.DataFrame: Una fila por ventana y campaña positiva anterior a 2023.
    """

    with closing(sqlite3.connect(f"{base.resolve().as_uri()}?mode=ro", uri=True)) as db:
        ventanas = pd.read_sql_query(
            "SELECT v.id AS ventana,o.anio,v.xmin,v.ymin,v.xmax,v.ymax,"
            "r.imagen AS origen,r.fuente,"
            "GROUP_CONCAT(o.emplazamiento,';') AS selecciones "
            "FROM observaciones o JOIN emplazamientos e ON e.id=o.emplazamiento "
            "JOIN ventanas v ON v.id=e.ventana "
            "LEFT JOIN recursos r ON r.ventana=v.id AND r.anio=o.anio "
            "WHERE o.anio<2023 AND o.decision='presente' "
            "GROUP BY v.id,o.anio ORDER BY v.id,o.anio DESC",
            db,
        )
    return ventanas.fillna("")


def guardar_png(imagen: Image.Image, ruta: Path) -> None:
    """Publica un recorte completo para poder reanudar descargas interrumpidas.

    Args:
        imagen: Recorte RGB de la campaña.
        ruta: PNG de destino, todavía sin anotaciones.
    """

    temporal = ruta.with_suffix(".png.part")
    try:
        imagen.save(temporal, format="PNG")
        temporal.replace(ruta)
    finally:
        temporal.unlink(missing_ok=True)


def preparar(
    ventanas: pd.DataFrame,
    campanias: list[dict],
    salida: Path = SALIDA,
) -> pd.DataFrame:
    """Reutiliza o descarga las mismas ventanas físicas en las campañas elegidas.

    Args:
        ventanas: CSV con ventana y límites en EPSG:25830. El año es opcional;
            si se indica, la fila solo se prepara para esa campaña.
        campanias: Año, cobertura WCS, etiqueta, resolución nativa y divisor RGB.
        salida: Carpeta con manifest.csv e images/; los JSON quedan junto al PNG.

    Returns:
        pd.DataFrame: Manifiesto acumulado, incluidas las imágenes ya preparadas.

    Raises:
        ValueError: Se intenta cambiar el encuadre o la escala de un PNG existente.
    """

    imagenes = salida / "images"
    imagenes.mkdir(parents=True, exist_ok=True)
    manifest = salida / "manifest.csv"
    anteriores = (
        pd.read_csv(manifest, dtype={"ventana": str}, keep_default_na=False)
        if manifest.exists()
        else pd.DataFrame()
    )
    filas = {(f["ventana"], int(f["anio"])): f for f in anteriores.to_dict("records")}
    comprobadas = set()

    try:
        for ventana in ventanas.to_dict("records"):
            nombre = str(ventana["ventana"])
            if Path(nombre).name != nombre:
                raise ValueError(
                    "El identificador de ventana no puede contener carpetas"
                )
            if ventana.get("origen") and ventana.get("anio", "") == "":
                raise ValueError(
                    "Indica el año de cada recorte local en la columna anio"
                )
            for campania in campanias:
                anio = campania["anio"]
                if ventana.get("anio", "") != "" and int(ventana["anio"]) != anio:
                    continue
                clave = (nombre, anio)
                previa = filas.get(clave, {})
                limites = [float(ventana[c]) for c in LIMITES]
                if previa and any(
                    abs(v - float(previa[c])) > 1e-6 for c, v in zip(LIMITES, limites)
                ):
                    raise ValueError(f"Ha cambiado el encuadre de {nombre}, {anio}")
                fila = {**previa, **ventana, "ventana": nombre, "anio": anio}
                fila.update(zip(LIMITES, limites))
                destino = imagenes / f"{nombre}__{anio}.png"
                if previa and destino.exists():
                    fila["origen"] = previa.get("origen", "")
                    fila["fuente"] = previa.get("fuente", "")
                resolucion = campania["resolucion_m"]
                ancho = round((fila["xmax"] - fila["xmin"]) / resolucion)
                alto = round((fila["ymax"] - fila["ymin"]) / resolucion)

                if not destino.exists():
                    origen = (
                        ruta_proyecto(fila["origen"]) if fila.get("origen") else None
                    )
                    if origen and origen.is_file():
                        with Image.open(origen) as imagen:
                            guardar_png(imagen, destino)
                        fila["origen"] = ruta_registrada(origen)
                    else:
                        fuente = {
                            "tipo": "wcs",
                            "url": campania.get("url", URL_WCS),
                            "cobertura": campania["cobertura"],
                            "etiqueta": campania["etiqueta"],
                            "bandas": [1, 2, 3],
                            "divisor_rgb": campania["divisor_rgb"],
                        }
                        if anio not in comprobadas:
                            comprobar_wcs(fuente, resolucion)
                            comprobadas.add(anio)
                        recorte = {
                            **fila,
                            "tile_id": nombre,
                            "ancho_nativo": ancho,
                            "alto_nativo": alto,
                        }
                        imagen, fila["fuente"] = extraer_wcs(recorte, fuente)
                        guardar_png(imagen, destino)
                        fila["origen"] = ""

                with Image.open(destino) as imagen:
                    if imagen.size != (ancho, alto):
                        raise ValueError(f"Ha cambiado la escala de {nombre}, {anio}")
                fila.update(
                    ancho=ancho,
                    alto=alto,
                    crs="EPSG:25830",
                    imagen=f"images/{destino.name}",
                    anotacion=f"images/{destino.stem}.json",
                    resolucion_x=(fila["xmax"] - fila["xmin"]) / ancho,
                    resolucion_y=(fila["ymax"] - fila["ymin"]) / alto,
                )
                filas[clave] = fila
    finally:
        if filas:
            guardar_csv(pd.DataFrame(filas.values()), manifest)

    return pd.DataFrame(filas.values())


def finalizar(manifest: Path, sin_json_negativos: bool = False) -> pd.DataFrame:
    """Registra una revisión terminada y la versión exacta de sus anotaciones.

    Args:
        manifest: Manifiesto de las imágenes revisadas en LabelMe.
        sin_json_negativos: Excepción explícita para lotes revisados en los que
            se dejaron sin JSON las imágenes sin paneles. No se deduce por defecto.

    Returns:
        pd.DataFrame: Estados y huellas de los JSON; las dudosas siguen sin confirmar.
    """

    tabla = pd.read_csv(manifest, dtype={"ventana": str}, keep_default_na=False)
    for indice, fila in tabla.iterrows():
        imagen = manifest.parent / fila.imagen
        anotacion = manifest.parent / fila.anotacion
        if not anotacion.exists() and sin_json_negativos:
            with Image.open(imagen) as png:
                ancho, alto = png.size
            datos = {
                "version": version("labelme"),
                "flags": {"ausente": True},
                "shapes": [],
                "imagePath": imagen.name,
                "imageData": None,
                "imageWidth": ancho,
                "imageHeight": alto,
            }
            anotacion.write_text(json.dumps(datos, indent=2), encoding="utf-8")

        estado, completa, firma = "pendiente", False, ""
        if anotacion.exists():
            datos = revisar_par(imagen, anotacion)
            estado = "positiva" if datos["shapes"] else "ausente"
            if excluida(datos):
                estado = "no_evaluable"
            completa = estado != "no_evaluable"
            firma = huella_archivo(anotacion)
        tabla.loc[indice, "estado"] = estado
        tabla.loc[indice, "revision_completa"] = completa
        tabla.loc[indice, "anotacion_sha256"] = firma

    guardar_csv(tabla, manifest)
    return tabla


def main() -> None:
    """Prepara ventanas de un CSV o recupera los positivos de la revisión anterior."""

    parser = argparse.ArgumentParser(description=__doc__)
    origen = parser.add_mutually_exclusive_group(required=True)
    origen.add_argument("--ventanas", type=Path, help="CSV de ventanas EPSG:25830")
    origen.add_argument(
        "--recuperar", action="store_true", help="Positivos del visor anterior"
    )
    origen.add_argument(
        "--finalizar", type=Path, metavar="MANIFEST", help="Confirmar un lote revisado"
    )
    parser.add_argument(
        "--sin-json-negativos",
        action="store_true",
        help="El revisor dejó sin JSON las imágenes vacías del lote terminado",
    )
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--anios", type=int, nargs="+", default=[2020, 2017, 2014])
    parser.add_argument("--config", help="JSON alternativo de campañas")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument(
        "--limite", type=int, help="Máximo de filas de entrada para una prueba"
    )
    args = parser.parse_args()

    if args.sin_json_negativos and not args.finalizar:
        parser.error("--sin-json-negativos solo se utiliza al finalizar una revisión")
    if args.finalizar:
        tabla = finalizar(ruta_proyecto(args.finalizar), args.sin_json_negativos)
        print(tabla.estado.value_counts().to_dict())
        return

    config = cargar_config(args.config)
    campanias = [c for c in config["campanias"] if c["anio"] in args.anios]
    if set(args.anios) != {c["anio"] for c in campanias}:
        parser.error("Falta la configuración de alguna campaña solicitada")
    if args.limite is not None and args.limite < 1:
        parser.error("El límite debe ser positivo")
    ventanas = (
        recuperar_positivos(ruta_proyecto(args.base))
        if args.recuperar
        else pd.read_csv(
            ruta_proyecto(args.ventanas), dtype={"ventana": str}, keep_default_na=False
        )
    )
    if args.limite is not None:
        ventanas = ventanas.head(args.limite)
    salida = ruta_proyecto(args.salida)
    tabla = preparar(ventanas, campanias, salida)
    print(
        f"Imágenes preparadas: {len(tabla)}; JSON guardados: {len(list((salida / 'images').glob('*.json')))}"
    )
    print(salida / "images")


if __name__ == "__main__":
    main()
