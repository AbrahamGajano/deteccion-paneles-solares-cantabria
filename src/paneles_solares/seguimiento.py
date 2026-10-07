"""Único punto de entrada: preparar, revisar, auditar y exportar el seguimiento."""

import argparse
import tkinter as tk

from paneles_solares.datos.seguimiento import AlmacenSeguimiento, cargar_config


def main():
    """Continúa lotes pequeños o ejecuta una fase explícita sin reiniciar el progreso."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "accion",
        nargs="?",
        default="continuar",
        choices=("continuar", "preparar", "auditar", "resumen", "region"),
    )
    parser.add_argument("--config", default="seguimiento.json")
    parser.add_argument("--lote", type=int)
    parser.add_argument(
        "--sitio", action="append", help="Prepara este emplazamiento; admite repetir la opción"
    )
    parser.add_argument(
        "--modo", choices=("indice", "fuera_indice", "todas"), default="fuera_indice"
    )
    parser.add_argument(
        "--gpkg", action="store_true", help="Exporta también geometrías a GeoPackage"
    )
    args = parser.parse_args()
    config = cargar_config(args.config)
    lote = config.get("lote", 12) if args.lote is None else args.lote
    if lote < 1:
        parser.error("El lote debe ser positivo")
    almacen = AlmacenSeguimiento()
    try:
        from paneles_solares.etiquetado.preparar_seguimiento import (
            analizar_desbordes,
            importar_piloto,
            incorporar_inventario,
            incorporar_muestra_previa,
            incorporar_predicciones_existentes,
            preparar_lote,
            preparar_region,
        )
        from paneles_solares.modelos.resumir_seguimiento import (
            SALIDA,
            generar_auditoria,
            generar_resumen,
        )

        importar_piloto(almacen, config)
        incorporar_inventario(almacen, config)
        incorporar_muestra_previa(almacen, config.get("muestra_previa", []))
        if args.accion == "auditar":
            preparar_lote(almacen, config, lote, recuperar=True)
            incorporar_predicciones_existentes(almacen)
            analizar_desbordes(almacen)
            generar_auditoria(almacen)
        elif args.accion == "region":
            preparar_region(almacen, config, lote, args.modo)
            generar_auditoria(almacen)
        elif args.accion in ("continuar", "preparar"):
            if not almacen.meta("auditoria_generada"):
                preparar_lote(almacen, config, lote, recuperar=True)
                analizar_desbordes(almacen)
                generar_auditoria(almacen)
            if args.accion == "preparar" or args.sitio:
                preparar_lote(almacen, config, lote, args.sitio)
        resumen = generar_resumen(almacen, geopackage=args.gpkg)
        if args.accion == "continuar":
            from paneles_solares.visualizacion.revisar_seguimiento import (
                VisorSeguimiento,
            )

            raiz = tk.Tk()
            VisorSeguimiento(raiz, almacen, config, lote)
            raiz.mainloop()
            return
        print(
            f"Históricos: {resumen['historicos_guardados']} guardados; {resumen['historicos_pendientes']} pendientes.\n"
            f"Referencias 2023 revisadas: {resumen['referencias_2023_revisadas']}; se confirman dentro del primer par.\n"
            f"Resultados: {SALIDA}"
        )
    finally:
        almacen.cerrar()


if __name__ == "__main__":
    main()
