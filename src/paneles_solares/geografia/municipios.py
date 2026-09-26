"""Descarga los límites municipales de Cantabria desde el WFS del IGN."""

from io import BytesIO

import geopandas as gpd
import requests

from paneles_solares.rutas import ruta_proyecto

SALIDA = ruta_proyecto("data/geografia/municipios_cantabria.gpkg")
URL = "https://www.ign.es/wfs-inspire/unidades-administrativas"


def main() -> None:
    """Guarda los municipios oficiales en un GeoPackage."""
    respuesta = requests.get(
        URL,
        params={
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": "au:AdministrativeUnit",
            "count": 1000,
            "bbox": "340000,4730000,510000,4830000,EPSG:25830",
            "srsName": "EPSG:25830",
        },
        timeout=120,
    )
    respuesta.raise_for_status()
    unidades = gpd.read_file(BytesIO(respuesta.content))

    cantabria = unidades.loc[
        (unidades["text"] == "Cantabria")
        & (unidades["LocalisedCharacterString"] == "Comunidad autónoma"),
        "geometry",
    ].iloc[0]
    municipios = unidades[unidades["LocalisedCharacterString"] == "Municipio"].copy()
    municipios = municipios[municipios.representative_point().within(cantabria)]
    municipios = municipios[municipios["nationalCode"].astype(str).str.startswith("34063939")]
    municipios = municipios[["nationalCode", "text", "geometry"]].rename(
        columns={"nationalCode": "codigo", "text": "municipio"}
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    municipios.to_file(SALIDA, driver="GPKG")
    print(f"{len(municipios)} municipios: {SALIDA}")


if __name__ == "__main__":
    main()
