"""Descarga los límites municipales de Cantabria desde el WFS del IGN."""

from io import BytesIO

import geopandas as gpd
import requests

from paneles_solares.rutas import MUNICIPIOS_CANTABRIA

SALIDA = MUNICIPIOS_CANTABRIA
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

    municipios = municipios[
        municipios["nationalCode"].astype(str).str.startswith("340639")
    ]
    municipios = municipios[["nationalCode", "text", "geometry"]].rename(
        columns={"nationalCode": "codigo", "text": "municipio"}
    )

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    municipios["codigo"] = municipios.codigo.astype(str)
    municipios["tipo_unidad"] = "municipio"
    municipios.loc[municipios.codigo == "34063953072", "tipo_unidad"] = (
        "territorio_comunal"
    )
    SALIDA.unlink(missing_ok=True)
    municipios.to_file(SALIDA, layer="municipios", driver="GPKG", index=False)
    gpd.GeoDataFrame({"geometry": [cantabria]}, crs=municipios.crs).to_file(
        SALIDA, layer="cantabria", driver="GPKG", index=False
    )
    print(f"{len(municipios)} unidades (incluido territorio comunal): {SALIDA}")


if __name__ == "__main__":
    main()
