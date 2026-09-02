from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


ruta_tif = Path(
    "data/raw/pnoa/PNOA_MA_OF_ETRS89_HU30_h25_0035_1.tif"
)

salida = Path("outputs/tesela_prueba.png")


with rasterio.open(ruta_tif) as tif:

    # Creamos una versión pequeña de toda la ortofoto.
    ancho_previo = 1200
    proporcion = tif.height / tif.width
    alto_previo = int(ancho_previo * proporcion)

    vista_previa = tif.read(
        [1, 2, 3],
        out_shape=(3, alto_previo, ancho_previo),
        resampling=Resampling.bilinear,
    )

    vista_previa = np.moveaxis(vista_previa, 0, -1)

    # Mostramos la ortofoto completa para elegir una zona.
    plt.imshow(vista_previa)
    plt.title("Haz clic en una zona con edificios")
    plt.axis("off")

    punto = plt.ginput(1, timeout=0)[0]
    plt.close()

    x_click, y_click = punto

    # Transformamos el clic de la imagen pequeña
    # a los píxeles reales del TIFF.
    columna_centro = int(x_click * tif.width / ancho_previo)
    fila_centro = int(y_click * tif.height / alto_previo)

    # Colocamos el centro de la ventana sobre el punto elegido.
    columna = columna_centro - 256
    fila = fila_centro - 256

    # Impedimos que la ventana salga fuera del TIFF.
    columna = max(0, min(columna, tif.width - 512))
    fila = max(0, min(fila, tif.height - 512))

    ventana = Window(
        columna,
        fila,
        512,
        512,
    )

    imagen = tif.read(
        [1, 2, 3],
        window=ventana,
    )


imagen = np.moveaxis(imagen, 0, -1)

salida.parent.mkdir(parents=True, exist_ok=True)
plt.imsave(salida, imagen)

plt.imshow(imagen)
plt.title("Tesela seleccionada de 512 × 512")
plt.axis("off")
plt.show()

print("Fila:", fila)
print("Columna:", columna)
print("Guardada en:", salida)