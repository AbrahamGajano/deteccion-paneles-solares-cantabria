# Paneles solares en Cantabria

El proyecto busca paneles solares en las ortofotos PNOA de Cantabria. Catastro
se utiliza para evitar recorrer zonas sin edificios y los resultados se pueden
ver en mapas o convertir en una estimación aproximada de energía con PVGIS.

Cada archivo se puede ejecutar desde VS Code. Las opciones que normalmente
querrás cambiar están al principio del archivo.

## Preparación

Crea un entorno con Python 3.10 o posterior e instala el proyecto:

```powershell
python -m pip install -e .
```

Esto instala las librerías de `requirements.txt` y permite que los archivos de
`src` se importen entre sí.

## Etiquetado y entrenamiento

El flujo normal es este:

1. `etiquetado/crear_labeling.py` extrae nuevas teselas de PNOA.
2. Etiqueta sus polígonos en LabelMe con la clase `panel_solar`.
   Una imagen sin paneles también necesita un JSON vacío.
3. `etiquetado/incorporar_labeling.py` mueve lo terminado al pool.
4. `datos/decidir_dataset.py` reparte las imágenes entre entrenamiento y
   validación. También conserva los negativos en los que fallan los modelos.
5. `datos/sincronizar.py` actualiza los datasets de YOLO y U-Net.
6. Entrena con `modelos/entrenar_yolo.py` o `modelos/entrenar_unet.py`.
7. Revisa el resultado con `modelos/evaluar_yolo.py` o
   `modelos/evaluar_unet.py`.

Las anotaciones originales siempre son los JSON de `data/pool/annotations`.
Los datasets se generan a partir de ellos, así que no conviene editar sus TXT o
máscaras a mano.

## Cantabria completa

`data/manifests/teselas.csv` contiene las teselas que intersectan edificios de
Catastro. Para generar resultados de toda Cantabria:

1. Ejecuta `geografia/municipios.py` para descargar los límites municipales.
2. Elige YOLO o U-Net al principio de `modelos/inferir_cantabria.py` y ejecútalo.
3. Ejecuta `visualizacion/mapa_cantabria.py` para crear el mapa general y el
   resumen por municipio.

La inferencia se guarda por bloques en `runs/inferencia`. Si se interrumpe,
continúa desde el último bloque terminado. Si cambias el modelo, los pesos o el
umbral usando el mismo nombre, elimina los resultados anteriores y comienza de
nuevo automáticamente.

`visualizacion/mapa_resultados.py` crea un mapa del conjunto de evaluación.
Sirve para localizar aciertos y errores, pero no representa toda Cantabria.

## Indicadores U-Net por edificio y municipio

`modelos/indicadores_unet.py` aplica `weights/unet_resnet34.pt` a las teselas
PNOA del índice `data/manifests/teselas.csv`. Usa RGB, 512 × 512 píxeles,
normalización ImageNet y umbral 0,5, como `evaluar_unet.py`. Las teselas de
borde menores de 512 píxeles se completan con negro y se recortan tras inferir.
Las máscaras se mantienen en el CRS original EPSG:25830 y se cuentan por
píxel dentro de un único edificio. En los solapes entre ortofotos prevalece
la primera por nombre. Solo entran en los indicadores los edificios con
cobertura completa en las teselas procesadas.
Los edificios cuyo punto interior cae fuera de los límites municipales se
asignan al municipio más cercano (13 casos en los datos actuales).

Ejecuta desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_unet
```

La ejecución sobrescribe tres archivos fijos en `runs/indicadores/unet_actual`:

- `edificios.gpkg`: geometría catastral original, `fid` local como
  `id_edificio`, municipio, superficie de cubierta, superficie fotovoltaica
  detectada, ocupación y presencia.
- `municipios.csv`: conteos y tasas por edificio, superficies detectadas,
  media y mediana entre edificios positivos y ocupación total de cubierta.
- `municipios.gpkg`: los mismos indicadores municipales con sus límites para QGIS.

La superficie detectada es planimétrica: píxeles positivos dentro de la
huella catastral × 0,0225 m². La cubierta se aproxima mediante la huella
catastral proyectada; no se calcula la superficie inclinada real de los
módulos. `superficie_fv_corregida_test_m2` y
`superficie_fv_corregida_test_total_m2` son una **calibración experimental**:
superficie detectada / 0,9246, según el sesgo global de −7,54 % medido en
el test. Siempre se conserva también la superficie sin corregir. Este sesgo
global no garantiza una corrección válida para cada edificio o municipio.
La presencia indica píxeles segmentados dentro de la huella, no el número
de instalaciones; no se aplica un filtro de área mínima. Algunos porcentajes
de ocupación pueden superar 100 % por errores de segmentación o por la
aproximación de píxeles a los límites de Catastro.

## Estimación de energía

`visualizacion/estimar_energia.py` agrupa los paneles detectados en zonas de
3 km y consulta PVGIS. Genera:

- `energia_muestras.csv`, con las consultas individuales.
- `energia_mensual.csv`, con el resumen por municipio y mes.
- `energia_mensual.png`, con la gráfica mensual.
- `mapa_energia.html`, con la energía anual por municipio.

La cifra es un escenario aproximado. Actualmente supone 0,20 kWp por cada m²
proyectado en la ortofoto, orientación sur, 30° de inclinación y 14 % de
pérdidas. No conoce la potencia real de los paneles, su orientación ni las
sombras cercanas.

## Carpetas

```text
data/labeling/       imágenes pendientes de etiquetar
data/pool/           imágenes y anotaciones ya revisadas
data/datasets/       datasets generados para YOLO y U-Net
data/pnoa/           ortofotos originales
src/paneles_solares/ código del proyecto
runs/                entrenamientos, evaluaciones, inferencias y mapas
weights/             pesos de U-Net
```

Los PNG de los datasets son enlaces a los del pool y no ocupan el espacio dos
veces.
