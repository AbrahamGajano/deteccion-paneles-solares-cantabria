# Detección de paneles solares en Cantabria

Este proyecto consiste en intentar detectar paneles solares en Cantabria utilizando las ortofotos del PNOA y la información de los edificios de Catastro.

La idea es entrenar un modelo de segmentación para que pueda localizar los paneles y calcular aproximadamente la superficie que ocupan. Con estos resultados se podría estudiar su distribución por zonas y crear un mapa de calor.

## Datos utilizados

Para realizar el proyecto se están utilizando:

- Ortofotos PNOA del IGN con una resolución de 15 cm por píxel.
- Geometrías de los edificios descargadas de Catastro.
- Más adelante se podrá estudiar el uso de datos LiDAR para conocer la inclinación de las cubiertas.

Los datos originales no se guardan en GitHub porque ocupan demasiado espacio.

## Proceso

El proceso que se pretende seguir es el siguiente:

1. Descargar los edificios de Catastro de todos los municipios de Cantabria.
2. Unirlos en un único GeoPackage.
3. Recorrer las ortofotos y localizar las teselas de 512 × 512 píxeles que contienen algún edificio.
4. Guardar las posiciones de esas teselas en un manifest.
5. Seleccionar una muestra y convertirla en imágenes PNG.
6. Etiquetar manualmente los paneles con LabelMe.
7. Entrenar un modelo YOLO de segmentación.
8. Aplicar el modelo sobre las ortofotos de Cantabria.

## Estado actual

Por el momento se ha realizado:

- La descarga y unión de los edificios de Catastro.
- La creación del GeoPackage de edificios de Cantabria.
- El filtrado de las teselas que intersectan con edificios.
- La creación del manifest con las teselas encontradas.
- El inicio del código para preparar las imágenes que se etiquetarán con LabelMe.

Todavía faltan el etiquetado, el entrenamiento del modelo y la generación de los resultados finales.

## Estructura de carpetas

data/
├── pnoa/                Datos del satelite
├── raw/                 Datos originales
├── processed/           Datos ya procesados y manifests
└── labeling/            Imágenes y anotaciones de LabelMe

src/                     Código del proyecto
requirements.txt         Librerías necesarias