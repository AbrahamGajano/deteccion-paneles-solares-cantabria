# Paneles solares en Cantabria

Abre el archivo que necesitas, cambia sus variables de arriba y pulsa **Run Python File**
en VS Code. Selecciona el intérprete `.venv/Scripts/python.exe`.
Cada proceso tiene su propio `main()`; no hay argumentos por consola ni configuración JSON.

## Flujo de trabajo

1. **`etiquetado/crear_labeling.py`** extrae imágenes nuevas desde PNOA.
   Por defecto pide 1.000 aleatorias, sin ejecutar YOLO. Puedes combinar cuotas
   aleatorias, de confianza intermedia, muchas detecciones y sin detección.
   No vuelve a pedir imágenes del pool ni de labeling.
2. **Etiqueta en LabelMe** las imágenes de `data/labeling/images/` y guarda
   los JSON en `data/labeling/annotations/`, con el mismo nombre.
   Clase: `panel_solar`, polígonos. Un negativo necesita un JSON guardado
   sin polígonos; no tener JSON significa que falta revisarlo.
   Usa las flags `dudosa` o `descartar` para excluir una imagen.
3. **`etiquetado/incorporar_labeling.py`** mueve los pares revisados al pool.
   Las descartadas también se conservan, con `uso=no_usar`.
   Lo incompleto o incorrecto se queda en labeling para corregirlo.
   Al terminar todo, labeling queda vacío de imágenes y JSON; su CSV queda sin filas.
4. **`datos/decidir_dataset.py`** decide sobre las muestras con `uso=pendiente`.
   Incluye las positivas y compara las negativas con el modelo que elijas:
   si detecta algo por encima del umbral, es un negativo difícil.
   Puedes limitar cuántos negativos difíciles y fáciles incluyes.
5. **`datos/sincronizar.py`** actualiza el mismo dataset YOLO.
   Genera los TXT desde LabelMe y refleja incorporaciones, cambios de conjunto y retiradas.
6. **`modelos/entrenar_yolo.py`** entrena. **`modelos/evaluar_yolo.py`**
   guarda la revisión visual del test.

Los archivos están dentro de `src/paneles_solares/`.

## Carpetas

```text
data/
  labeling/
    images/                PNG pendientes
    annotations/           JSON de LabelMe pendientes
    manifest.csv           procedencia de esas imágenes

  pool/
    images/                todos los PNG revisados
    annotations/           JSON originales de LabelMe
    manifest.csv           estado y uso de cada imagen

  datasets/yolo/
    images/train, val, test/
    labels/train, val, test/
    dataset.yaml

  pnoa/                    ortofotos originales
  geografia/catastro/       edificios preparados
  manifests/teselas.csv     índice de todas las teselas candidatas
  raw/catastro/             descargas originales de Catastro
```

Los PNG del dataset son **enlaces duros** a los del pool: no duplican sus píxeles
en disco. No edites esas imágenes; edita los JSON originales y vuelve a sincronizar.
LabelMe es la anotación original reutilizable; actualmente solo se generan etiquetas YOLO.

## Qué significa el manifiesto

- `estado`: `pendiente` en labeling; `valida` o `descartada` en el pool.
- `tipo`: `positiva` o `negativa`, según tu anotación.
- `uso`: `pendiente`, `train`, `val`, `test` o `no_usar`.
- `uso_anterior`: conserva el conjunto de una muestra retirada durante la migración.
- `motivo`: por qué se seleccionó, excluyó o necesita revisión.
- `modelo_yolo`, `confianza_yolo`, `detecciones_yolo`, `umbral_yolo`:
  predicción usada para decidir. Vacío significa desconocido, no ausencia de paneles.
- El resto conserva origen y posición en PNOA.

Puedes editar `uso` a mano. Para reconsiderar una muestra válida marcada `no_usar`,
cámbiala a `pendiente` y ejecuta decidir; para forzar su inclusión, asígnale un conjunto.
Después ejecuta sincronizar. Las asignaciones existentes no se sortean otra vez.

## Entrenamiento y evaluación

Los PNG originales son de 512 × 512. `IMGSZ = 640` indica el tamaño de entrada
al que YOLO los adapta; no cambia los archivos originales. Se mantiene 640
porque los pesos actuales se entrenaron con ese tamaño. También puedes usar 512,
pero conviene comparar resultados antes de cambiarlo en todos los procesos.

Las nuevas muestras van por defecto a train: ya tienes validación y test.
Las fracciones se cambian en `decidir_dataset.py`. El reparto es aproximado por
grupos espaciales, no un número exacto de imágenes. Los casos buscados por YOLO
van a train; val/test nuevos usan muestras aleatorias sin filtrar los fondos fáciles.
El script evita añadir vecinos a otro conjunto; no rehace la separación histórica.

Las negativas fáciles no son inútiles por definición: puedes incluir algunas.
La selección por confianza es una heurística, no una medida fiable de incertidumbre.
Una positiva detectada tampoco implica que todas sus máscaras sean correctas.

En entrenamientos se conservan `best.pt`, `last.pt`, parámetros y resultados.
Cada ejecución completa sustituye a la anterior de ese modelo.
Las revisiones siguen en `runs/evaluacion/cantabria/{yolo11n,yolo11s}/revision/`.
Incluyen positivas detectadas, positivas sin detección, posibles falsos positivos
y un CSV con todas las imágenes. Las negativas correctas se registran solo en el CSV.
Un fallo conserva el resultado anterior y puede dejar una carpeta `.*_preparando`
para inspeccionarla antes de repetir.

## Datos migrados

Se conservan los 6.600 PNG y 6.596 JSON. Hay 6.592 imágenes en el pool
(81 descartadas) y 8 en labeling: 4 sin JSON y 4 con polígonos fuera de la imagen.
No se han corregido sus anotaciones automáticamente.
El dataset válido contiene 1.429 train, 850 val y 988 test.
Las revisiones y pesos anteriores se conservaron tal como estaban; sus resultados
pertenecen al dataset con el que se generaron, no a una evaluación nueva.

## Instalación en otro equipo

Crea un entorno Python 3.10 o posterior y ejecuta una vez:
`python -m pip install -e .`. La instalación permite compartir funciones entre
archivos aunque los ejecutes individualmente. Los `__init__.py` solo identifican
las carpetas del paquete; no son procesos que tengas que ejecutar.
