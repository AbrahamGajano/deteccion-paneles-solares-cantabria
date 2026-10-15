# Paneles solares en Cantabria

El proyecto detecta superficies fotovoltaicas en ortofotos PNOA de 2023 y las
resume por edificio y municipio. Compara U-Net con YOLO y estima potencia y
producción eléctrica. También permite revisar en qué campañas históricas
aparecen los paneles.

El recorrido es: **ortofotos → etiquetado → entrenamiento → detección →
indicadores → estimación de energía**. La revisión histórica se hace después,
con ayuda de un visor y decisiones manuales.

## Qué hay en cada carpeta

| Carpeta | Para qué sirve |
| --- | --- |
| `src/paneles_solares/` | Código del proyecto, separado por etapas. |
| `data/` | Ortofotos, geografía, etiquetas originales y datasets preparados. |
| `weights/` | Pesos de los modelos actuales y anteriores. |
| `runs/` | Evaluaciones, mapas, indicadores y resultados de las ejecuciones. |
| `docs/` | Memoria técnica, figuras y guía detallada. |
| `outputs/` | Auditoría de datos conservada de una ejecución anterior. |

Dentro del código, `geografia/` prepara los datos geográficos; `etiquetado/`
permite seleccionar y anotar imágenes; `datos/` organiza los datasets;
`modelos/` entrena, evalúa y detecta; `visualizacion/` genera mapas y el visor.
`rutas.py` reúne las rutas compartidas.

## Preparar el entorno

Se necesita Python 3.10 o posterior. Ejecuta los comandos desde la raíz:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

`pyproject.toml` permite instalar el proyecto y encontrar sus módulos.
`requirements.txt` contiene las bibliotecas necesarias.

Las ortofotos originales deben estar en `data/pnoa/`. Las anotaciones manuales
se conservan en `data/pool/annotations/`; las máscaras U-Net y etiquetas YOLO
se generan desde esos JSON.

## Ejecutar el flujo

Prepara Catastro, municipios e índice de teselas cuando cambien los datos:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.geografia.catastro
.\.venv\Scripts\python.exe -m paneles_solares.geografia.municipios
.\.venv\Scripts\python.exe -m paneles_solares.geografia.teselas
```

Para ampliar el etiquetado, selecciona imágenes, anótalas en LabelMe con la
clase `panel_solar` y luego incorpóralas al dataset. Los negativos también
necesitan su JSON vacío:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.crear_labeling
# Etiquetar los PNG de data/labeling/images/ antes del siguiente paso.
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.incorporar_labeling
.\.venv\Scripts\python.exe -m paneles_solares.datos.decidir_dataset
.\.venv\Scripts\python.exe -m paneles_solares.datos.sincronizar
```

Entrena y evalúa cada modelo por separado. Estos comandos pueden sustituir
pesos o resultados anteriores; revisa las constantes de los scripts antes
de ejecutarlos:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.entrenar_unet
.\.venv\Scripts\python.exe -m paneles_solares.modelos.evaluar_unet
.\.venv\Scripts\python.exe -m paneles_solares.modelos.entrenar_yolo
.\.venv\Scripts\python.exe -m paneles_solares.modelos.evaluar_yolo
```

U-Net es el modelo utilizado para los indicadores regionales. La inferencia
guarda GeoPackage de edificios y municipios; después se estima energía con
PVGIS y se genera el panel:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_unet
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.estimar_energia
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.panel_indicadores
```

Las salidas quedan en `runs/indicadores/unet_actual/`. Si ya existen los
indicadores y la producción, basta el último comando para regenerar el panel:
no repite inferencia ni consultas PVGIS.

## Validación y revisión histórica

La [guía](docs/guia.md) detalla la validación a otra resolución, el visor
histórico, sus resultados y sus limitaciones. Las opciones se editan junto
a cada flujo, sin archivos de configuración separados:

| Opciones | Dónde se cambian |
| --- | --- |
| Campañas históricas, lote y precarga | `CONFIGURACION` en [datos/seguimiento.py](src/paneles_solares/datos/seguimiento.py). |
| Campaña de validación y tamaño de muestra | `CAMPANIAS` en [datos/validacion.py](src/paneles_solares/datos/validacion.py). |

Para continuar la revisión o generar un avance sin modificar las respuestas:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos
```

El progreso manual queda en `data/datasets/seguimiento/seguimiento.sqlite`.
La recuperación histórica regional sigue pendiente.

## Interpretar los resultados

Una detección es una predicción del modelo. La superficie es planimétrica;
la corrección `superficie_detectada / 0.9246` utiliza el sesgo del test de 2023
y no garantiza precisión por edificio. La evaluación U-Net de ese test obtuvo
Dice 0,856 y sesgo de superficie −7,54 %.

La potencia y la energía son estimaciones. Los escenarios usan 0,17 kWp/m²
sobre superficie detectada, o 0,20 y 0,23 sobre superficie corregida.
PVGIS supone 30° de inclinación, orientación sur y pérdidas del 14 %.
El perfil de 8.760 horas es representativo, no producción medida.

La [memoria técnica](docs/memoria_tecnica.pdf) reúne metodología, resultados
y limitaciones. Su fuente editable es `docs/memoria_tecnica.tex`; requiere
LaTeX para recompilarla. Los mapas se regeneran con:

```powershell
.\.venv\Scripts\python.exe docs/generar_mapas.py
```
