# Paneles solares en Cantabria

Detección de paneles en ortofotos PNOA, etiquetado con LabelMe y análisis de
superficie y periodos de instalación. U-Net y YOLO comparten las anotaciones.
Las edificaciones y parcelas aportan contexto: no se excluyen terrenos,
aparcamientos ni zonas sin edificios.

## Entorno y datos

Python 3.10 o posterior. Ejecutar desde la raíz del repositorio:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Los datos y pesos no se distribuyen con el código:

| Carpeta | Contenido |
| --- | --- |
| `data/pnoa/` | Ortofotos PNOA 2023 locales: RGB, EPSG:25830, 0,15 m. |
| `data/raw/catastro/2023/` | Descargas oficiales y licencia. |
| `data/geografia/` | Límites municipales, edificaciones y parcelas preparadas. |
| `data/pool/` | Imágenes y anotaciones originales del dataset. |
| `data/datasets/` | Exportaciones para modelos y revisiones de otras campañas. |
| `weights/`, `runs/entrenamiento/` | Pesos de U-Net y YOLO. |
| `runs/` | Evaluaciones, indicadores y visualización. |

Las ortofotos se obtienen del Gobierno de Cantabria o del CNIG. El etiquetado
histórico también puede descargar recortes del WCS sin tener el TIFF completo.
Para inferir se necesitan los pesos; para volver a entrenar, el dataset y los
pesos iniciales definidos en `modelos/entrenar_unet.py` o `entrenar_yolo.py`.

## 1. Preparar la geografía de 2023

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.geografia.municipios
.\.venv\Scripts\python.exe -m paneles_solares.geografia.catastro
.\.venv\Scripts\python.exe -m paneles_solares.geografia.regenerar
```

Se utilizan el parcelario oficial de principios de 2023 y CBT23/LULC23, derivado
de Catastro y otras fuentes de 2023. Las unidades cartográficas no equivalen
necesariamente a edificios físicos completos. Se conservan sus usos oficiales
sin inferir titularidad pública ni tipo de vivienda.

`regenerar` prepara el índice territorial y actualiza el contexto de los
manifiestos. Retira indicadores, avances y paneles invalidados por el cambio de
referencia; no se ejecuta para reanudar una inferencia. Las etiquetas se conservan.
La [revisión de la referencia](docs/revision_catastro_2023.md) recoge las fuentes.

## 2. Etiquetar y preparar el dataset

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.crear_labeling
.\.venv\Scripts\labelme.exe data/labeling/images --output data/labeling/annotations --labels panel_solar --flags dudosa,descartar --config '{with_image_data: false, auto_save: true, keep_prev: false}'
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.incorporar_labeling
.\.venv\Scripts\python.exe -m paneles_solares.datos.decidir_dataset
.\.venv\Scripts\python.exe -m paneles_solares.datos.sincronizar
```

Se dibujan polígonos `panel_solar`. Una imagen negativa necesita un JSON vacío
guardado; no se interpreta la falta de anotación como ausencia. Las banderas
`dudosa` y `descartar` excluyen imágenes. Validación y test se mantienen separados
al incorporar nuevas muestras. `sincronizar` exporta máscaras U-Net y etiquetas
YOLO desde el mismo pool.

Los módulos `modelos.entrenar_unet`, `modelos.evaluar_unet`,
`modelos.entrenar_yolo` y `modelos.evaluar_yolo` usan esas exportaciones.

## 3. Inferir sobre las ortofotos locales

Probar primero con una zona pequeña:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_unet --bbox 387515.6 4799608.4 387692.4 4799785.2
```

La prueba se guarda en `runs/prueba_2023/indicadores/`. Para procesar todo el
territorio local o reanudar desde la última hoja terminada:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_unet
```

El avance está en `runs/indicadores/.unet_actual_avance/`; se comprueban pesos,
referencias, índice y código para no mezclar ejecuciones. La exportación regional
se publica en `runs/indicadores/unet_actual/` al terminar.

Se conservan detecciones completas, cobertura procesada y relaciones con
edificios, parcelas y municipios. `comparacion_cobertura.csv` mide el área dentro
y fuera de edificios. Los recuentos son componentes detectados, no instalaciones.

Después de terminar y revisar la inferencia regional, el escenario energético
uniforme se calcula con:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.estimar_energia
```

Exporta producción anual y mensual en CSV y una figura. Usa densidades de
0,17/0,20/0,23 kWp/m², orientación sur, inclinación de 30° y pérdidas del 14 %.
Es un escenario agregado sin periodos individuales, distinto del seguimiento.

## 4. Etiquetar otras campañas

Para un lote nuevo se parte de un CSV con `ventana,xmin,ymin,xmax,ymax`, en
EPSG:25830. Incluir también la campaña de referencia permite revisar su área.
No hace falta la base de datos del antiguo visor.

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --ventanas ventanas.csv --anios 2023 2020 2017 2014
.\.venv\Scripts\labelme.exe data/datasets/seguimiento/labelme/images --labels panel_solar --flags ausente,dudosa --config '{with_image_data: false, auto_save: true, keep_prev: false, display_label_popup: false}'
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --finalizar data/datasets/seguimiento/labelme/manifest.csv
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos
```

Dibujar todos los paneles de cada imagen, con polígonos separados y sin Group ID.
Marcar `ausente` si no hay ninguno y `dudosa` si no se puede decidir. Los PNG se
preparan a resolución nativa; los JSON quedan a su lado. Repetir la preparación
conserva el trabajo. `--config` permite definir otras campañas.

`--finalizar` registra qué imágenes están terminadas. El autoguardado de LabelMe
también conserva trabajo parcial, por lo que guardar un JSON no cierra la revisión.
Una anotación modificada después del cierre necesita finalizarse otra vez.
El seguimiento necesita algún polígono positivo en al menos una campaña;
las otras campañas pueden estar completamente vacías.

El [ejemplo pequeño y la guía](docs/guia.md#etiquetado-histórico) explican los
encuadres, la reanudación, la recuperación antigua y la incorporación de negativos.

Si la referencia de 2023 procede de U-Net, se usa su salida vectorial en el
mismo análisis, sin base de datos del visor:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos --manifest data/datasets/seguimiento/labelme/manifest.csv --detecciones runs/indicadores/unet_actual/detecciones.gpkg
```

Se leen los polígonos de la capa `detecciones` que intersectan las ventanas
del manifiesto y se conservan completos. Las campañas anteriores proceden de
LabelMe; los dibujos de 2023 revisados, si los hay, tienen prioridad sobre
la superficie estimada. Las campañas sin evidencia permanecen desconocidas.
`--limite 3 --salida runs/prueba_historico/resultados` permite probar el flujo
sin sustituir el resultado habitual. El GeoPackage se genera siempre con el
CSV: no requiere conversiones ni pasos manuales adicionales.

### Lote histórico ya revisado de este proyecto

Las 1.486 imágenes actuales están cerradas. Para regenerar sus resultados,
aprovechando las respuestas antiguas y la referencia estimada de 2023:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos --base data/datasets/seguimiento/seguimiento.sqlite
```

`--base` es una opción de recuperación, no un requisito del flujo general.
Sus respuestas se leen sin modificar la base. Las agrupaciones, superficies
y periodos se guardan en `runs/indicadores/seguimiento_historico/`.
Para probar un subconjunto, usar `--limite 20 --salida runs/prueba_historico/resultados`.

La salida tiene tres archivos:

- `instalaciones.csv`: una fila por agrupación y campaña, con municipio,
  latitud, longitud, edificios, parcelas, estado, superficie y periodo de aparición.
  Incluye el cambio desde la campaña anterior cuando las áreas son comparables.
- `atribucion.gpkg`: polígonos de paneles, agrupaciones, encuadres revisados,
  edificios y parcelas. Las tablas de relaciones y respuestas antiguas se
  conservan dentro del mismo GeoPackage.
- `resumen.json`: recuentos y alcance del lote.

Las superficies son horizontales. Se distingue el área anotada de la estimada
por el modelo; una ausencia comprobada vale cero y un dato desconocido queda
vacío. Los identificadores agrupan provisionalmente por edificio y solape:
compartir parcela no confirma una instalación.

La capa `instalaciones` del GeoPackage contiene las mismas filas y campos del
CSV, con la geometría de los paneles de cada campaña. Se puede filtrar por
`anio`. En el lote actual, 2014, 2017 y 2020 proceden de los dibujos manuales
y 2023 procede del modelo. Las ausencias y observaciones sin área disponible
no tienen geometría. `emplazamientos` reúne la huella de todas las campañas
para localizar cada agrupación en el visor.

Para crear máscaras se utiliza `instalaciones` o la capa detallada `paneles`,
filtrada por `anio`, junto a la cobertura revisada de `observaciones`.
La [guía de máscaras](docs/guia.md#datos-para-máscaras-y-geotiff)
describe los campos y la diferencia entre ausencia y zona no estudiada.
Guardar los rásteres en `runs/mascaras_historicas/`, fuera de la carpeta de
indicadores que se sustituye al regenerar. El flujo actual no crea esos GeoTIFF.

## 5. Generar el panel

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.panel_indicadores
```

Abrir `runs/visualizacion/panel_indicadores.html` en el navegador. Se puede
enviar ese único archivo: incluye los datos, gráficos, Leaflet y sus recursos.
No necesita Python, el repositorio ni los CSV del ordenador que lo generó.
Los botones de descarga crean los CSV a partir de los datos incluidos.
Los filtros seleccionan campaña, municipio, uso cartográfico y estado de la atribución.
El mapa permite ver máscaras, edificios y parcelas sobre la ortofoto de cada año.

El mismo HTML tiene dos pestañas:

- **Mapa y resumen:** superficie, energía y consulta de cada emplazamiento.
- **Instalaciones y evolución:** distribución de tamaños, mediana y percentil 90,
  reparto de usos cartográficos, relación con edificios, periodos y cambios
  de superficie. Incluye una tabla buscable y la descarga de la selección.

Los gráficos por instalación usan agrupaciones con área positiva en la campaña
elegida; los ceros y valores desconocidos se conservan en la tabla. Los usos
proceden de las edificaciones CBT23/LULC23, no de una clasificación manual ni
de un supuesto tipo de parcela. El gráfico cuenta cada agrupación una vez y
mantiene «Sin dato». Los filtros se comparten entre ambas pestañas. La atribución
a edificios distingue, por campaña, coincidencia clara, parcial o dudosa y
ausencia de coincidencia con las huellas. Esta última no confirma que sea suelo.

El panel obtiene los años de las observaciones. La comparación usa una muestra
común con áreas revisadas en todas las campañas comparables. El inventario
conserva también estimaciones del modelo y estados pendientes. Un dato
desconocido no equivale a cero; los recuentos siguen siendo provisionales.

La sección de energía muestra tres escenarios, sus supuestos y fuentes.
La estimación corresponde al inventario observado, sin extrapolar a toda
Cantabria. Las áreas se agregan sin duplicar solapes.

En la misma carpeta se guardan los CSV municipales, producción estimada,
factores por emplazamiento, hipótesis y procedencia. Las consultas PVGIS se
conservan en `pvgis_anual.json`. El mapa abre sin fondo y funciona sin conexión,
con municipios y paneles incluidos. Las ortofotos y el mapa de calles se activan
desde el selector y requieren Internet. Los recursos de Leaflet 1.9.4 y su licencia
se conservan en `src/paneles_solares/visualizacion/plantillas/leaflet/`.
Se incluyen en el HTML al generarlo, sin descargas adicionales. `--carpeta`
y `--salida` permiten usar otra salida histórica, por ejemplo la prueba pequeña
descrita en la guía.

## Documentación

- [Guía de etiquetado, validación y seguimiento](docs/guia.md).
- [Memoria técnica](docs/memoria_tecnica.tex).
- [Fuentes catastrales de 2023](docs/revision_catastro_2023.md).

Los mapas estáticos incluidos en la memoria se regeneran después del panel:

```powershell
.\.venv\Scripts\python.exe docs/generar_mapas.py --anio 2023
```

La memoria se conserva en LaTeX; para obtener el PDF se necesita una instalación
de LaTeX y compilar desde `docs/` para resolver las rutas de las figuras.

Datos del Gobierno de Cantabria, Dirección General del Catastro e IGN/CNIG.
La licencia CBT23 se conserva en `data/raw/catastro/2023/licencia.txt`.
