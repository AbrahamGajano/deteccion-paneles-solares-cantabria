# Validación de otras resoluciones

El modelo actual se mantiene congelado: `weights/unet_resnet34.pt`, RGB,
normalización ImageNet, sigmoid y umbral 0,5. Se reutilizan `predecir`,
`calcular_indicadores`, `clasificar_resultado` y `crear_visualizacion` de
`evaluar_unet`, y la lectura y rasterización de polígonos de `datos/anotaciones`.
No intervienen YOLO, entrenamiento ni la corrección de superficie de 2023.

## Configuración y preparación

`validaciones.json` contiene una entrada por campaña. `20cm` ya está conectada a
la **Ortofoto 2024 True Ortho del Gobierno de Cantabria**, la campaña de 20 cm
más cercana al test de 2023. No hace falta descargar hojas ni completar rutas.
Se verificó con `DescribeCoverage` que la cobertura 16 del WCS tiene rejilla
nativa de 0,20 m y EPSG:25830. El año y el producto figuran también en los
[metadatos oficiales](https://geoservicios.cantabria.es/inspire/rest/services/Ortofoto_2024/MapServer).
`procedencia` identifica el producto, su campaña y el enlace de referencia.

`fuentes` admite la configuración WCS actual, rutas a GeoTIFF ya disponibles o
URL HTTP/HTTPS de COG con lectura parcial. Las rutas relativas parten de la raíz
del proyecto. Para otra campaña con datos locales, por ejemplo:

```json
"anio": 2024,
"procedencia": "Producto oficial, campaña 2024 y enlace a sus metadatos",
"fuentes": ["D:/ortofotos/hoja_20cm.tif"]
```

Se comprueban EPSG:25830, resolución nativa y RGB. No se aceptan como
validación nativa los TIF de 15 cm del proyecto ni una imagen WMS que simplemente
se haya solicitado a 20 cm. Para fuentes remotas se exige HTTP Range y bloques
pequeños; no se descarga el fichero completo ni se escribe una caché de ortofotos.
Rasterio lee los bloques que contienen cada ventana, que pueden ser algo mayores
que el recorte ([documentación de Rasterio](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html)).
El WCS devuelve únicamente la ventana pedida, como GeoTIFF en memoria, con las
bandas RGB 1, 2 y 3. La cobertura original es de 16 bits: se convierte a RGB de
8 bits con **división fija por 257 y redondeo** (0..65535 → 0..255), igual en todas
las teselas. No se aplica ajuste de contraste por tesela ni se conserva el
GeoTIFF intermedio o el infrarrojo. El cambio radiométrico queda identificado en
la configuración; no es una imagen tomada de la caché WMS ni una conversión
de las ortofotos de 15 cm del entrenamiento.
Las ventanas que cruzan hojas se unen solo en memoria y solo dentro de sus 76,8 m.
Si falta cobertura o hay nodata, se detiene la extracción y se conserva lo ya
completado; no se sustituyen esas localizaciones por otras elegidas a posteriori.

Desde la raíz:

```powershell
# Ya ejecutado: solo coordenadas y metadatos, sin descargar imágenes.
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_validacion 20cm --solo-seleccion

# Descargar únicamente las ventanas que falten (la fuente ya está configurada):
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_validacion 20cm
```

Se sortean primero 219 de las 988 localizaciones del test, sin consultar las
predicciones, y después se añaden hasta 31 positivos conocidos no incluidos ya.
Con semilla 42 quedan **243 teselas: 219 aleatorias y 24 positivos añadidos**.
Los 31 positivos de 2023 están presentes; 7 pertenecen al grupo aleatorio.
El grupo `positivos` describe el criterio de selección de 2023, no garantiza que
la imagen de la nueva campaña siga teniendo paneles.

Se conserva la extensión física exacta del test: 76,8 × 76,8 m. A 20 cm se guarda
un único PNG RGB de 384 × 384 por localización. Si las rejillas no coinciden,
Rasterio interpola el recorte para mantener ese mismo encuadre. El CSV conserva
límites, centro, CRS, año, resolución, fuente, semilla, grupo y referencia de 2023.
No se guardan copias a 512 ni máscaras derivadas: se crean en memoria al evaluar.

Una repetición reanuda las imágenes que faltan, sin tocar etiquetas. Se pueden
completar año y fuente mientras no haya imágenes extraídas. Una vez extraídas,
cambiar su origen o selección exige otra campaña para no invalidar las etiquetas.
Los parámetros de bootstrap y número de ejemplos sí pueden cambiarse.

## LabelMe

Después de extraer, abre cada grupo por separado:

```powershell
.\.venv\Scripts\labelme.exe data/datasets/validacion_20cm/images/positivos --output data/datasets/validacion_20cm/annotations/positivos --labels panel_solar --flags revisada,dudosa,descartar --config '{with_image_data: false}'
.\.venv\Scripts\labelme.exe data/datasets/validacion_20cm/images/aleatorias --output data/datasets/validacion_20cm/annotations/aleatorias --labels panel_solar --flags revisada,dudosa,descartar --config '{with_image_data: false}'
```

Dibuja polígonos `panel_solar` sobre el PNG de 384 × 384. Marca la bandera de
imagen `revisada` y guarda cada JSON, también para negativos, con `shapes: []`.
La bandera permite guardar una imagen vacía de forma explícita. La configuración
`with_image_data: false` (LabelMe 7.3 instalado) evita incluir una copia base64 de
la imagen dentro de cada JSON. `dudosa` o `descartar`
excluyen una imagen y su exclusión se registra en los resultados.

No uses `incorporar_labeling`, `decidir_dataset` ni `sincronizar` para esta
validación: sus etiquetas permanecen en la campaña y no entran en train/val/test.
Las anotaciones de 2023 sirven de referencia, pero deben comprobarse de nuevo en
la nueva ortofoto; no se toman automáticamente como verdad de la nueva campaña.

## Evaluación

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.evaluar_validacion 20cm
```

La evaluación exige que todas las teselas seleccionadas tengan imagen y JSON
revisado o exclusión explícita. No interpreta una etiqueta ausente como negativo
ni evalúa silenciosamente solo la parte etiquetada. Redimensiona RGB a 512 con
bilineal y máscaras con vecino más próximo. Un píxel evaluado corresponde a
0,15 × 0,15 m² por la extensión física, **no** a 0,20 × 0,20 m².

`runs/evaluacion/validacion_20cm/` contiene:

- `resultados.csv`: métricas por tesela y metadatos.
- `resumen.csv`: métricas de píxeles agregados para total y ambos grupos;
  diferencia respecto a esas mismas localizaciones de 2023, reevaluadas con los
  mismos pesos. Las diferencias son absolutas, no porcentajes de mejora.
- `referencia_15cm.csv`: resultados de esa reevaluación limitada de 2023.
- `comparacion_test_historico.csv`: comparación contextual con el test completo
  guardado en `unet_actual`, si existe. No prueba que sus pesos históricos fueran
  idénticos ni que su composición sea comparable a la muestra estratificada.
- `intervalos.csv`: bootstrap percentil 95%, 1.000 repeticiones por defecto;
  muestrea teselas con reemplazo dentro de cada grupo y mantiene los pares para
  calcular diferencias. Las réplicas donde una métrica es indefinida se omiten y
  se informa de cuántas quedaron. `bootstrap: 0` lo desactiva.
- `excluidas.csv` y `config.json`: exclusiones y configuración efectiva, incluyendo
  SHA256 de los pesos, umbral, procesamiento y fecha.
- `ejemplos/`: hasta dos peores errores de cada categoría, seis PNG como máximo.
  Verde: acierto; rojo: falso positivo; azul: falso negativo.

`fp` y `fn` son errores de **píxel**, como en `evaluar_unet`.
`imagenes_falso_positivo` cuenta imágenes sin panel real pero con algún píxel
predicho; `imagenes_falso_negativo` cuenta imágenes positivas sin ningún píxel
predicho. Las columnas `categoria_*` usan los límites de revisión originales
(más de 10 píxeles y más del 10 %), que también pueden señalar errores parciales.
No se cuentan instalaciones u objetos individuales. Precision/Recall/sesgo
pueden estar indefinidos si no hay predicciones o superficie real.

Se prepara una salida pequeña y se sustituye la anterior únicamente al terminar
correctamente, mediante `reemplazar_carpeta`. Un fallo conserva los resultados
anteriores y retira la salida incompleta del intento. No se crean carpetas numeradas.

## Ampliación y conclusiones

Añade una entrada `25cm_2025`, por ejemplo, y ejecuta los mismos módulos con ese
nombre. Mantén semilla y cantidades si quieres repetir las localizaciones. Para
25 cm, 76,8 / 0,25 = 307,2: se remuestrea el encuadre a 307 × 307 antes de entrar
a 512. Esa pequeña diferencia de rejilla queda descrita por límites y tamaño en
el CSV; no se cambia la extensión a 76,75 m. No hace falta duplicar scripts.

La prueba mide transferencia del modelo congelado al producto, año y lugares
seleccionados, y permite inspeccionar pérdidas de segmentación y sesgo de área.
El total enriquecido con positivos **no estima prevalencia**, superficie total
ni rendimiento representativo de toda Cantabria. El grupo aleatorio representa
el marco del test de 2023, que procede de teselas con edificios, no todo el
territorio. Las exclusiones pueden introducir sesgo y deben revisarse.

Comparar las mismas coordenadas reduce diferencias de composición, pero un año
distinto también cambia instalaciones, iluminación y calidad del producto. La
campaña 2024 añade además corrección True Ortho y radiometría de 16 bits; no
atribuye por sí solo todas las diferencias a la resolución. Con solo 31 positivos
conocidos la incertidumbre puede ser grande. El bootstrap por tesela no corrige
dependencia entre lugares cercanos ni error de etiquetado, y no tiene interpretación
de intervalo regional. No se aplica el −7,54 % del test de 2023 a esta campaña.

## Revisión y limpieza de data

Se revisaron todas las carpetas y las rutas utilizadas por los scripts. Antes:

```text
data/
  datasets/{unet,yolo}/     imágenes enlazadas al pool, máscaras y TXT derivados
  geografia/               municipios y edificios
  labeling/                1.000 PNG pendientes, manifest.csv y manifest_backup.csv
  manifests/teselas.csv     índice de teselas con edificios
  pnoa/                    65 TIF originales, 31 XML y 31 ZIP auxiliares
  pool/                    10.082 PNG y 10.082 JSON originales, manifest.csv
  raw/catastro/zips/        103 ZIP originales
```

Después se mantienen esas rutas de trabajo y se añade únicamente la campaña
dentro de `datasets`, donde ya están los datos utilizados para evaluación:

```text
data/datasets/validacion_20cm/
  config.json
  manifest.csv
  images/{positivos,aleatorias}/        243 PNG preparados de 2024 (62,8 MB)
  annotations/{positivos,aleatorias}/  salida de LabelMe, pendiente de etiquetar
```

Se eliminó `labeling/manifest_backup.csv`: sus 717 registros ya estaban
incorporados al pool (710 válidos y 7 descartados), con los mismos metadatos de
origen y coordenadas y todos sus PNG/JSON conservados. Se eliminaron las tres
cachés regenerables `datasets/yolo/labels/{train,val,test}.cache`.

No se movieron archivos: las rutas actuales son coherentes y los PNG de ambos
datasets **ya son enlaces físicos al pool**, no copias que ocupen espacio doble.
Se conservaron los datasets, etiquetas, geografía, originales de Catastro y PNOA,
los ZIP auxiliares PNOA (contienen geometrías, no copias de los TIF), y las 1.000
teselas pendientes de LabelMe. La carpeta vacía `labeling/annotations` sigue
siendo la salida utilizada por el flujo existente. Los modelos anteriores y
resultados históricos también se conservan: no consta que puedan descartarse.

Comprobaciones realizadas: pruebas automáticas de ventanas, cobertura,
revisión de negativos, cambio de configuración, métricas y bootstrap; las 9.532
parejas de imágenes/etiquetas de los datasets y sus enlaces al pool; metadatos y
anotaciones de referencia de las 243 localizaciones. Una inferencia real sobre
un positivo y un negativo reprodujo los conteos y métricas de `evaluar_unet`.
La extracción WCS se ha completado con el servicio oficial: 243 PNG RGB de 384 ×
384, no vacíos, con año y URL de cada recorte comprobados; unos 63 MB, sin TIF
intermedios persistentes. Las diez pruebas automáticas pasan, incluyendo conversión
fija de RGB de 16 a 8 bits y rechazo de otra resolución nativa en WCS.
La muestra ya está etiquetada y evaluada; el resultado se describe a continuación.
Las pruebas se repiten con:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Resultado de la muestra etiquetada

Se evaluaron 240 de las 243 teselas: 33 positivas y 207 negativas. Las tres
exclusiones corresponden a las banderas manuales `dudosa` o `descartar`, no a
sus predicciones. Los mismos pesos congelados y umbral 0,5 se usaron tanto en
2024 como en las mismas 240 localizaciones de 2023.

| Métrica | Referencia 2023, 15 cm | Campaña 2024, 20 cm |
| --- | ---: | ---: |
| Dice | 0,8657 | 0,7254 |
| IoU | 0,7631 | 0,5692 |
| Precision | 0,9121 | 0,9141 |
| Recall | 0,8237 | 0,6013 |
| Sesgo de superficie | −9,69 % | −34,22 % |

El bootstrap por tesela y grupo da IC95 % para el Dice de 2024 de [0,6003;
0,8345] y para la diferencia respecto a 2023 de [−0,2511; −0,0391]. Su alcance
es esta muestra estratificada y no el territorio regional.

En 2024 hay 3.296 píxeles falsos positivos y 23.259 falsos negativos. Ninguna de
las 207 imágenes negativas contiene píxeles predichos como panel, y cuatro de
las 33 positivas no tienen ninguna predicción. Esto no significa que las otras
positivas estén bien segmentadas: se pierde también superficie dentro de ellas.
Una tesela concentra el 45,3 % de los píxeles falsos negativos:
`PNOA_MA_OF_ETRS89_HU30_h25_0057_1_f29184_c59904`; su Dice pasa de 0,8181 en
2023 a 0,4414 en 2024. Conviene tenerla presente al interpretar las métricas
globales, que agregan píxeles y dan más peso a las instalaciones grandes.

La superficie etiquetada pasa de 898,81 m² en 2023 a 1.312,67 m² en 2024.
Hay cuatro localizaciones negativas en la referencia que se etiquetaron positivas
en 2024 y dos en sentido contrario. Las imágenes de 2024 dan una superficie
predicha de 863,51 m², sin corrección. Los resultados están en
`runs/evaluacion/validacion_20cm/`, con cinco ejemplos representativos y los CSV
de resultados, referencia, exclusiones, comparación e intervalos.

Esta prueba muestra una pérdida de segmentación, principalmente por omisión, al
usar el modelo congelado sobre el producto de 2024 preparado aquí. No avala usarlo
como equivalente al de 2023 para medir superficie. No identifica qué parte de la
pérdida procede de resolución, radiometría, geometría, cambios de instalaciones
o diferencias de etiquetado, ni establece rendimiento en toda Cantabria.

Los 18 cm del vuelo y los 15 cm de la ortofoto PNOA 2023 describen cosas distintas:
el GSD de captura y el paso de la rejilla del producto final. El tamaño de píxel
final no garantiza detalle óptico independiente a esa distancia
([especificaciones del IGN](https://pnoa.ign.es/pnoa-imagen/especificaciones-tecnicas-pnoa-anual-y-otros-productos)).
La campaña de 2024 es **Ortofoto 2024 True Ortho del Gobierno de Cantabria**,
financiada íntegramente por ese gobierno, con vuelo a GSD de 18 cm, producto a
20 cm y radiometría original de 16 bits
([registro oficial de publicación](https://mapas.cantabria.es/?trk=public_post_main-feed-card-text)).
Se extrajo por WCS, cobertura 16 `Ortofoto_de_2024_True_Ortho`, con RGB convertido
a 8 bits mediante divisor fijo 257. Se trata de una campaña y un producto
distintos del PNOA 2023, no de sus imágenes remuestreadas a 20 cm.

## Barrido exploratorio de umbrales

Se probaron 19 umbrales, de 0,00001 a 0,70, sobre las mismas 240 imágenes y sus
localizaciones de 2023. La inferencia se realiza una sola vez por imagen y los
umbrales se aplican al mapa sigmoid en memoria. No se guardan probabilidades,
no se reentrena, no se cambia el umbral compartido del modelo y no se aplica
calibración de superficie.

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.barrer_umbrales 20cm
# También permite una lista concreta; siempre incluye la referencia 0,5:
.\.venv\Scripts\python.exe -m paneles_solares.modelos.barrer_umbrales 20cm --umbrales 0.03 0.1 0.3 0.5
```

La salida fija es `runs/evaluacion/validacion_20cm/umbrales/`: `resumen.csv`,
`resultados.csv`, `presencia_entre_campanias.csv`, `config.json`, `barrido.png` y
hasta tres comparaciones visuales de 0,5 frente al candidato con mayor Dice de
los probados. Repetir el barrido sustituye únicamente su propia carpeta. Si se
repite la evaluación principal, se sustituye la validación completa, incluido
este diagnóstico, que puede regenerarse con el comando anterior.

| Umbral | Dice | Precision | Recall | Sesgo superficie | Negativas con algún FP | Negativas con más de 10 píxeles FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5 | 0,7254 | 0,9141 | 0,6013 | −34,22 % | 0 | 0 |
| 0,1 | 0,7444 | 0,9022 | 0,6336 | −29,77 % | 0 | 0 |
| 0,03 | 0,7549 | 0,8943 | 0,6531 | −26,97 % | 0 | 0 |
| 0,003 | 0,7755 | 0,8799 | 0,6933 | −21,21 % | 4 | 0 |
| 0,0001 | 0,7953 | 0,8116 | 0,7797 | −3,93 % | 207 | 37 |

El denominador de negativas es 207. «Algún FP» significa al menos un píxel, no
una instalación falsa completa. En 0,003 esas cuatro manchas suman 12 píxeles,
0,27 m², pero siguen sin detectarse las cuatro positivas omitidas en 0,5. En
0,0001 las negativas acumulan 1.666 píxeles falsos, 37,49 m²; 37 imágenes superan
el límite de revisión de 10 píxeles de `evaluar_unet`. No se han filtrado esas
manchas ni se ha cambiado el procesamiento para ocultarlas.

0,0001 maximiza Dice entre los valores probados. Su sesgo casi nulo no implica
segmentación correcta: conserva 12.852 píxeles falsos negativos y añade 10.562
falsos positivos que se compensan parcialmente en el área total. No se adopta
automáticamente como nuevo umbral. En el grupo aleatorio, 0,003 alcanza Dice
0,8565 y recall 0,8221; en los positivos añadidos, Dice 0,7390 y recall 0,6408.
Esta diferencia de composición también desaconseja extrapolar el total regional.

El diagnóstico temporal trabaja a escala de tesela con presencia definida por
cualquier píxel predicho, como los indicadores actuales del repositorio. A
0,03 aparecen cuatro cambios de ausencia a presencia entre predicciones, pero
uno corresponde a una tesela que ya era positiva en las etiquetas de 2023. Dos
desapariciones predichas corresponden a teselas todavía positivas en 2024. Son
ejemplos de cambios aparentes provocados por omisiones; no fechas de instalaciones.
A 0,0001 se predice presencia en todas las teselas de ambas campañas y desaparece
por completo la señal de cambio bajo esa definición.

El barrido mejora algo la segmentación, pero no acredita seguimiento automático
de altas por año. La elección se ha hecho usando estas mismas etiquetas, que
además pueden tener incertidumbre de interpretación y contornos; los valores
son exploratorios y requieren validación independiente para adoptar un umbral.
No permiten decidir que el problema sea imposible o que no existan datos de
otros años. El servicio oficial contiene, entre otras, campañas de 2017, 2020,
2023, 2024 y 2025
([series de ortofotos](https://geoservicios.cantabria.es/inspire/rest/services/Ortofoto_Series_WCS/MapServer)).
Incluso con detección fiable, las imágenes permiten acotar aparición entre las
fechas de los vuelos, no deducir una fecha exacta de instalación.

Para discutir el alcance con el IFCA, se puede informar de que el modelo de 2023
necesita validación por producto y campaña, que bajar el umbral ofrece mejora
limitada con riesgo de presencias espurias, y que una primera aparición confirmada
manualmente entre vuelos sería un objetivo más defendible que asignar años
exactos automáticamente. Eso permite decidir qué validación adicional aporta
valor antes de ampliar datos o entrenamientos.

Se verificó que el barrido reproduce exactamente los TP, FP, FN y métricas
anteriores a umbral 0,5 en ambos años, con la misma huella SHA256 de pesos.
Pasan las doce pruebas, incluyendo la recuperación de recall al bajar umbral y
la distinción entre una omisión antigua y una aparición nueva a escala de tesela.
