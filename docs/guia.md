# Guía de validación y seguimiento

Esta guía reúne los flujos complementarios al proceso de 2023 descrito en el
[README](../README.md). Los comandos se ejecutan desde la raíz del repositorio.

- [Validación de otras resoluciones](#validación-de-otras-resoluciones)
- [Etiquetado histórico](#etiquetado-histórico)
- [Superficies y periodos](#atribución-agrupación-y-superficies-históricas)

## Validación de otras resoluciones

Evalúa la U-Net congelada sobre las mismas localizaciones del test de 2023,
usando otra campaña y resolución nativa. Mantiene `weights/unet_resnet34.pt`,
RGB, normalización ImageNet, sigmoid y umbral 0,5. Sus etiquetas permanecen
en la campaña: no entran en entrenamiento ni se incorporan al pool.

### Preparación y etiquetas

`CAMPANIAS`, al principio de `src/paneles_solares/datos/validacion.py`, contiene
una entrada por campaña. `20cm` utiliza la
Ortofoto 2024 True Ortho del Gobierno de Cantabria, cobertura WCS 16
`Ortofoto_de_2024_True_Ortho`, rejilla de 0,20 m y EPSG:25830.
El producto está identificado en sus [metadatos oficiales](https://geoservicios.cantabria.es/inspire/rest/services/Ortofoto_2024/MapServer).

Desde la raíz del repositorio:

```powershell
# Guardar solo coordenadas y metadatos.
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_validacion 20cm --solo-seleccion
# Extraer las ventanas que falten.
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_validacion 20cm
```

Se sortean 219 localizaciones de las 988 del test antes de consultar predicciones
y se añaden hasta 31 positivos conocidos. Con semilla 42 quedan 243 teselas:
219 aleatorias y 24 positivos adicionales; los otros siete positivos ya están
en el grupo aleatorio. El grupo indica el criterio de selección de 2023,
no garantiza presencia en 2024.

Cada recorte conserva 76,8 × 76,8 m. A 20 cm se guarda un PNG RGB de 384 × 384;
si las rejillas no coinciden, se interpola para mantener el encuadre físico.
En `data/datasets/validacion_20cm/` se conservan `config.json`, `manifest.csv`,
`images/{positivos,aleatorias}/` y `annotations/{positivos,aleatorias}/`.
El manifiesto registra límites, CRS, centro, tamaño, campaña, fuente y selección.

El WCS devuelve solo la ventana como GeoTIFF en memoria, con límite de 8 MB.
El RGB uint16 se divide por 257 y redondea a uint8, sin realce por imagen.
No se guardan TIF intermedios ni copias a 512. `fuentes` también admite GeoTIFF
locales o COG remotos con HTTP Range y bloques pequeños; las rutas relativas
parten de la raíz. Se comprueban CRS, resolución nativa, RGB y cobertura.
Una imagen WMS solicitada a 20 cm no acredita resolución nativa.
Los bordes entre hojas se unen únicamente en memoria. Nodata o falta de
cobertura detienen la extracción y conservan lo completado.

Repetir reanuda las imágenes pendientes y conserva etiquetas. Año y fuente
pueden completarse antes de extraer; después, cambiar el origen o la selección
exige otra campaña. Bootstrap y número de ejemplos sí pueden variar.

Para etiquetar cada grupo:

```powershell
.\.venv\Scripts\labelme.exe data/datasets/validacion_20cm/images/positivos --output data/datasets/validacion_20cm/annotations/positivos --labels panel_solar --flags revisada,dudosa,descartar --config '{with_image_data: false}'
.\.venv\Scripts\labelme.exe data/datasets/validacion_20cm/images/aleatorias --output data/datasets/validacion_20cm/annotations/aleatorias --labels panel_solar --flags revisada,dudosa,descartar --config '{with_image_data: false}'
```

Dibuja polígonos `panel_solar`, marca `revisada` y guarda cada JSON, también
para negativos con `shapes: []`. `dudosa` o `descartar` excluyen la imagen
y registran el motivo. `with_image_data: false` evita duplicar el PNG en base64.
Las etiquetas de 2023 son referencia visual, no verdad de la nueva campaña.

### Evaluación y salidas

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.evaluar_validacion 20cm
```

Todas las teselas necesitan imagen y JSON revisado o exclusión explícita.
Una etiqueta ausente no se interpreta como negativo. RGB se redimensiona
a 512 con bilineal y la máscara con vecino más próximo. Cada píxel evaluado
representa 0,15 × 0,15 m² por el encuadre físico, no 0,20 × 0,20 m².

`runs/evaluacion/validacion_20cm/` contiene:

| Salida | Contenido |
| --- | --- |
| `resultados.csv` | Métricas por tesela y metadatos. |
| `resumen.csv`, `referencia_15cm.csv` | Métricas por grupo y diferencias absolutas frente a las mismas localizaciones de 2023, reevaluadas con los mismos pesos. |
| `intervalos.csv` | Bootstrap percentil 95 % por tesela y grupo, manteniendo los pares entre campañas. |
| `excluidas.csv`, `config.json` | Exclusiones, configuración efectiva y SHA256 de pesos. |
| `comparacion_test_historico.csv` | Comparación contextual con el test completo guardado, si existe. Sus pesos históricos no constan en el CSV. |
| `ejemplos/` | Peores errores: verde para aciertos, rojo para FP y azul para FN. |

El bootstrap usa 1.000 réplicas por defecto; `bootstrap: 0` lo desactiva.
Se omiten las réplicas con métricas indefinidas y se informa cuántas quedaron.
La salida anterior se sustituye únicamente al terminar correctamente.

`tp`, `fp` y `fn` cuentan píxeles. Los errores por imagen cuentan negativos
con algún píxel predicho y positivos sin ninguno. Las categorías de revisión
mantienen los límites de más de 10 píxeles y más del 10 %; pueden señalar
errores parciales. No se cuentan instalaciones. Precision, recall y sesgo
quedan indefinidos si falta el denominador.

Para otra resolución, añade una entrada completa en `CAMPANIAS` y pasa
su nombre a los mismos comandos. A 25 cm, 76,8 / 0,25 = 307,2: el PNG se prepara
a 307 × 307 conservando los 76,8 m antes de entrar al modelo a 512.

### Resultado de la muestra etiquetada

Se evaluaron 240 de las 243 teselas: 33 positivas y 207 negativas. Las tres
exclusiones proceden de banderas manuales. Ambas campañas usan los mismos
pesos y umbral 0,5.

| Métrica | Referencia 2023, 15 cm | Campaña 2024, 20 cm |
| --- | ---: | ---: |
| Dice | 0,8657 | 0,7254 |
| IoU | 0,7631 | 0,5692 |
| Precision | 0,9121 | 0,9141 |
| Recall | 0,8237 | 0,6013 |
| Sesgo de superficie | −9,69 % | −34,22 % |

El IC95 % del Dice de 2024 es [0,6003; 0,8345]; el de su diferencia respecto
a 2023, [−0,2511; −0,0391]. Solo describen esta muestra estratificada.

En 2024 hubo 3.296 píxeles FP y 23.259 FN. Ningún negativo tuvo predicción
y cuatro positivos quedaron sin detectar. La tesela
`PNOA_MA_OF_ETRS89_HU30_h25_0057_1_f29184_c59904` concentra el 45,3 % de los FN:
su Dice pasa de 0,8181 a 0,4414. Las métricas globales dan más peso a
instalaciones grandes. La superficie anotada pasa de 898,81 a 1.312,67 m²;
la predicha de 2024 es 863,51 m² sin corrección. Cuatro localizaciones
negativas en 2023 se etiquetaron positivas en 2024 y dos en sentido contrario.

### Barrido exploratorio de umbrales

Se infiere una vez por imagen y se aplican los umbrales al mapa sigmoid
en memoria. No se cambia el umbral compartido ni se calibra la superficie.

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.barrer_umbrales 20cm
.\.venv\Scripts\python.exe -m paneles_solares.modelos.barrer_umbrales 20cm --umbrales 0.03 0.1 0.3 0.5
```

Siempre se incluye 0,5. La carpeta `umbrales/` dentro de la evaluación contiene
resultados y resumen CSV, diagnóstico de presencia entre campañas, configuración,
gráfica y hasta tres comparaciones visuales. Repetir la evaluación principal
sustituye también este diagnóstico.

Se probaron 19 umbrales entre 0,00001 y 0,70:

| Umbral | Dice | Precision | Recall | Sesgo de superficie | Negativos con algún FP | Con más de 10 píxeles FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5 | 0,7254 | 0,9141 | 0,6013 | −34,22 % | 0 | 0 |
| 0,1 | 0,7444 | 0,9022 | 0,6336 | −29,77 % | 0 | 0 |
| 0,03 | 0,7549 | 0,8943 | 0,6531 | −26,97 % | 0 | 0 |
| 0,003 | 0,7755 | 0,8799 | 0,6933 | −21,21 % | 4 | 0 |
| 0,0001 | 0,7953 | 0,8116 | 0,7797 | −3,93 % | 207 | 37 |

El denominador es de 207 negativos. En 0,003 las cuatro manchas suman
12 píxeles, 0,27 m²; siguen sin detectarse los cuatro positivos omitidos
en 0,5. En 0,0001 los negativos acumulan 1.666 píxeles FP, 37,49 m².
Ese umbral maximiza Dice entre los probados, pero conserva 12.852 FN
y añade 10.562 FP. Un sesgo pequeño puede compensar errores.
No se adopta como nuevo umbral y requiere validación independiente.

### Alcance

La prueba mide transferencia al producto, año y lugares seleccionados.
El total enriquecido no estima prevalencia ni superficie regional.
El grupo aleatorio representa el marco del test de 2023, formado por teselas
con edificios. Las exclusiones pueden introducir sesgo.

Cambian a la vez resolución, año, radiometría, geometría, instalaciones
y etiquetado. La campaña 2024 añade True Ortho y RGB original de 16 bits;
la prueba no atribuye las diferencias a una sola causa. El GSD de captura
y la rejilla de la ortofoto son magnitudes distintas. El bootstrap no corrige
dependencia espacial ni error de etiquetado. No se aplica la corrección
del −7,54 % del test completo de 2023 a esta campaña.

## Etiquetado histórico

El flujo general parte de ventanas geográficas y campañas de ortofoto. No
necesita las respuestas del antiguo visor ni ejecuta un modelo para decidir
qué dibujar. Se preparan recortes RGB y se etiquetan directamente en LabelMe.
Las mismas ventanas físicas permiten comparar campañas con distinta resolución.

### Preparar las imágenes

Un CSV de entrada define las ventanas en **EPSG:25830**:

```csv
ventana,xmin,ymin,xmax,ymax
zona_001,395238.8,4801655.6,395315.6,4801732.4
```

Cada ventana es un recorte, no una instalación. El identificador debe ser único
para ese encuadre. Se pueden seleccionar recortes a partir de detecciones
revisadas, edificios o localizaciones conocidas, incluidas instalaciones en
suelo; el preparador no aplica filtros de Catastro. La selección de entrada
se conserva para poder reproducir la muestra.

Prueba pequeña con el CSV de ejemplo incluido en el repositorio:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --ventanas docs/ejemplo_ventanas_historicas.csv --anios 2023 2020 2017 2014 --salida runs/prueba_historico
```

Para preparar una selección propia:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --ventanas ventanas.csv --anios 2023 2020 2017 2014
```

Por defecto se escribe `data/datasets/seguimiento/labelme/`. `--salida` permite
preparar otra muestra y `--limite N` limita las filas del CSV para una prueba.
Una columna opcional `anio` permite solicitar solo una campaña por fila.
Si ya existe un PNG local, se puede indicar su ruta en `origen`, junto con su
año; si falta, se recupera el encuadre desde el WCS oficial.

Las campañas se definen en `datos/seguimiento.py`: 2020 y 2023 a 15 cm, 2014 y
2017 a 25 cm. `--config archivo.json` admite otra configuración con la misma
lista `campanias` y los campos `anio`, `resolucion_m`, `cobertura`, `etiqueta`
y `divisor_rgb`; `url` es opcional. Se comprueba una vez la cobertura y su
resolución antes de descargar. No se amplían fotos antiguas para aparentar
más detalle: 76,8 m corresponden aproximadamente a 307 píxeles a 25 cm.
En un lote nuevo se incluye 2023 si se necesita medir también su superficie.
El análisis y el panel obtienen las campañas del manifiesto; no es necesario
cambiar sus listas de años en el código.

Las coberturas usan los nombres publicados por el
[WCS oficial](https://geoservicios.cantabria.es/inspire/services/Ortofoto_Series_WCS/MapServer/WCSServer?SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCapabilities),
por ejemplo `Ortofoto_de_2020`. Las configuraciones antiguas con números
se consultan por su etiqueta, conservando los metadatos de las muestras existentes.

Cada imagen se llama `ventana__anio.png`. `manifest.csv` conserva su campaña,
fuente, encuadre, escala real por píxel y rutas relativas del PNG y el JSON.
También sirve como entrada para volver a preparar las mismas imágenes.

Repetir el comando reutiliza los PNG existentes y descarga solo los que faltan.
El manifiesto conserva las filas de preparaciones anteriores, aunque se ejecute
un subconjunto. Si se interrumpe una descarga, se conservan los recortes ya
terminados. No se cambian los encuadres de imágenes existentes ni se escriben
JSON vacíos o predicciones que puedan confundirse con anotaciones manuales.

### Etiquetar en LabelMe

Para la carpeta habitual, ya preparada:

```powershell
.\.venv\Scripts\labelme.exe data/datasets/seguimiento/labelme/images --labels panel_solar --flags ausente,dudosa --config '{with_image_data: false, auto_save: true, keep_prev: false, display_label_popup: false}'
```

Para otra muestra cambia solo la carpeta, por ejemplo a
`runs/prueba_historico/images`.

1. Dibuja todos los paneles visibles con `panel_solar`, incluidos los que
   omitió el modelo. Usa polígonos separados para conjuntos separados; no
   asignes Group ID ni unas casas diferentes. Ctrl+N crea y Ctrl+J edita.
2. Cada cambio se guarda automáticamente. D pasa a la siguiente imagen y A
   vuelve atrás. Ctrl+Z deshace. Los JSON quedan junto a los PNG.
3. En una selección nueva que incluya imágenes vacías, marca `ausente` si no
   hay ningún panel en toda la foto. Esa marca guarda también el JSON vacío.
   Marca `dudosa` si la imagen no permite decidir; no inventes contornos.

Al volver, selecciona en la lista la imagen donde te quedaste. Tener JSON no
significa que esté terminada: el autoguardado conserva trabajo parcial. La
preparación nunca modifica los JSON ni da por revisadas imágenes por tenerlos.

### Recuperar la revisión anterior de este proyecto

La recuperación es una opción específica del mismo preparador:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --recuperar
```

Lee `data/datasets/seguimiento/seguimiento.sqlite` en modo de solo lectura.
Recupera observaciones humanas `presente` anteriores a 2023, relacionándolas
con sus ventanas y recursos, y entrega esas filas al preparador general.
`--base` permite indicar otra copia de la base. Se incluyen positivos
históricos aunque la selección se descartara en 2023. Las respuestas originales
no se modifican y las imágenes y anotaciones ya preparadas se conservan.

Con las respuestas actuales son 754 selecciones y 1.486 imágenes: 402 de 2014,
440 de 2017 y 644 de 2020. No equivalen a 754 instalaciones. El manifiesto
conserva las selecciones positivas de origen para poder recuperar su alcance.

Las ausencias antiguas solo afectan a lo que se seleccionó entonces, nunca a
toda la imagen. Un positivo significaba que seguía al menos uno de esos paneles,
no que siguieran todos. Esas respuestas no son superficies ni se propagan
como etiquetas completas de entrenamiento. El modelo de 2023 tampoco tenía
la misma fiabilidad en campañas anteriores, por lo que no se precargan máscaras.

Los casos que solo aparecen en 2023 no necesitan dibujos de años anteriores
cuando sus ausencias ya tienen un alcance suficiente. Su geometría de 2023
se recupera como una estimación del modelo; todavía puede requerir revisión.
Se relacionan polígonos, edificios, parcelas y candidatos para obtener
superficies planimétricas e intervalos entre vuelos.
No se deduce un año exacto de instalación ni la superficie inclinada del panel.

### Cerrar la revisión e incorporar negativos

Cuando el revisor termina el lote se registra su cierre:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --finalizar data/datasets/seguimiento/labelme/manifest.csv
```

El manifiesto conserva el estado, la confirmación de revisión y el SHA256 de
cada JSON. Si se modifica después, deja de considerarse confirmado hasta
finalizarlo de nuevo. Las imágenes dudosas siguen sin confirmar. Las imágenes
sin JSON siguen pendientes por defecto.

En el lote recuperado de este proyecto, el revisor confirmó que las imágenes
sin dibujos eran falsos positivos. Ese cierre se ejecutó con la excepción
`--sin-json-negativos`, que crea sus JSON vacíos con `ausente`. No debe usarse
en lotes que sigan pendientes de revisión. Quedaron 1.417 imágenes positivas
y 69 negativas, todas con su anotación y procedencia.

Los negativos confirmados se incorporan al pool habitual y se sincronizan:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.incorporar_labeling --historico data/datasets/seguimiento/labelme/manifest.csv
.\.venv\Scripts\python.exe -m paneles_solares.datos.sincronizar
```

Se copian los recortes al pool conservando el archivo histórico original.
Cada muestra conserva año, ventana y límites. Las
huellas de validación y test de 2023 se comprueban antes de asignar otras
campañas a entrenamiento: 60 negativos entraron en train y 9 quedaron en
el pool con `no_usar` por solape con validación. No se añaden casos seleccionados
por errores a la validación ni se cambia el test. Los negativos producen TXT
vacíos para YOLO y máscaras vacías para U-Net. Esto no reentrena los modelos.
U-Net adapta RGB y máscara a 512 píxeles al cargar cada muestra; las máscaras
usan vecino más próximo. Así pueden convivir campañas de 307 y 512 píxeles
en un lote. Los archivos originales y la escala usada para medir no cambian;
ampliar una imagen antigua no recupera el detalle que no tiene.

### Atribución, agrupación y superficies históricas

Después de etiquetar el ejemplo pequeño, se puede completar el flujo sin
ninguna base de datos anterior:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.preparar_seguimiento --finalizar runs/prueba_historico/manifest.csv
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos --manifest runs/prueba_historico/manifest.csv --salida runs/prueba_historico/resultados
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.panel_indicadores --carpeta runs/prueba_historico/resultados --salida runs/prueba_historico/panel
```

Se georreferencian los polígonos con el encuadre y tamaño reales de cada imagen.
La misma exportación acepta la referencia de 2023 producida por U-Net:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos --manifest data/datasets/seguimiento/labelme/manifest.csv --detecciones runs/indicadores/unet_actual/detecciones.gpkg
```

`--detecciones` lee la capa `detecciones` de la inferencia de 2023 y utiliza
sus identificadores, campaña, geometrías y marcas de borde. Solo incluye
polígonos que intersectan los encuadres del manifiesto; no los recorta a esos
encuadres ni a Catastro. La procedencia queda como `modelo_2023`, sin dar
por confirmada su presencia. Las anotaciones revisadas tienen prioridad en
el estado y el área de cada agrupación. No se cargan predicciones de los
años antiguos. `--base` y `--detecciones` son alternativas: la primera
recupera el trabajo del visor retirado; la segunda permite recrear el flujo
con una inferencia nueva. Ninguna es necesaria si todas las campañas se
han delimitado en LabelMe.

`--limite 20` permite reducir la prueba a veinte ventanas. Para repetir casos
concretos se puede añadir `--ventana identificador` varias veces. Un JSON
guardado puede estar incompleto hasta cerrar la revisión.
Las imágenes sin anotación, dudosas o con geometría inválida
no se convierten en ausencias ni en superficie cero. Los cruces de contorno
se reparan en las geometrías derivadas y se cuentan en la capa `observaciones`,
sin cambiar los dibujos originales. Las geometrías sin superficie válida
quedan pendientes. Se señala también si un polígono toca el borde de la imagen.

Para regenerar el lote recuperado de este proyecto:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos --base data/datasets/seguimiento/seguimiento.sqlite
```

Una atribución a edificio o parcela necesita al menos el 80 % de solape con
un candidato y ningún segundo candidato que alcance el 20 %. Se conservan
todas las intersecciones para revisar los casos ambiguos. No se asigna un
edificio por proximidad. Un panel exterior a CBT23 puede estar sobre una
cubierta que falta en la cartografía: no se clasifica automáticamente como suelo.
Las referencias y los usos cartográficos son de 2023, también en otras campañas.

La salida completa está en `runs/indicadores/seguimiento_historico/`:

- `instalaciones.csv`: una fila por agrupación y campaña, con localización,
  estado, áreas, soportes, periodo de aparición y cambios comparables.
- `atribucion.gpkg`: geometrías y tablas de detalle, descritas abajo.
- `resumen.json`: alcance y recuentos, sin presentarlos como censo.

El CSV es UTF-8, separado por comas y con punto decimal. `id_instalacion`
y `anio` identifican una fila; no se crea un CSV distinto para cada campaña.
Las columnas principales son:

| Columnas | Significado |
| --- | --- |
| `id_instalacion`, `anio` | Agrupación provisional y campaña observada. |
| `municipio`, `codigo`, `latitud`, `longitud` | Municipio del punto interior de la huella conjunta; coordenadas WGS84 en grados. No delimitan los paneles. |
| `edificios`, `parcelas`, `referencias_catastrales` | Soportes atribuidos claramente; varios identificadores se separan con `;`. Son referencias de 2023. |
| `estado`, `fuente_estado` | Presente, ausente, desconocido o pendiente, y evidencia utilizada. |
| `superficie_horizontal_m2`, `fuente_superficie` | Área revisada si existe; en su defecto, estimación del modelo disponible. Ausencia acreditada: cero; desconocido o dibujo parcial sin área confirmada: vacío. |
| `superficie_horizontal_anotada_m2`, `superficie_horizontal_modelo_m2` | Valores originales separados para poder seleccionar solo medidas revisadas o estimaciones. |
| `instalacion_desde`, `instalacion_hasta`, `estado_periodo` | Última ausencia y primera presencia; límite inferior exclusivo y superior inclusivo. No fechan las ampliaciones. |
| `anio_anterior`, `cambio_comparable`, `cambio_horizontal_m2` | Diferencia respecto a la campaña anterior de la serie; queda vacía si alguna de las áreas no es comparable. |
| `instalacion_confirmada`, `revision`, `conflicto` | Estado de la agrupación y posibles contradicciones. |
| `superficie_comparable`, `anotaciones_completas`, `recorte_en_borde`, `cobertura_completa_confirmada` | Límites de la medición. Dibujos terminados no garantizan que se haya cubierto toda la instalación. |
| `n_conjuntos`, `solape_entre_anotaciones_m2` | Número de polígonos de esa campaña y área solapada antes de disolver. Cero polígonos no significa ausencia. |
| `usos_cartograficos` | Uso original de la cartografía, sin inventar tipos de vivienda o titularidad. |

El resto conserva la evidencia de cobertura y respuestas anteriores, la fuente
de primera presencia y el resumen de campañas con geometría. Al leer el CSV,
tratar códigos, identificadores y referencias catastrales como texto.

La carpeta completa se sustituye al terminar la exportación. Una interrupción
no publica una mezcla de archivos nuevos y anteriores. Las capas adicionales
`referencia_2023` y `selecciones_anteriores` conservan la procedencia recuperada.
La tabla `revision_anterior` conserva todas las respuestas originales al usar
`--base`; no se eliminan ni se convierten en etiquetas nuevas de instalación.
Al sustituir los resultados habituales se retira el panel anterior; el comando
del panel vuelve a generarlo. Las pruebas necesitan una salida distinta de la
carpeta del lote completo.

La agrupación propone candidatos con estas reglas:

- Los paneles atribuidos claramente a una misma huella de edificio comparten candidato.
- Entre campañas, un solape de al menos el 50 % del polígono menor propone un vínculo.
- En el mismo año, se exige el 80 % del polígono mayor para vincular posibles duplicados.
- Si una cadena de vínculos conecta edificios distintos, se conserva la separación
  y se señala el conflicto. Compartir parcela o estar cerca no une conjuntos.

Las reglas son hipótesis para revisar, no una identificación automática de
instalaciones reales. CBT23 puede representar solo una parte de un edificio
o incluir varias viviendas en una misma huella. También pueden quedar separados
conjuntos de una instalación en suelo que no se solapan entre años. La capa de
candidatos muestra la unión de las huellas observadas, no toda la finca.

Los polígonos idénticos de un mismo año se deduplican conservando sus ventanas
de origen. La agregación por candidato y año usa la unión geométrica para no
sumar dos veces un solape; su área también queda indicada para revisar
anotaciones discordantes. No se rellenan con cero los años sin polígonos.
La superficie es horizontal. `anotaciones_completas` distingue los dibujos
cerrados; `recorte_en_borde` advierte de paneles posiblemente cortados por
el encuadre. Terminar el dibujo no confirma la extensión de toda la instalación.
Los tipos son los usos originales de CBT23: no se inventan categorías de
chalet, urbanización ni titularidad pública.

Para corregir una agrupación se puede proporcionar `--vinculos ruta.csv`, con
columnas `id_panel,id_instalacion,instalacion_confirmada`. Las filas indican
qué polígonos separar o reunir, incluso entre varias cubiertas. Se confirma
un grupo solo si todos sus miembros tienen confirmación explícita (`True`).
No se pide Group ID al dibujar ni se modifican los JSON. Los identificadores
automáticos son provisionales; las correcciones usan el identificador elegido
para la instalación y deben revisarse si se redibuja un polígono.

Confirmar la agrupación no confirma que estén dibujados todos sus paneles:
`cobertura_completa_confirmada` permanece en falso. La primera campaña con
paneles tampoco es una fecha exacta de instalación.

### Intervalos y reutilización de las ausencias

El flujo general funciona con las campañas de LabelMe. La opción `--base`
añade la recuperación de este proyecto: máscaras de 2023 y respuestas humanas
del visor, sin usar sus predicciones de 2014, 2017 o 2020. Las referencias de
2023 marcadas como falsas no entran en la agrupación activa. Las demás siguen
siendo geometrías estimadas: `fuente_geometria` y `fuente_superficie` impiden
confundirlas con los dibujos manuales. No se añaden sus áreas a la columna
de superficie anotada de la serie temporal.

Una respuesta positiva antigua solo confirma presencia en un candidato si
todos los componentes de aquella selección han quedado relacionados con él.
Si se reparten entre varias casas, cada detección permanece sin confirmar.
Un intervalo basado únicamente en esas detecciones lleva
`estado_periodo=presencia_por_confirmar`: depende de comprobar esa presencia.

Una ausencia antigua completa se puede aplicar a todos los conjuntos de la
selección. Se exige que la unión de selecciones ausentes cubra toda la huella
conocida del candidato, sin extenderla a la imagen, edificio o parcela.
Las respuestas con alcance `zona_visible` no certifican ausencia completa.
Se conservan la fracción cubierta y los identificadores de las selecciones.

Para LabelMe se aceptan las imágenes vacías confirmadas. En una foto positiva,
la falta de solape con la posición de otra campaña no basta: puede ser un
desplazamiento o un vínculo sin resolver. Solo se deduce ausencia si está
revisado todo el soporte de referencia y no hay otros paneles sin relacionar
en él. Una campaña sin evidencia queda como `desconocido`, con área vacía.

`instalacion_desde` es la última ausencia anterior a la primera presencia y
es un límite exclusivo; `instalacion_hasta` es la primera presencia y es
inclusivo. Por ejemplo, `2017 < instalación ≤ 2020`. Sin ausencia anterior,
el límite inferior queda vacío. Una revisión completa de LabelMe sustituye
una respuesta antigua incompatible, dejando `respuesta_anterior_superada`
para auditoría. No vuelve a pedir revisar las imágenes vacías que ya se
confirmaron. Las contradicciones sin resolver y ausencias posteriores a una
presencia se señalan, sin decidir si son retirada, desplazamiento o error.

Este intervalo fecha el inicio del candidato, no toda su superficie posterior.
`cambio_horizontal_m2` compara únicamente dibujos cerrados o ceros acreditados,
con la misma huella revisada y sin paneles cortados por el borde. Las áreas
del modelo de 2023 y los casos dudosos no producen diferencias medidas.
Un aumento de área todavía no identifica la época de cada módulo. El escenario
energético central usa la época del emplazamiento como aproximación explícita;
no convierte ese aumento en una fecha de instalación de módulos.

### Datos para máscaras y GeoTIFF

Las geometrías están en `runs/indicadores/seguimiento_historico/atribucion.gpkg`,
en EPSG:25830, con coordenadas y superficies en metros. Se puede abrir
directamente en QGIS o leer una capa con GeoPandas. Contiene:

| Capa o tabla | Contenido |
| --- | --- |
| `paneles` | Un polígono o multipolígono por conjunto dibujado o detectado, no necesariamente por módulo físico. Incluye `id_panel`, `id_instalacion`, `anio`, `fuente_geometria`, `revision_completa`, `borde_imagen`, `ventanas`, atribución y superficie horizontal. |
| `instalaciones` | Una fila por agrupación y campaña, con los mismos campos del CSV y la unión de los paneles de ese año. La geometría corresponde a `fuente_superficie`. Ausencias y observaciones sin superficie disponible tienen geometría nula. Es la capa principal para consultar áreas y exportar máscaras por año. |
| `emplazamientos` | Unión de las huellas de cada agrupación a lo largo de las campañas. Sirve para localizarla en el visor, no como máscara de un año concreto. |
| `observaciones` | Un rectángulo por imagen y campaña: `ventana`, `anio`, `estado`, `revision_completa`, límites y ruta relativa de la anotación. Incluye también imágenes pendientes o no evaluables. |
| `edificios_2023`, `parcelas_2023` | Contexto cartográfico utilizado en la atribución. |
| `paneles_edificios`, `paneles_parcelas` | Tablas sin geometría con todas las intersecciones, sus áreas y fracciones. |
| `enlaces_espaciales` | Tabla de vínculos propuestos entre máscaras y decisión de agrupación. |
| `referencia_2023`, `selecciones_anteriores`, `revision_anterior` | Máscaras, selecciones y respuestas originales recuperadas con `--base`. La referencia incluye también detecciones descartadas: para la máscara activa se usa `paneles`. |

Para consultar una campaña basta cargar `instalaciones` y filtrar, por ejemplo,
`"anio" = 2017`. `superficie_horizontal_m2` contiene los m² de los paneles de
esa agrupación en ese año, sin duplicar sus solapes. Todos los años utilizan
el mismo `id_instalacion`. El estado y la fuente distinguen las ausencias,
los dibujos manuales y las estimaciones del modelo.

Para las máscaras históricas manuales a partir del detalle, filtrar `paneles` por `anio`,
`fuente_geometria=labelme` y `revision_completa=True`. En `observaciones`,
usar como cobertura conocida únicamente las filas del mismo año con
`revision_completa=True` y estado `presencia_anotada` o `ausencia_imagen`.
Las geometrías ya están transformadas desde los píxeles originales.

Una codificación sencilla para un GeoTIFF de tipo `uint8` es:

- `1`: panel en la campaña seleccionada.
- `0`: parte revisada de la imagen sin paneles.
- `255`: NoData, zona no revisada o no evaluable.

Los años antiguos solo cubren los recortes estudiados; no hay una máscara
manual completa de Cantabria. Las ausencias del visor anterior se refieren
a las geometrías seleccionadas, no a toda la foto: si se incorporan, cruzar
`revision_anterior` con `selecciones_anteriores` por `id_seleccion` y respetar
su `alcance`. No extender ese cero al edificio o parcela completos.

En 2023 la capa activa actual procede del modelo (`fuente_geometria=modelo_2023`),
no de un dibujo manual. Se puede rasterizar como predicción, conservando esa
distinción. Este GeoPackage no contiene toda la inferencia regional: representa
el inventario seleccionado para seguimiento. Su exterior no acredita ausencia.

Las imágenes y dibujos originales están en
`data/datasets/seguimiento/labelme/images/`, en PNG y JSON. Su
`data/datasets/seguimiento/labelme/manifest.csv` relaciona imagen, anotación,
campaña, límites, dimensiones, CRS y resolución. Las rutas de imagen y
anotación se resuelven desde la carpeta del manifiesto. En el lote actual,
2014 y 2017 tienen unos 0,25 m/píxel; 2020 tiene 0,15 m/píxel.
Los TIFF regionales originales de 2023 están en `data/pnoa/`, a 0,15 m/píxel.

Guardar los nuevos rásteres fuera de los indicadores regenerables:

```text
runs/mascaras_historicas/
    paneles_2014.tif
    paneles_2017.tif
    paneles_2020.tif
    paneles_2023.tif
```

Para comparar píxeles entre campañas se necesita una rejilla común: mismo
CRS, origen, extensión y resolución. Remuestrear una imagen antigua no añade
detalle. Las superficies del CSV se calculan desde los polígonos y no cambian
al elegir la resolución del GeoTIFF. Este flujo no genera aún esos rásteres.

### Consultar las instalaciones en el panel

`visualizacion.panel_indicadores` genera un único HTML que se puede enviar
sin archivos auxiliares. Incluye datos, estilos, código del mapa e iconos.
El lector puede abrirlo sin conexión y sin disponer del repositorio. Los
botones de resumen municipal, energía y selección generan los CSV desde
los datos incluidos; no enlazan a archivos locales.

El mapa utiliza por defecto un fondo vacío, con municipios, paneles y
contexto vectorial incluidos. Los fondos de ortofoto y calles requieren
Internet y solo se solicitan al activarlos. Los enlaces bibliográficos son
consultas opcionales, no recursos necesarios para abrir el panel. Leaflet
1.9.4 está en `src/paneles_solares/visualizacion/plantillas/leaflet/`, con
su licencia BSD; el generador incluye los estilos e iconos como datos y
no necesita descargar esa biblioteca.

«Mapa y resumen»
contiene los indicadores principales y las ortofotos. «Instalaciones y
evolución» contiene los gráficos y la tabla por agrupación y campaña.
Ambas pestañas comparten año, municipio, uso, agrupación y selección de
inventario o muestra comparable.

El histograma de tamaños, la mediana y el percentil 90 usan las superficies
positivas disponibles. Una presencia antigua sin área delimitada no entra
en esa distribución. Los ceros y estados desconocidos siguen en la tabla.
Las áreas son horizontales y se disuelven por agrupación y año. Las
distribuciones cuentan agrupaciones completas, adscritas al municipio de
su punto interior; el mapa y los totales de superficie recortan al límite
municipal y disuelven también solapes entre agrupaciones.

El gráfico de atribución usa los paneles de cada campaña, no la lista de
edificios reunida a lo largo de todos los años. Una coincidencia clara exige
que todos los paneles de la agrupación tengan edificio asignado. Los casos
parciales o dudosos se muestran aparte. «Sin coincidencia» significa que
ningún polígono coincide con las huellas disponibles: puede ser terreno o
una cubierta ausente de la cartografía. Las categorías sin casos se omiten.

El reparto de usos cuenta agrupaciones con superficie, no m². Utiliza el
uso cartográfico de sus edificaciones CBT23/LULC23. Los usos minoritarios
se reúnen como «Otros usos» solo en el dibujo. La descarga conserva el texto
original; la tabla lo ofrece al pasar el cursor sobre su etiqueta abreviada.
Si una agrupación combina varios usos, cuenta
una vez como «Varios usos». Los desconocidos se muestran como «Sin dato».
No se interpreta el campo `TIPO` del parcelario como industrial, residencial
o tipo de vivienda, ni se deduce titularidad pública.

La tabla permite buscar por identificador o municipio, paginar, abrir una
agrupación en el mapa y descargar todas las filas de la selección, no solo
la página visible. Ese CSV conserva estado, fuente de superficie, intervalo
y confirmación de la agrupación.

### Energía y antigüedad

Se han preparado los cálculos por cohorte en `estimar_energia.estimar_cohorte`.
Cada cohorte representa módulos de una misma época; una ampliación tiene
su propia edad y no rejuvenece los paneles anteriores. Se distinguen:

- Potencia inicial por m² de módulo, según tecnología y época.
- Degradación acumulada por antigüedad.
- Pérdidas operativas de inversor, cableado y otros efectos.

La fórmula es `A_módulos × potencia_inicial_por_m² × rendimiento_PVGIS ×
(1 - pérdidas_operativas) × (1 - degradación_anual)^edad`. El rendimiento debe
consultarse con `loss=0`, pues las pérdidas operativas se aplican fuera una sola
vez. Los efectos de radiación, temperatura y orientación los calcula PVGIS.
La función recibe superficie inclinada de módulos; no debe confundirse con la
superficie horizontal de las máscaras. La inclinación seguirá siendo una hipótesis
cuando no exista medición del tejado o de los módulos.

El [JRC](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/data-sources-calculation-methods_en)
explica que el 14 % recomendado para PVGIS ya incluye envejecimiento medio.
Por eso no se añade otra degradación al resultado del método uniforme anterior.
La función por cohortes parte provisionalmente de 10 % de pérdidas operativas
y 0,5 % anual de degradación, configurables. El 10 % es una hipótesis de trabajo,
no una pérdida medida. El 0,5 % procede de la mediana de la
[revisión de Jordan y Kurtz, NREL](https://nrel.gov/docs/fy12osti/51664.pdf).

La potencia inicial por m² es un parámetro separado: el
[informe fotovoltaico de Fraunhofer ISE](https://www.ise.fraunhofer.de/en/publications/studies/photovoltaics-report.html)
documenta mejoras tecnológicas, pero no identifica la placa de cada tejado de
Cantabria. Todavía no se fija una tabla de eficiencias locales por año ni se
atribuyen cambios de tecnología a partir del color de la ortofoto.

Para una instalación entre 2017 y 2020, la edad en 2023 queda entre 3 y 6 años.
Se calculan extremos y un escenario central situado a mitad del intervalo.
Ese punto medio no es una fecha observada. Si ya existía en el primer vuelo,
solo se conoce una edad mínima: el cálculo deja sin determinar la edad máxima,
la producción mínima y el valor central. El rango calculado cubre la duda en
antigüedad, no toda la incertidumbre de orientación, tecnología o sombras.

El panel único conecta esta función con las superficies del inventario. Se usa
una posición municipal ponderada por área y tres consultas PVGIS con `loss=0`,
clima medio SARAH3, silicio cristalino, montaje ventilado y horizonte del terreno.
Los intervalos siguen siendo provisionales. La salida es producción anual
representativa de los equipos de cada imagen, no consumo, producción medida
durante el año del vuelo ni una extrapolación a toda Cantabria.

| Supuesto | Conservador | Central | Favorable |
| --- | --- | --- | --- |
| Inclinación | 15° | 30° | 40° |
| Orientación | Oeste | Sur | Sur |
| Pérdidas operativas | 18 % | 10 % | 5 % |
| Degradación anual | 1 % | 0,5 % | 0,2 % |
| Eficiencia inicial respecto a su época | × 0,85 | × 1 | × 1,15 |
| Fecha de los módulos | Inicio supuesto del intervalo | Punto medio del intervalo | Año de la imagen |

La eficiencia se interpola a partir de la tabla 2 del
[informe JRC 2025](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC144141/JRC144141_01.pdf):
16,3 % en 2014, 17,7 % en 2017, 20 % en 2020 y 21,8 % en 2023, conservando
también sus valores anuales intermedios. Son referencias de mercado, no medidas
locales. Antes de 2014 se suponen 10 % en 1989, 12 % en 2000 y 15 % en 2010.

Sin límite inferior compatible se supone un intervalo desde 25 años antes de
la primera geometría hasta esa primera observación. Se conserva como edad
modelada, sin modificar el intervalo observado original. El escenario central
puede infravalorar ampliaciones o renovaciones recientes; el favorable permite
que toda el área sea de equipos nuevos. Esto evita rejuvenecer automáticamente
los módulos por diferencias entre el modelo de 2023 y los dibujos anteriores.

El área de módulos es `área_horizontal / cos(inclinación)`. Por emplazamiento y
municipio se disuelven máscaras del mismo año. Para agregar emplazamientos con
solapes, la mezcla de producciones se escala al área de su unión; el reparto
de esos solapes se pondera por superficie. El total es la suma de municipios
y del área sin adscripción. No se duplica producción por repetir un contorno.

La amplitud de los escenarios describe sensibilidad a estas hipótesis, no un
intervalo de confianza. No cuantifica errores de detección, pendientes reales,
sombras locales, fallos ni omisiones. Las máscaras de 2023 siguen siendo
estimadas. Confirmar instalaciones y fechar cada ampliación permitiría sustituir
las hipótesis, pero no es una condición para mostrar este cálculo exploratorio.

El comando del panel regenera `produccion_estimada.csv`, `factores_energia.csv`
y `hipotesis_energia.json` junto al HTML. La caché `pvgis_anual.json` conserva
respuestas y parámetros, sin mezclar el 14 % del escenario regional uniforme.

### Entrenamiento posterior

Los negativos históricos confirmados ya están incorporados según el flujo
anterior. Los positivos permanecen como referencia manual para comparar
predicciones y seleccionar casos difíciles, respetando las zonas de
validación y test también entre años. Hay que revisar además imágenes sin
detecciones para encontrar falsos negativos; seleccionar solo errores visibles
en detecciones dejaría esas omisiones fuera.

