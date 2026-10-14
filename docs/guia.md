# Guía de validación y seguimiento

Esta guía reúne los flujos complementarios al proceso de 2023 descrito en el
[README](../README.md). Los comandos se ejecutan desde la raíz del repositorio.

- [Validación de otras resoluciones](#validación-de-otras-resoluciones)
- [Revisión histórica fotovoltaica](#revisión-histórica-fotovoltaica)
- [Indicadores históricos](#indicadores-históricos)

## Validación de otras resoluciones

Evalúa la U-Net congelada sobre las mismas localizaciones del test de 2023,
usando otra campaña y resolución nativa. Mantiene `weights/unet_resnet34.pt`,
RGB, normalización ImageNet, sigmoid y umbral 0,5. Sus etiquetas permanecen
en la campaña: no entran en entrenamiento ni se incorporan al pool.

### Preparación y etiquetas

`validaciones.json` contiene una entrada por campaña. `20cm` utiliza la
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

Para otra resolución, añade una entrada completa en `validaciones.json` y pasa
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

### Alcance y comprobación

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

Las pruebas cubren ventanas, cobertura, conversión RGB, revisión de negativos,
configuración, métricas y bootstrap:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_validacion.py -v
```

## Revisión histórica fotovoltaica

Prepara y revisa 2020, 2017 y 2014 con 2023 como referencia fija.
Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento
```

La configuración está en `seguimiento.json`. El lote comprende 12 imágenes
compartidas por defecto; todas las detecciones de una imagen van seguidas.
El recorrido es **2020 → 2017 → 2014 → siguiente emplazamiento**.
Una ausencia o duda no elimina campañas anteriores. Las predicciones son ayuda
visual y no se convierten automáticamente en observaciones humanas.

### Datos y recuperación

El índice original tenía 100.591 teselas de 512 píxeles de 63 ortofotos locales.
Seleccionaba teselas con edificios; los indicadores contaban píxeles asociados
a Catastro y edificios con cobertura completa. Las 3.188 entradas positivas
eran registros catastrales internos, no instalaciones verificadas.
El piloto seleccionó 400 registros, divididos en 412 candidatas por cubierta,
y conservó 44 decisiones.

Las máscaras regionales U-Net no se guardaron y los resúmenes YOLO no permiten
reconstruir polígonos. Recuperar la señal exige inferencia con los pesos
congelados, sin presentarla como copia exacta de predicciones perdidas.

Se recuperaron los componentes de los 400 recortes iniciales de 2023,
sin mínimo de superficie ni recorte catastral:

| Grupo | Componentes | Superficie aproximada |
| --- | ---: | ---: |
| Antes de Catastro | 1.180 | 24.450 m² |
| Intersectan edificios, incluidas intersecciones parciales | 1.067 | 23.530 m² |
| Próximos sin intersección | 36 | 210 m² |
| Exteriores más alejados | 77 | 710 m² |
| Total completamente fuera | 113 | 920 m² |

Los componentes parciales aportan unos 1.480 m² fuera de Catastro.
Son superficies predichas, no confirmadas, de esos recortes iniciales.
Los resultados actuales añaden ejemplos del pool y lotes regionales por origen.

En 275 componentes parciales mayores de 1 m², la distancia de un punto interior
del desborde al edificio tuvo mediana 0,42 m y percentil 95 de 1,30 m.
Ese percentil describe proximidad: no estima desplazamiento físico ni corrige
máscaras. Una asociación próxima no confirma presencia ni soporte.

`runs/evaluacion/seguimiento/auditoria_visual.html` muestra distintos tamaños,
bordes y relaciones catastrales. No representa cobertura regional completa.
Los otros 2.788 registros del inventario son localizaciones pendientes.
Las zonas sin componentes de 2023 asignados se apartan del recorrido principal,
conservando respuestas y acceso desde las herramientas avanzadas.

Para recuperar ventanas de las ortofotos de 2023 ya descargadas:

```powershell
# Ventanas del índice original, conservando toda la señal.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento region --modo indice --lote 100
# Ventanas excluidas por no tener edificios.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento region --modo fuera_indice --lote 100
# Recorrido completo por ventanas locales.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento region --modo todas --lote 100
```

Repetir continúa el cursor y omite ventanas terminadas, incluso al cambiar
de modo. Se guardan PNG donde hay detecciones y metadatos para los negativos.
Los componentes solapados conservan su origen: sus áreas no se deben sumar
como superficie regional única. La recuperación regional completa sigue pendiente.

### Revisión en el visor

La unidad es un **emplazamiento observado**, con agrupación corregible.
No equivale a una titularidad o instalación administrativa. Componentes
originales, asociaciones, agrupaciones y respuestas se conservan por separado.

Izquierda: 2023. Derecha: campaña histórica. Rueda para ampliar y arrastre para
desplazar ambas fotos. El zoom se conserva entre años del mismo caso.
**Centrar objetivo** recupera el caso si quedó fuera de vista.

| Localizador o capa | Interpretación |
| --- | --- |
| Objetivo naranja | Caso al que se asignan las respuestas, localizado con sus componentes originales. |
| Otro caso azul | Candidato vecino; pulsarlo cambia de caso sin copiar respuestas. |
| Zona provisional gris y discontinua | Ubicación antigua sin componentes de 2023 asociados; no acredita presencia ni ausencia. |
| Predicción verde | Señal del modelo para la foto visible. |
| Geometría original gris | Grupo conservado del emplazamiento. |
| Catastro amarillo | Huellas opcionales, sin carácter histórico por campaña. |

Los localizadores siguen visibles al ocultar capas. En el histórico señalan
la ubicación del objetivo de 2023, no una máscara de paneles de ese año.
Si las fotos se desplazan, comprueba el entorno.

La primera comparación comprueba también la foto izquierda. Sin respuesta de
2023, el selector propone presencia en casos localizados y duda en provisionales
o candidatos manuales; se confirma al responder al histórico. Cámbialo si
la referencia es ausente, dudosa o no evaluable. Las referencias ya revisadas
se conservan salvo corrección explícita.

| Tecla | Acción |
| --- | --- |
| 1 | Hay paneles en la campaña histórica. |
| 2 | No hay paneles; exige visibilidad suficiente para descartarlos. |
| 3 | No está claro si el objeto es un panel. |
| 4 | La resolución, sombras o cobertura impiden evaluar. |
| 5 | No hay paneles en ambas fotos; corrige 2023 y guarda ausencia histórica. |
| ← / → | Anterior / siguiente pendiente; saltar no inventa respuestas. |
| Ctrl+Z | Deshacer la última acción, incluso de otra sesión. |
| F1 | Guía breve del visor. |

Los atajos numéricos se desactivan al editar texto. Cada respuesta guarda
y avanza al año anterior; después de 2014 abre otro emplazamiento en 2020.
Una foto peor no equivale a ausencia. Si 2023 es falso pero antes había
paneles, corrige solo el selector y responde presencia al histórico.

**Cambio entre las fotos** anota más paneles, menos paneles o distinta
distribución en 2023 respecto al histórico mostrado, conservando su decisión
y notas. Si 2023 sigue pendiente, el cambio queda como borrador hasta la primera
comparación. No exige contar módulos ni estimar porcentajes sobre fotos borrosas.

El soporte se indica por emplazamiento: cubierta, cubierta catastrada,
construcción no catastrada, marquesina, suelo, mixto o desconocido.
Notas y etiquetas describen retiradas, reinstalaciones, cambios de edificio,
sombras o incertidumbre sin alterar las respuestas de otras campañas.

**Confirmar agrupación** se usa al comprobar qué pertenece al emplazamiento
y es independiente de las respuestas por año. Fuera del piloto, componentes
que se tocan o se separan hasta 0,6 m pueden proponerse juntos. Compartir
edificio no basta y un grupo revisado no se amplía automáticamente.

Las herramientas avanzadas corrigen asociaciones, dividen o fusionan grupos
y muestran originales archivados. Al repartir componentes, las comas separan
componentes y el punto y coma separa grupos. Originales y observaciones
se conservan; sus respuestas no se copian a los nuevos grupos.
Los identificadores de cubiertas se mantienen separados de los componentes.

**Más contexto** prepara ventanas pequeñas alrededor de los componentes
conservando caso, año y notas. Si no se ve todo, usa `zona_visible`;
una ausencia parcial no fecha el emplazamiento completo. Para una agrupación
de cubiertas sin resolver, usa `agrupacion_cubiertas`.

**Añadir candidato histórico** permite señalar una instalación retirada antes
de 2023 o que el modelo omitió. Pulsa sobre cualquiera de las fotos; Esc cancela.
Reutiliza las fotos y crea un caso independiente. Pulsar sobre un caso localizado
lo selecciona sin duplicarlo.

### Persistencia y lotes

El progreso se guarda en `data/datasets/seguimiento/seguimiento.sqlite`.
Decisiones, correcciones, notas y posición del visor se confirman mediante
transacciones. La primera comparación guarda ambas respuestas como una acción
reversible: deshacer restaura las dos juntas. Al abrir se recupera el cursor.

Los CSV, GeoPackage y PNG del piloto son fuentes de importación conservadas.
Después de importar, editar `revisiones.csv` no cambia el progreso de SQLite.
Borradores y versiones anteriores quedan en metadatos e historial de acciones.

El visor prepara hasta un lote de reserva en segundo plano y permite usar
cada imagen terminada. Un fallo conserva respuestas y permite reintentar;
`precarga: false` desactiva la anticipación. Los cálculos geográficos se hacen
fuera de transacciones de escritura para permitir responder durante la preparación.

2023 reutiliza recortes o lee ventanas de TIFF locales. Los históricos usan
el WCS configurado: coberturas 13/2020 a 15 cm y 11/2017 y 10/2014 a 25 cm.
El RGB uint16 de 2017 se divide por 257. Se conserva el encuadre y resolución
nativa; el modelo recibe 512 píxeles. La validación de 2024 queda fuera.
Se comprueban cobertura, formato y espacio libre; los GeoTIFF de descarga
permanecen en memoria.

```powershell
# Preparar recursos sin abrir el visor.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento preparar --lote 12
# Recuperar un lote del piloto y regenerar su auditoría.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento auditar --lote 12
# Regenerar resultados; --gpkg añade geometrías.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento resumen --gpkg
```

### Resultados y límites

`runs/evaluacion/seguimiento/` contiene resúmenes, observaciones, emplazamientos,
agrupaciones, acciones, intervalos, distribución de aparición, resultados
municipales y auditoría. La exportación opcional añade `seguimiento.gpkg`.
Cerrar el visor actualiza los CSV y el resumen.

Primera presencia significa primera campaña en que se observó el emplazamiento.
El intervalo requiere ausencias anteriores claras y completas. Presencia en
2014 significa «ya visible en 2014», sin fecha inicial inventada.
Dudas, campañas pendientes y agrupaciones ambiguas conservan la incertidumbre.
Presencia seguida de ausencia se señala como anomalía, sin corregirla automáticamente.

Los recuentos distinguen candidatos, observaciones y emplazamientos con
agrupación y secuencia resueltas. Tiempo medio y trabajo pendiente se calculan
sobre las tres respuestas históricas por caso, incluyendo comprobar 2023
en el primer par y las pausas. No estiman las regiones todavía sin recuperar.

El [avance de indicadores históricos](#indicadores-históricos) exporta por
separado una instantánea de solo lectura con periodos y cobertura municipal.
Muestra agrupaciones pendientes sin certificarlas como instalaciones.

Las pruebas cubren persistencia, correcciones, deshacer, cronología, agrupación,
precarga, contexto y navegación. Los casos reales utilizan copias temporales
de SQLite. Las pruebas de software no certifican soporte físico ni cambios
de una instalación: requieren revisión humana.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Indicadores históricos

Se puede generar un resultado provisional mientras continúa el etiquetado:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_historicos
```

Este comando es independiente del visor. Abre SQLite en modo de solo lectura,
copia una instantánea en memoria por bloques pequeños y cierra el origen antes
de calcular indicadores. No escribe respuestas, metadatos, agrupaciones ni la
posición del visor. No descarga imágenes, ejecuta inferencia, consulta PVGIS ni
modifica el mapa o los resultados anteriores. Si no consigue obtener la
instantánea en diez segundos, se detiene y puede reintentarse después.

Cada ejecución crea una carpeta nueva en
`runs/indicadores/seguimiento_historico/avance_FECHA_HORA/` con:

| Archivo | Contenido |
| --- | --- |
| `avance_historico.html` | Informe visual autocontenido, con barras por periodo y cobertura municipal. Se abre directamente en el navegador. |
| `emplazamientos_historicos.csv` | Decisiones, intervalos, estado de revisión y agrupación de cada emplazamiento activo. |
| `indicadores_municipales.csv` | Cobertura de revisión, referencias presentes y casos con periodo utilizable. |
| `despliegue_por_periodo.csv` | Recuentos y ritmo medio nominal por municipio y periodo entre vuelos. |
| `emplazamientos_mapa.geojson` | Puntos WGS84 de los objetivos localizados, con los atributos del CSV. Preparado para incorporar al mapa después. |
| `resumen.json` | Totales, fecha UTC de la instantánea, última respuesta y última acción disponibles, alcance y limitaciones. |

Los archivos son una fotografía del progreso, no un panel que se actualice en
directo. Una nueva ejecución incluye las respuestas guardadas hasta esa nueva
instantánea y conserva los avances anteriores. Opcionalmente se puede elegir
otra base con `--base` o una carpeta nueva con `--salida`.

### Cómo interpretar el resultado

Se reutiliza la derivación de intervalos del seguimiento. La primera presencia
observada no se convierte en un año exacto de instalación. `aparicion_despues_de`
es un límite abierto: ausencia en 2017 y presencia en 2020 sitúan la aparición
después del vuelo de 2017 y hasta el de 2020. La clase `hasta_2014` indica presencia
en el primer vuelo, sin límite inferior conocido.

`incluir_en_despliegue` exige un objetivo no provisional, las cuatro campañas
respondidas, presencia humana en 2023 e historia resuelta. Las dudas, ausencias
de alcance parcial que no permiten acotar la aparición, secuencias anómalas y
observaciones con agrupación de cubiertas sin resolver no se reparten entre
periodos. Sus respuestas y motivos permanecen en el CSV individual. Una duda
posterior a una primera presencia ya acotada puede conservar ese intervalo;
se exportan las decisiones para revisar la evidencia.

La confirmación de agrupación se presenta por separado. No se exige pulsar
Confirmar agrupación para mostrar el trabajo hecho ni se cambia su estado a
confirmado. Los recuentos con agrupaciones pendientes son **emplazamientos
observados**, no instalaciones administrativas verificadas. No se cuentan
componentes de máscara ni edificios como si fueran instalaciones distintas.

Las zonas provisionales del inventario sin componentes de 2023 asignados se
conservan en el CSV y se contabilizan aparte. Los originales archivados no se
cuentan. Los candidatos manuales siguen disponibles aunque no tengan máscara;
en ese caso no se inventa una superficie ni un punto de paneles.

El porcentaje municipal divide respuestas históricas guardadas entre respuestas
esperadas de los candidatos localizados de la instantánea. Puede cambiar al
localizar nuevos casos. Si no hay objetivos localizados se deja vacío, no se
interpreta como ausencia de fotovoltaica en ese municipio.

El ritmo medio divide apariciones de un periodo cerrado entre la diferencia de
años de las campañas: por ejemplo, 60 casos entre 2020 y 2023 dan 20 casos/año.
Es una media nominal del intervalo, no evidencia de que apareciesen 20 cada año.
No se calcula un ritmo anterior al primer vuelo. Comparar municipios requiere
considerar su cobertura de revisión y de detección; no se extrapola a los casos
pendientes. Las retiradas anteriores a 2023 no están recuperadas sistemáticamente.

### Superficie y siguiente fase

`superficie_fv_detectada_2023_m2` es el área planimétrica de la unión de los
componentes originales asignados, sin duplicar sus solapes y sin utilizar el
área del edificio. Es una predicción, no una delimitación manual certificada.
Se conserva incluso en candidatos rechazados para permitir auditoría; hay que
usar las decisiones humanas para elegir los casos de un análisis de potencia.

Este avance todavía no estima potencia por época ni generación histórica.
La siguiente fase podrá combinar intervalos y coeficientes documentados de
potencia por m² para escenarios conservador, central y superior. Se deberán
mantener separados los supuestos sobre superficie, tecnología y fecha. No se
puede atribuir toda la superficie de 2023 a la primera aparición: ampliaciones
o sustituciones pueden contener módulos de distintas épocas. No anotar un
cambio tampoco certifica que la superficie se haya mantenido constante.

El GeoJSON permite preparar la futura capa de periodos sin modificar el mapa
actual. Los CSV y el informe ya permiten mostrar el avance de la revisión sin
otra ronda de etiquetado.
