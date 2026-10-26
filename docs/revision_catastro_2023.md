# Revisión de Catastro para PNOA 2023

Revisión realizada el 16 de octubre de 2026. El registro de la sustitución y los
recuentos finales están en `outputs/revision_catastro_2023.json`.

## Qué hacía el proyecto

Había 103 ZIP INSPIRE BU de la DGC, generados el 21–22 de febrero de 2026.
`catastro.py` conservaba únicamente las geometrías de `Building`: descartaba
referencias, usos, fechas, partes y otras construcciones. El GeoPackage tenía
153.011 filas, incluidas ocho geometrías duplicadas. Los identificadores
posteriores eran FID locales, sin correspondencia estable con Catastro.

El índice contenía 100.591 ventanas de 512 píxeles de 63 ortofotos. Solo
incluía ventanas que intersectaban edificios. La inferencia regional U-Net
además contaba exclusivamente píxeles dentro de esas huellas y excluía
edificios parcialmente cubiertos. Se perdían tanto instalaciones exteriores
como desbordes de máscaras. No se guardaron las máscaras de aquella inferencia
regional, por lo que sus tablas no permiten recuperar lo descartado.

El seguimiento posterior sí conservaba componentes completos en los recortes
recuperados y tenía un recorrido fuera del índice. Seguía usando geometrías
y FID de 2026 para describirlos. No equivalía a cobertura regional completa.

También había una exclusión territorial adicional: `municipios.py` filtraba
los códigos IGN por `34063939`, omitiendo la Comunidad de Campoo y Cabuérniga,
cuyo código es `34063953072`. Son aproximadamente 68,87 km². Se ha incluido
con su nombre y código oficiales como territorio comunal, separado de los
102 municipios. El límite autonómico IGN se conserva como capa propia para
definir la extensión a analizar.

## Fuente seleccionada

Se utilizan dos productos oficiales compatibles temporalmente:

* **Edificaciones CBT23/LULC23**, Gobierno de Cantabria, descargadas desde sus
  [paquetes oficiales por hojas](https://geoservicios.cantabria.es/inspire/rest/services/Servicio_de_Descarga/MapServer/67).
  La [documentación metodológica](https://geoservicios.cantabria.es/Series2/CBT23_LULC23/DOC_AUX/Documentacion_LULC_Cantabria_v2_0.pdf)
  identifica imágenes de agosto de 2023, LiDAR de octubre de 2023 y datos de
  la Dirección General del Catastro. Incluye revisión humana y clasificación
  de usos. La publicación se anunció el 14 de octubre de 2024 en
  [Mapas Cantabria](https://mapas.cantabria.es/).
* **Parcelario «Catastro 2023»**, capa 15 del servicio oficial
  [Cdt](https://geoservicios.cantabria.es/inspire/rest/services/Cdt/MapServer/15).
  El anuncio del 1 de febrero de 2023 lo sitúa a principios de ese año.
  Se conservan sus 627.018 entidades con referencias de 14 caracteres,
  código municipal catastral, masa, parcela y tipo.

No se ha encontrado un volcado abierto INSPIRE BU congelado exactamente en
2023. El servicio ATOM de la DGC sirve entregas actualizadas periódicamente,
no una reconstrucción histórica por fecha. La
[cartografía vectorial histórica de la DGC](https://www.catastro.hacienda.gob.es/ayuda/ayuda_antecedentes.htm)
permite filtrar altas y bajas, pero su descarga necesita identificación en
la Sede. La alternativa elegida evita filtrar retrospectivamente edificios
actuales por año de construcción, lo que no recuperaría demoliciones ni
geometrías anteriores.

La limitación es explícita: **CBT23 es cartografía derivada, no el registro
INSPIRE BU de 2023**. Los polígonos pueden fragmentar edificios, incluir
elementos auxiliares o quedar cortados entre hojas. No identifican planos de
tejado ni instalaciones reales. Tampoco hay un día exacto común al parcelario,
CBT23 y cada pasada PNOA.

Se descargaron las 812 hojas oficiales. Tras pasar a 2D, reparar geometrías
y separar componentes poligonales, se retiraron 37.049 duplicados geométricos
exactos y quedaron 511.749 unidades cartográficas. Ese número no debe compararse
como recuento físico con los antiguos 153.003 edificios únicos.

Los originales, índices, metadatos, metodología, licencia y SHA256 de las
descargas están en `data/raw/catastro/2023/`. No se han obtenido atributos
personales ni se han descargado imágenes teseladas del visor.

## Estructura y relaciones

Se mantiene el GeoPackage existente `data/geografia/catastro/edificios_cantabria.gpkg`,
con las capas `edificios` y `parcelas`. Los identificadores `cbt23_...` se
derivan de la geometría normalizada; **no son referencias catastrales**.
`edificios_parcelas.csv` relaciona ambas capas mediante intersecciones con área
positiva. Se conservan todos los candidatos de una relación muchos a muchos.

La inferencia guarda `detecciones.gpkg`, su cobertura procesada y los CSV
`detecciones_edificios`, `detecciones_parcelas` y `detecciones_municipios`.
Cada predicción conserva geometría, área, tesela de origen y marca de borde.
Las relaciones con edificios incluyen el ID de cubierta cartográfica usado
por el seguimiento, el uso oficial, el subtipo y la fracción de solape.
Si hay varios edificios se conservan sus usos por separado. La distancia al
edificio más próximo describe contexto;
no asigna automáticamente soporte ni descarta exteriores.

Las máscaras y las observaciones del seguimiento se conservan. Se recalculan
sus relaciones sobre 2023, verificando que las huellas de las observaciones y
de las geometrías originales permanecen iguales. Los nombres heredados
`edificio_N` son IDs de candidatos; los campos de asociación actuales usan
CBT23. Los candidatos antiguos sin máscara de 2023 conservan su ventana de
búsqueda, sin reutilizar la huella de 2026 como si siguiera vigente.

En esta revisión se exportaron también relaciones de componentes y
emplazamientos con parcelas. Aquel resumen del seguimiento se ha retirado:
sus agrupaciones no identificaban instalaciones reales. Se conservan las
respuestas y máscaras originales para el etiquetado de polígonos descrito
en la [guía](guia.md#etiquetado-histórico). El contexto de campañas anteriores
sigue usando 2023, identificado expresamente en la base.

## Usos

Se conservan `uso_del_suelo`, `subtipo` y su procedencia. La fuente distingue
vivienda colectiva o no colectiva, usos industriales/logísticos, agrícolas,
educativos, sanitarios, religiosos, administrativos y otros. Son usos
cartográficos que pueden provenir del contexto de la parcela; no certifican
el uso individual de cada local ni permiten deducir titularidad pública.
No se convierte «industrial» en «nave» ni «educativo» en «edificio público».

## Cobertura fuera de edificios

El índice único incluye las teselas de Cantabria con cobertura local, haya o
no edificaciones. Catastro solo aporta un recuento y una prioridad descriptiva.
Se descuentan solapes de hojas y se recorta la predicción al territorio
procesado, sin usar edificios como máscara de exclusión. Los píxeles sin
datos declarados o con RGB completamente cero no generan detecciones.

Para un inventario exhaustivo no resulta defendible limitarse a un buffer de
edificios, parcelas edificadas o instalaciones eléctricas conocidas: cualquiera
de ellos puede omitir instalaciones aisladas. Se deja preparado el recorrido
territorial completo a 15 cm. Trabajar primero sobre zonas pequeñas permite
validar costes y errores. Un muestreo del resto del territorio serviría para
estimar prevalencia u omisiones, pero no sustituye un inventario exhaustivo.
No se introduce una pasada a menor resolución sin validar su sensibilidad.

`teselas.gpkg` identifica cobertura nominal, hojas y zonas sin ortofoto local.
El índice regenerado contiene 957.468 ventanas de 65 TIFF: 94.677 intersectan
edificaciones y 862.791 no. Incluye las hojas alternativas en solapes para
recuperar posibles bordes negros de la primera imagen; las detecciones y su
cobertura se contabilizan una sola vez. Quedan 24.307 m² de territorio nominal
sin ortofoto local en Castro-Urdiales, delimitados en la capa `sin_ortofoto`.
La extensión de un TIFF no certifica visibilidad ni calidad radiométrica; los
archivos locales no declaran nodata. Hay que revisar los pequeños huecos y
las zonas con sombras, bordes o imágenes inutilizables antes de afirmar que
la cobertura es completa.

## Sustitución y pruebas

Se reconstruyen el GeoPackage, las relaciones con parcelas, el índice y los
recuentos de edificios de los manifiestos del pool, labeling y validación.
Se regeneran las relaciones, auditorías, capas y resúmenes del seguimiento y
un nuevo avance histórico. Se conservan imágenes, etiquetas, pesos, particiones
y observaciones humanas: sus píxeles no dependen de la nueva cartografía.
El conjunto de entrenamiento y test conserva el sesgo de selección antiguo;
no se presenta como una nueva muestra territorial aleatoria.

Tras comprobar la sustitución se retiran los ZIP de 2026, los productos
regionales U-Net/YOLO y energéticos antiguos, las tablas y capas importadas
del piloto que dependían de 2026, y las figuras/PDF regionales obsoletos.
`revisiones.csv` permanece como fuente humana ya importada en SQLite.
La memoria editable conserva las evaluaciones de modelos y declara pendientes
los nuevos resultados regionales. No se publican cifras regionales nuevas
sin haber repetido la inferencia.

Pruebas realizadas:

* Zona de 400 × 400 m en Camargo: 36 teselas, incluidas 27 sin edificios;
  ninguna detección con los pesos y el umbral actuales.
* Zona de 176,8 × 176,8 m alrededor de un caso conocido: nueve teselas;
  una predicción de 18,8325 m² completamente fuera de edificaciones, conservada
  a unos 2,04 m de la más cercana. Es una predicción, no una certificación
  de superficie real.
* Pruebas automáticas de conservación de exteriores, hojas solapadas,
  recuperación de una hoja alternativa ante bordes negros, asociaciones con
  varias parcelas, cubiertas parcialmente observadas y energía en municipios
  con detecciones exteriores y ninguna cubierta positiva. Cuatro pruebas pasan.

Los comandos de prueba y ejecución regional están en el [README](../README.md).
Antes de procesar toda Cantabria queda revisar visualmente la prueba y reunir
una validación independiente de suelo, prados, aparcamientos y cubiertas.
La calibración energética no extrapola a suelo el sesgo del antiguo test:
los escenarios usan superficie detectada y densidades supuestas.

Realizado a partir de datos del Gobierno de Cantabria disponibles de forma
gratuita en https://mapas.cantabria.es; parcelario de la Dirección General del
Catastro y ortofotos PNOA de IGN/CNIG. Se conserva la licencia entregada con
CBT23 junto a los originales.

## Limpieza de los temporales

Limpieza completada el 22 de octubre de 2026. Se han retirado las copias de
fuentes e índices del piloto, `data/raw/catastro/2023/muestra.*` y el registro
auxiliar de descarga. También se han eliminado los informes y borradores del
visor histórico descartado. Se conservan las 812 hojas oficiales, el parcelario,
las referencias definitivas y los resultados válidos de la prueba de 2023.

Las imágenes y anotaciones manuales, la base de respuestas humanas y el avance
regional se conservan. El etiquetado histórico vigente se documenta en la
[guía](guia.md#etiquetado-histórico).
