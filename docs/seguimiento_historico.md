# Revisión histórica fotovoltaica

Desde la raíz, un único punto de entrada prepara y continúa lotes pequeños:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento
```

La configuración está en `seguimiento.json`. El lote predeterminado comprende
12 imágenes compartidas; no hay que terminarlo en una sesión. Las detecciones
de cada imagen van seguidas y ninguna imagen se divide al alcanzar el límite del
lote. Las campañas se observan
independientemente. El recorrido manual es **2020 → 2017 → 2014 → siguiente
emplazamiento**, con 2023 como referencia fija a la izquierda. La respuesta de
2023 se confirma dentro de la primera comparación, sin una pantalla adicional.
Una ausencia, duda o
referencia incorrecta no elimina campañas anteriores. Las predicciones son ayuda
visual y nunca se convierten automáticamente en decisiones humanas.

## Qué se encontró en el flujo anterior

El índice tenía 100.591 teselas de 512 píxeles, de 63 ortofotos locales. Se
seleccionaban **solo teselas con edificios**, aunque la U-Net predecía todos los
píxeles dentro de cada tesela seleccionada. Los indicadores descartaban los
píxeles sin etiqueta de edificio y excluían registros sin cobertura completa.
Las 3.188 entradas positivas eran registros catastrales internos, no todos los
componentes ni un número verificado de edificios independientes o instalaciones.

El piloto sorteó 400 de esos registros y los dividió en 412 candidatas por
cubierta. Conservaba 400 imágenes de 2023, 400 de 2020, sus predicciones y **44
decisiones**: diez presencias en ambas campañas, 23 ausencias de referencia y una
duda. Había 53 agrupaciones ambiguas y 92 candidatas parcialmente fuera del
encuadre. Su interfaz ofrecía presente, ausente, dudosa e imagen inválida.
La referencia se recortaba con Catastro. El histórico usaba una zona compatible
limitada por el edificio con una tolerancia fija de 2 m. El recorrido se detenía
en ausencias. Eso perdía superficies exteriores, otros soportes, posibles
retiradas y reinstalaciones, y cambios de soporte o distribución.

Los resultados regionales no guardaron máscaras U-Net; los CSV regionales de
YOLO solo contienen cantidades, área y centro de tesela. No permiten reconstruir
los polígonos originales. Las máscaras anotadas del pool son etiquetas humanas,
no las predicciones perdidas. Por tanto, la recuperación exige repetir inferencia
con los pesos congelados. No se presenta la reinferencia como una copia exacta de
las máscaras perdidas: algunos recortes tienen otro encuadre.

## Recuperación y auditoría

Se recuperó la señal **completa** de los 400 recortes de 2023 existentes, sin
descargarlos de nuevo, sin mínimo de superficie y sin recorte catastral:

| Grupo en esos recortes | Componentes | Superficie aproximada |
| --- | ---: | ---: |
| Antes de Catastro | 1.180 | 24.450 m² |
| Intersectan edificios, incluidas intersecciones parciales | 1.067 | 23.530 m² |
| Próximos sin intersección | 36 | 210 m² |
| Exteriores más alejados | 77 | 710 m² |
| Total completamente fuera, que el flujo anterior perdía | 113 | 920 m² |

Los componentes parciales aportan además unos 1.480 m² fuera de Catastro.
Las cifras son de predicción, **no superficie fotovoltaica confirmada**. Hay
ejemplos visuales inequívocos de paneles en cubiertas fuera del contorno catastral,
pero todos los casos conservan su revisión humana pendiente. Esta auditoría
corresponde a los recortes preparados; no se extrapola a Cantabria.
La tabla describe los 400 recortes iniciales. Los JSON y CSV actuales añaden los
ejemplos del pool y los pequeños lotes regionales posteriores, desglosados por origen.

`runs/evaluacion/seguimiento/auditoria_visual.html` ofrece una muestra deliberada:
pequeños y grandes, distintos municipios, intersecciones claras, parciales,
próximos, exteriores, ambiguos y bordes. Se añadieron tres recortes del pool ya
etiquetado tras inspección visual: paneles en terreno junto a una vivienda,
construcciones auxiliares de jardín y estructuras junto a un aparcamiento. Se
mantiene el soporte sin confirmar hasta la revisión: una imagen aérea no siempre
distingue una marquesina de un cobertizo. Las etiquetas no acreditan la cronología.

En 275 componentes parciales mayores de 1 m² se midió la distancia de un punto
interior del desborde al edificio: mediana 0,42 m y percentil 95 de 1,30 m. Se usa
ese percentil **solo para describir próximos**, conservando la distancia exacta.
No es una estimación de desplazamiento físico: también incluye segmentación,
posición del panel y relieve del edificio. No hay traslación automática, ajuste
de máscaras, búfer universal de asociación ni filtro de conservación. El Catastro
disponible no es un registro histórico por campaña; un edificio nuevo o demolido
se anota como tal. Una asociación próxima nunca se asigna automáticamente.

Los otros 2.788 registros positivos del inventario se incorporaron como
localizaciones pendientes, conservando los identificadores del piloto. Se deben
localizar sus paneles y comprobar el encuadre; no se cuentan como emplazamientos
confirmados. También falta reconstruir las teselas originales y explorar las
zonas sin edificios. Ambas fases usan las ortofotos de 2023 **ya descargadas**:

```powershell
# Ventanas que se procesaban antes; esta vez se conservan todos sus componentes.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento region --modo indice --lote 100
# Ventanas que el índice excluía: terreno, explanadas y zonas sin edificios.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento region --modo fuera_indice --lote 100
# Alternativamente, un recorrido completo por ventanas locales del territorio.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento region --modo todas --lote 100
```

Repetir el mismo comando continúa el cursor y evita ventanas terminadas incluso
si se cambia de modo. Solo guarda PNG cuando hay detecciones y guarda metadatos
de ventanas negativas. Los componentes solapados conservan su procedencia; su
agrupación evita recuentos definitivos por fragmento. Las áreas originales de
componentes pueden solaparse y no deben sumarse como superficie regional única.
No se ha ejecutado la inferencia regional completa ni iniciado una revisión
masiva. Antes de ampliarla, comprueba la muestra, las asociaciones y agrupaciones;
las cifras regionales seguirán pendientes hasta completar la recuperación.

## Unidad de análisis y conservación

La unidad humana es un **emplazamiento observado**, con una agrupación espacial
explícita y revisable. No representa una titularidad, vivienda o instalación
administrativa. Se mantienen separados:

- Componentes originales y campaña, geometría WKB, superficie, borde y modelo.
- Asociaciones catastrales propuestas, fracción y distancia originales.
- Agrupaciones, identificadores de cubiertas y tipo físico corregibles.
- Observaciones por campaña, su alcance, etiquetas, notas y tiempo.
- Acciones anteriores y conclusiones derivadas regenerables.

Fuera del piloto, componentes que se tocan o quedan separados hasta 0,6 m pueden
proponerse juntos: cuatro píxeles del modelo para cortes pequeños. No se unen
por compartir parcela o edificio. Las propuestas siempre necesitan comprobación;
componentes que podrían unir varios grupos se señalan como ambiguos. No se
extiende automáticamente una agrupación ya revisada con nuevas detecciones.

Los grupos del piloto conservan sus identificadores y sus geometrías de revisión
anteriores. La nueva máscara completa tiene otro papel: nunca se vuelve a recortar
al contorno seleccionado. **Confirmar agrupación** confirma que lo observado
pertenece al emplazamiento. **Dividir / fusionar** reparte componentes originales
en nuevos grupos: comas entre componentes y punto y coma entre grupos. Los
originales se archivan, no se borran. Sus observaciones permanecen consultables;
no se copian como verdades a los nuevos grupos, que necesitan validación propia.
Si un único componente mezcla casos imposibles de separar con certeza, conserva
agrupación pendiente y una nota; no fuerces un recuento.

Las cubiertas tienen identificadores de parte catastral, o identificadores
manuales para cubiertas no catastradas. Dos grupos en una cubierta no duplican
esa cubierta en los recuentos confirmados. Un emplazamiento puede tener varias
cubiertas. El número de cubiertas no se infiere del número de componentes.

## Cómo revisar

Izquierda: referencia 2023. Derecha: campaña seleccionada, con su año explícito.
Las tres capas se controlan por separado. Verde es la predicción de la imagen
actual; gris muestra la geometría original del grupo; amarillo muestra Catastro
opcionalmente. Los localizadores permanecen visibles aunque desactives esas capas:

- **OBJETIVO naranja**: emplazamiento al que corresponden tus respuestas. Se
  localiza con sus componentes originales asociados, sin usar el edificio como
  sustituto de los paneles. Aparece en ambas fotos.
- **OTRO CASO azul**: candidato vecino. Un clic en su marco o rótulo lo selecciona
  y empieza su campaña pendiente más reciente. No copia respuestas del caso anterior.
- **ZONA PROVISIONAL gris y discontinua**: ubicación antigua del piloto o inventario
  sin componentes de 2023 asociados. El aviso superior lo identifica expresamente;
  esa geometría no acredita dónde hay paneles ni demuestra su ausencia. Si aún no
  tiene respuesta de 2023, se propone Dudoso hasta comprobar la foto.

Por ejemplo, en la depuradora de `edificio_13532`, la candidata antigua
`edificio_13532_cubierta_0` rodea las estructuras centrales y no tiene componentes
de paneles asociados. Los paneles de abajo a la izquierda se conservan en
`emplazamiento_63ca8740230cce8d14f7`: su marco azul permite seleccionarlos, y entonces
aparecen como OBJETIVO naranja. Las geometrías y revisiones anteriores se conservan.

El localizador del histórico señala la ubicación del objetivo de 2023, incluso
si antes no había paneles; no representa una máscara histórica ni se desplaza
para hacer coincidir las ortofotos. Si hay desplazamiento, comprueba el entorno.
Para observar presencia y primera campaña visible basta identificar el mismo
emplazamiento y distinguir los paneles; no se exige un solape píxel a píxel.
Para afirmar ausencia debes poder ver la zona suficientemente bien para descartar
paneles. Si dudas del objeto, usa 3; si la resolución, sombra o cobertura impide
decidir, usa 4 y continúa. Una fotografía peor no equivale a una ausencia.
Una ampliación solo se anota si el cambio es claro; no se cuentan módulos ni
se estiman superficies precisas a partir de una foto borrosa.
**Centrar objetivo** devuelve la vista al caso si quedó fuera al ampliar o arrastrar.
Al abrir se muestran los localizadores y la geometría original; la predicción y
Catastro se activan cuando hacen falta.
Rueda para ampliar y arrastre para desplazar ambas imágenes simultáneamente.

Revisa los tres históricos seguidos de **un mismo emplazamiento**: 2020, 2017 y
2014. Cada respuesta guarda y pasa automáticamente al año anterior; después de
2014 empieza el siguiente emplazamiento en 2020. Se saltan las campañas ya
guardadas. Nunca se muestra 2023 frente a 2023. El zoom y la posición se conservan
entre años del mismo caso y se restablecen al pasar a otro.

La primera comparación comprueba también la foto izquierda. Si 2023 aún no
tiene respuesta, el selector indica inicialmente **Presencia clara**; es una
propuesta sin guardar. El encabezado avisa expresamente de que el primer clic
confirma esa respuesta de 2023 junto con la del histórico. Si la referencia es
dudosa, no evaluable o ausente, cambia su selector antes de responder. Las
referencias ya revisadas se mantienen y se pueden corregir con ese selector.
Esto evita una cuarta etapa y mantiene una observación humana independiente.

**5 · No hay en ambas** afirma que no hay paneles en ninguna de las dos fotos:
guarda referencia falsa de 2023 y ausencia en el histórico mostrado, y avanza
con un solo clic. Si 2023 es falso pero antes sí había paneles, cambia únicamente
el selector de 2023 y responde presencia al histórico. No se rellenan los otros
años ni se termina el recorrido por encontrar una ausencia.

Una revisión separada de 2023 sumaría una cuarta respuesta a las tres históricas,
un 33 % más. El recorrido integrado requiere tres respuestas en los casos
normales y también cuando ambas fotos son negativas; corregir una referencia
dudosa o distinta requiere una acción adicional. En las 34 referencias del
piloto revisadas, 23 se marcaron ausentes, diez presentes y una dudosa. Esa muestra
no permite extrapolar la proporción de errores al inventario completo ni estimar
con precisión el tiempo adicional, pero aconseja conservar la comprobación.

La franja superior muestra la última respuesta guardada y el estado de cada año.
Mientras respondes, el visor prepara hasta un lote de imágenes de reserva en
segundo plano. Al acabar los tres años de un caso siguen las demás detecciones
de esa misma imagen, antes de cambiar de foto. Cada imagen nueva se puede usar
en cuanto está completa, sin esperar a que termine todo el lote ni pulsar un
botón de carga. La precarga no desactiva las respuestas ni cambia el caso actual.
Si revisas más deprisa que las descargas, el visor muestra el avance y continúa
automáticamente cuando llega la siguiente imagen; la ventana sigue respondiendo.
Un fallo de carga conserva las respuestas y permite reintentar con Siguiente.
La configuración `precarga: false` permite desactivar esta anticipación.

Las ubicaciones del inventario antiguo sin componentes de 2023 asignados se
apartan del recorrido principal. No se convierten en ausencias ni se borran sus
respuestas: **Herramientas avanzadas → Mostrar zonas provisionales** permite
consultarlas. Las ubicaciones aún sin procesar pueden prepararse en segundo plano
para localizar detecciones; si no aparece ningún objetivo, se evita descargar
sus tres históricos. Los candidatos añadidos manualmente siguen disponibles.
Si un filtro deja fuera todos los pendientes, el visor lo indica; cambia el
filtro para continuar. Siguiente durante la revisión salta una campaña sin
clasificarla; permanece pendiente.

| Atajo | Observación |
| --- | --- |
| 1 | Presencia clara |
| 2 | Ausencia clara en el alcance seleccionado |
| 3 | Dudoso |
| 4 | No evaluable |
| 5 | No hay en ambas: referencia 2023 falsa y ausencia del histórico mostrado |
| ← | Volver a una decisión anterior para corregirla |
| → | Siguiente pendiente; reintentar una carga fallida al llegar al final |
| Ctrl+Z | Deshacer la última acción, incluso de una sesión anterior |

Los atajos numéricos no funcionan dentro de las entradas de texto. El soporte
es independiente de las campañas: cubierta (sin exigir comprobar Catastro),
cubierta catastrada, construcción no catastrada,
marquesina/aparcamiento, suelo, mixto y desconocido. No se propone una certeza
física basándose solo en la intersección. Usa las notas y etiquetas para
ampliación, reducción, retirada, reinstalación, nuevo edificio, demolición,
reforma, desplazamiento, nube, sombra, falta de cobertura o cambios de soporte.
Las etiquetas son texto extensible, separado por comas; añadir una nueva no
requiere migrar ninguna observación.
El botón derecho sobre el campo ofrece las etiquetas habituales.

Para una ampliación respecto al histórico mostrado, comprueba que hay paneles en
ambas fotos y usa **Cambio entre las fotos → Más paneles en 2023** antes de
responder presencia al histórico. El menú también ofrece menos paneles y distribución
diferente. Añade la etiqueta a 2023 y una nota con el par comparado, conservando
su decisión y sus notas; no tienes que volver a clasificar 2023. Si la referencia
todavía no tiene respuesta, solo guarda el cambio como borrador hasta que el
primer clic confirme la referencia junto con el histórico. Las opciones se aplican a la comparación con 2023;
para otros pares usa las etiquetas y notas de la campaña más reciente del par.
No se exige contar módulos ni estimar porcentajes. No marques cambios que no
se distingan con seguridad por la resolución, sombras o desplazamiento.

| Caso visible | Qué marcar |
| --- | --- |
| Paneles en ambos años, sin cambio que puedas afirmar | Presencia en el histórico; la referencia se confirma con el primer clic |
| Paneles en ambos, más o menos en 2023 | Presencia en el histórico y el cambio correspondiente |
| No había y después sí | Ausencia en el anterior y presencia en el posterior; el intervalo se deriva automáticamente |
| Paneles en terreno o jardín | Presencia y soporte suelo |
| Paneles en techo y suelo dentro del mismo emplazamiento | Presencia y soporte mixto |
| Detección falsa de 2023 y ningún panel en el histórico | 5; guarda ambas respuestas y avanza |
| Detección falsa de 2023 y paneles en el histórico | Corrige solo el selector de 2023 y responde presencia al histórico |
| No sabes si el objeto son paneles | Dudoso |
| La imagen impide comprobar la presencia | No evaluable |

**Cómo revisar (F1)** abre esta guía breve dentro del visor. Las herramientas
de división, fusión, asociación, candidatos históricos y auditoría están bajo
**Herramientas avanzadas**. **Confirmar agrupación** se usa una vez cuando hayas
comprobado qué pertenece al emplazamiento; es independiente de las respuestas
por campaña y no requiere volver a pulsar presencia.

**Más contexto** obtiene ventanas contiguas pequeñas de las cuatro campañas;
lo hace en segundo plano, muestra el avance y conserva el caso y su campaña.
Se centra en los componentes asociados cuando existen, en lugar del antiguo
edificio. No modifica predicciones ni decisiones. Si no se ve todo, guarda
`zona_visible`. En casos nuevos, el aviso antiguo de edificio cortado no impone
un alcance parcial si sus componentes de paneles están completos; las observaciones
y borradores que ya guardaste conservan su alcance.
Una ausencia parcial no se usa para fechar el emplazamiento completo. Si hay
varias cubiertas cuya instalación no se localiza, usa `agrupacion_cubiertas`.
**Añadir candidato histórico** permite registrar una instalación retirada antes
de 2023 o que el modelo omitió: pulsa ese botón y haz clic sobre ella en cualquiera
de las fotos. Esc cancela. Reutiliza las fotos de ese recorte y crea un caso
independiente, sin copiar respuestas ni exigir una detección de 2023.
Las coordenadas se calculan con el zoom y el desplazamiento actuales.
Si pulsas sobre la geometría de un caso ya localizado, se selecciona ese caso
sin duplicarlo. Para candidatos manuales, la referencia de 2023 empieza propuesta
como dudosa; compruébala en la foto izquierda antes de confirmar la comparación.

El soporte y la confirmación de agrupación se indican una vez por caso cuando
puedas resolverlos. La agrupación acredita que lo seleccionado corresponde a un
emplazamiento; evita contar fragmentos o mezclar vecinos. Si queda dudosa, se
conserva pendiente. Las respuestas por año permanecen guardadas aunque estos
datos sigan sin resolver; se pueden completar después sin repetir los años.

Filtros: campaña, municipio, estado, soporte, relación con Catastro y muestra
previa. La lista permite consultar revisados y originales archivados. Cambiar
soporte o asociación no cambia observaciones. El resumen solo cuenta como
emplazamientos confirmados los que tienen agrupación confirmada y presencia
humana; los demás siguen apareciendo como pendientes o incertidumbres.

## Guardado, lotes y espacio

Todo el progreso actual se guarda en
`data/datasets/seguimiento/seguimiento.sqlite`, con transacciones SQLite,
`synchronous=FULL`, comprobación de claves y diario transitorio. Cada decisión,
corrección, nota, soporte y posición del visor se confirma inmediatamente. No
hay CSV de estado nuevo por ejecución. Cerrar la ventana o interrumpir el proceso
no requiere un botón Guardar. Al abrir se recupera la posición y se sigue el lote
preparado mientras el visor obtiene el siguiente en segundo plano. Los casos
terminados no vuelven a prepararse automáticamente. La geografía y los cálculos
de agrupación de la precarga se ejecutan fuera de las transacciones de escritura,
para permitir guardar respuestas durante el procesamiento.
Las observaciones son únicas por grupo y campaña;
sus versiones anteriores quedan en el historial de acciones.
La primera respuesta de una comparación guarda ambas campañas en una única
transacción y una sola acción reversible: Ctrl+Z restaura ambas juntas. Una
referencia ya revisada se conserva al responder históricos, salvo la corrección
explícita de su selector o el botón No hay en ambas. El contador del visor mide
las tres respuestas históricas por caso; las referencias se muestran aparte.

Los CSV, GeoPackage y PNG antiguos del piloto quedan como fuentes originales de
importación, sin modificarlos. Después de importar, la base es el único origen de
verdad; modificar `revisiones.csv` fuera del visor no modifica el progreso actual.
Su SHA256 inicial es
`c6cd30f6eaa09f736f6f1431ffdb75e8444d5731dea2aa6d937fcc4f4d38f3c3`.

Los recortes existentes de 2023 y 2020 se reutilizan. Para nuevos casos de 2023
se lee una ventana de los TIFF locales, nunca la hoja entera. Las campañas
históricas usan el WCS oficial:
<https://geoservicios.cantabria.es/inspire/services/Ortofoto_Series_WCS/MapServer/WCSServer>.
Se verificaron DescribeCoverage y las cuatro coberturas: 14/2023 y 13/2020 a
15 cm; 11/2017 y 10/2014 a 25 cm. El RGB de 2017 es uint16 y se divide entre
257; no se realza cada imagen por separado. Se conserva la resolución solicitada
y el encuadre; la entrada del modelo se remuestrea a 512 como en la validación.
Los datos de validación 2024 no forman parte de esta cronología.

Antes de preparar se imprime el espacio RGB máximo estimado y el libre. Se
descarga cada ventana con límite de 8 MB y GeoTIFF únicamente en memoria.
Imágenes y señales tienen rutas fijas. Una interrupción deja terminadas las
ventanas confirmadas en la base; repetir la preparación no modifica revisiones.

```powershell
# Preparar sin abrir el visor, por ejemplo para una próxima sesión.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento preparar --lote 12
# Recuperar otro lote del piloto y regenerar auditoría sin revisión manual.
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento auditar --lote 12
```

## Resultados

Cerrar el visor actualiza los CSV y el resumen; **Resumen** añade GeoPackage.
También se pueden regenerar sin abrir la interfaz:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.seguimiento resumen --gpkg
```

La carpeta única es `runs/evaluacion/seguimiento/`. Contiene `resumen.json`,
`resumen.csv`, `resumen_correo.txt`, `observaciones.csv`, `emplazamientos.csv`,
`agrupaciones.csv`, `historial_acciones.csv`, `intervalos_aparicion.csv`,
`distribucion_aparicion.csv`, `resultados_municipales.csv`, la auditoría y
`seguimiento.gpkg` con geometrías originales y grupos. Los CSV contienen las
observaciones independientes y los intervalos, incluso de grupos archivados.

Primera presencia significa **primera campaña en la que se vio**. El intervalo
de aparición solo se deriva cuando las campañas anteriores tienen ausencias
claras completas. Una presencia en 2014 significa «ya visible en 2014», sin
límite inferior inventado. Dudas, falta de imágenes o agrupaciones ambiguas
conservan el intervalo sin resolver. Una presencia seguida de ausencia se
identifica como secuencia anómala, incluida la retirada con posible reinstalación.
No se aplica una corrección monotónica silenciosa.

Los recuentos por campaña y municipio distinguen candidatos, observaciones y
emplazamientos confirmados, tipos de soporte y cubiertas. Se incluyen falsos
positivos explícitos, recuperados exteriores, etiquetas de cambio, incertidumbre,
progreso y tiempo medio. El tiempo medio y la estimación del trabajo pendiente
se calculan sobre respuestas históricas; incluyen comprobar 2023 dentro del
primer par y las pausas. Las referencias pendientes se informan aparte, sin
contarlas como una cuarta etapa. La estimación restante solo
cubre el inventario preparado, no las detecciones regionales aún desconocidas.

## Comprobación realizada

Se probó la reinferencia completa de los 400 recortes reales, reutilización de
2020 y descarga de 2014/2017 en una muestra deliberada. Se inspeccionaron el
visor y la galería. Las pruebas automatizadas cubren guardado durable y cierre
abrupto, reapertura y corrección, deshacer, conservación de geometría exterior y
un píxel positivo, relaciones parciales y próximas, soportes independientes,
ampliaciones, referencia falsa, ambos años negativos, edificio nuevo mediante
etiqueta, no evaluable, secuencia no monotónica y división/fusión reversible.
Una copia SQLite del inventario real permite probar decisiones sin añadir
revisiones de prueba al progreso del usuario. Los casos físicos de soporte y
cambio histórico siguen necesitando la validación humana de la muestra; las
pruebas de software no certifican que un emplazamiento sea una marquesina ni
que una diferencia de imagen sea realmente una ampliación.
Las pruebas cubren además una sola pulsación por campaña, el salto sin inventar
respuestas, la anotación de cambios en la referencia sin repetir su presencia
y un cambio guardado como borrador sin clasificar automáticamente esa referencia.
También se comprueba el avance por dos emplazamientos consecutivos, conservación
del zoom, conversión de un cursor antiguo de 2023 a un histórico pendiente,
referencia falsa con presencia anterior, ausencia en ambas con un solo clic,
deshacer ambas observaciones juntas y rollback completo si falla el segundo
guardado de una comparación.
La prueba con una copia real de la depuradora comprueba que se distinguen la
geometría antigua y los paneles en suelo, que pulsar el marco cambia de caso sin
traspasar observaciones y que ambas campañas muestran el mismo localizador.
También se comprueba que arrastrar no selecciona vecinos y que el objetivo sigue
visible con las capas desactivadas.
También se comprobó reanudar un contexto tras un corte
de descarga y agrupar componentes que atraviesan el borde entre dos teselas.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
