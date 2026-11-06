# Inclinación de paneles mediante LiDAR

## Objetivo y alcance

Estimar la inclinación de las detecciones cuyo plano sea compatible con una
cubierta local coplanar o aproximadamente paralela. Es un prototipo:
compatibilidad geométrica no significa montaje confirmado.

Las estructuras inclinadas respecto al tejado quedan fuera del objetivo. Cuando
falta evidencia se utiliza una inclinación predeterminada configurable, separada
de la estimada. Los 30° iniciales son una hipótesis coherente con el escenario
central del proyecto, no una medida ni un óptimo demostrado para cada instalación.

## Archivos y recorrido de lectura

| Archivo | Responsabilidad | Librerías |
|---|---|---|
| `src/paneles_solares/lidar/ajuste_planos.py` | RANSAC, calidad, estabilidad, búsqueda exterior y decisión | NumPy y Shapely |
| `src/paneles_solares/lidar/estimar_planos.py` | Lectura de casos y LAZ, informes y figuras | pandas, laspy, Shapely y Plotly |
| `tests/test_estimar_planos.py` | Escenas de geometría conocida y guardado de resultados | unittest y las librerías anteriores |

No se añaden dependencias. Para entender el recorrido, empezar por
`decidir_inclinacion()`, después `buscar_plano()` y finalmente `buscar_cubierta()`.

## Procedimiento implementado

1. **Interior.** Seleccionar los puntos del polígono original. El margen del
   recorte aporta contexto, no amplía la detección. RANSAC prueba ternas, elige
   el consenso mayor y reajusta por SVD. Las distancias son perpendiculares.
2. **Calidad.** Exigir porcentaje mínimo sobre todos los puntos interiores,
   cantidad mínima de inliers, extensión en dos direcciones horizontales e
   inclinación dentro del alcance. Los puntos alejados no deforman el ajuste,
   pero siguen en el denominador y en las métricas generales.
3. **Estabilidad.** Repetir con tolerancias de 0,8 y 1,2 veces la original y con
   una submuestra del 80 %, usando semillas fijas. Comparar cada normal y la
   distancia del centro original al nuevo plano. No se escoge la repetición
   que dé el resultado más favorable.
4. **Exterior independiente.** Buscar hasta dos planos en una franja exterior,
   retirando los inliers del primero para buscar el segundo. Esta función no
   conoce la normal interior. Separar los planos en parches mediante celdas XY
   ocupadas y sus ocho vecinas, y comprobar su calidad.
5. **Asociación local.** Exigir al menos tres puntos del parche con algún vecino
   interior próximo en 3D. No hace falta apoyo por todos los lados. Si hay
   superficies próximas distintas, declarar ambigüedad. Parches con normales
   a no más de 1° y p90 de distancia al plano de referencia no mayor que el
   umbral exterior se consideran una misma superficie; se usa el de más puntos
   sin fusionarlos.
6. **Ampliación limitada.** Si no hay candidato asociado, duplicar la franja
   hasta el máximo: por defecto, 2 y 4 m. Si se conoce la ventana del recorte,
   la franja debe caber entera. Una asociación ambigua, inestable o incompatible
   no se rescata ampliando sucesivamente la búsqueda.
7. **Comparación.** Comprobar estabilidad exterior, ángulo entre normales y p90
   de distancia de los inliers interiores al plano exterior. Así se admite una
   pequeña separación entre planos paralelos. La inclinación aceptada procede
   del interior, sin mezclar puntos del techo.

`evaluar_entorno()` conserva el diagnóstico anterior: puntos exteriores próximos
a la prolongación del plano interior. No es el ajuste independiente del tejado.
No se exige un porcentaje alto sobre todos los puntos del anillo exterior.

## Valores iniciales pendientes de calibración

| Opción | Valor | Significado |
|---|---:|---|
| `--umbral` | 0.10 m | Distancia perpendicular interior para ser inlier |
| `--umbral-exterior` | Igual al interior | Tolerancia del ajuste exterior |
| `--porcentaje-minimo` | 80 % | Apoyo sobre todos los puntos interiores |
| `--min-puntos` | 10 | Inliers mínimos por ajuste o parche |
| `--dispersion-minima` | 0.25 m | Desviación espacial en el eje XY principal menor |
| `--inclinacion-maxima` | 75° | Alcance para planos interiores y exteriores |
| `--angulo-maximo` | 5° | Diferencia máxima entre normales |
| `--separacion-maxima` | 0.30 m | p90 máximo de distancia interior al plano exterior |
| `--estabilidad-maxima` | 3° | Variación angular máxima en las repeticiones |
| `--anillo` / `--anillo-maximo` | 2 / 4 m | Anchuras inicial y máxima de búsqueda |
| `--distancia-vecinos` | 1 m | Proximidad 3D y lado de las celdas de conectividad |
| `--inclinacion-predeterminada` | 30° | Valor supuesto cuando se rechaza la estimación |
| `--iteraciones` / `--semilla` | 500 / 42 | Muestreo reproducible de RANSAC |

La estabilidad también exige un cambio de posición no mayor que dos veces la
tolerancia del ajuste. La dispersión mínima es una desviación espacial, no la
anchura total. La conectividad por celdas es aproximada y depende de su tamaño:
no demuestra que los puntos pertenezcan al mismo edificio.

## Ejecutar

Primer caso, con HTML:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.lidar.estimar_planos
```

Ejemplo que supera los criterios iniciales en la muestra actual:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.lidar.estimar_planos --panel componente_f6eec0b4a077b18d3bea
```

Todos los anotados, sin generar un HTML por caso:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.lidar.estimar_planos --todos --revisados --sin-html --salida runs/lidar/validacion_cubierta
```

`--revisados` solo selecciona formularios no vacíos; sus etiquetas no deciden
la aceptación. `--todos` sin ese filtro procesa toda la muestra. `--carpeta`
permite seleccionar otro conjunto con `casos.csv` y sus LAZ. Para comparar
configuraciones, usar salidas distintas.

## Resultados y procedencia

| Campo | Interpretación |
|---|---|
| `inclinacion_candidata` | ángulo RANSAC aunque el caso sea rechazado; vacío sin plano |
| `inclinacion_estimada` | ángulo aceptado; vacío en los demás casos |
| `inclinacion_utilizada` | Estimación aceptada o valor predeterminado |
| `origen` | `estimada` o `predeterminada` |
| `motivo` | Primera condición que falla, o `compatible_con_cubierta` |

Por caso se guardan `resumen.json`, `puntos.csv`, `cubierta.csv` y, si se solicita,
`ajuste.html`. El JSON conserva criterios, semillas, planos, calidad, estabilidad,
franjas probadas y comparación. `puntos.csv` contiene todos los puntos, incluso
los rechazados. `cubierta.csv` identifica el fragmento asociado, aunque la
comparación posterior lo descarte.

El HTML muestra el candidato interior en naranja y el exterior en azul, con su
fragmento resaltado. El título indica valor utilizado, procedencia y motivo.
Las superficies son extensiones matemáticas, no límites de tejado. Los percentiles
de altura limitan la vista, sin eliminar puntos de los cálculos.

La raíz contiene `resumen.csv` y `errores.csv`. La falta de evidencia produce una
fila con valor supuesto, incluso con menos de tres puntos interiores. Los fallos
de lectura o escritura se registran como errores de procesamiento y provocan
una salida con error. Las distancias desconocidas no se convierten en ceros.

Este módulo todavía no sustituye las hipótesis de inclinación del cálculo de
energía. Su integración requerirá conservar la procedencia y evaluar por separado
el efecto de los valores predeterminados.

## Fallos que todavía pueden escapar

- **Superficie equivocada dominante.** RANSAC puede elegir suelo u otra cubierta.
  Un 35 % de coincidencia visual no implica mayoría de puntos. Un plano incorrecto
  pero continuo y estable puede superar los criterios.
- **Panel no observado.** Si los retornos interiores pertenecen al techo, la
  coincidencia exterior no confirma el montaje. Una estructura independiente
  podría pasar inadvertida. No se incorpora una comprobación semántica de
  suelo/edificio ni se excluyen las detecciones vecinas del anillo.
- **Tejados complejos.** Solo se buscan dos planos exteriores. Buhardillas,
  curvaturas y muchas vertientes pueden quedar sin resolver.
- **Densidad y fechas.** Porcentaje de puntos y porcentaje de área son diferentes.
  Ortofoto y LiDAR de 2023 pueden proceder de fechas distintas.
- **Estabilidad limitada.** Las tres repeticiones miden sensibilidad; no producen
  un intervalo de confianza ni descartan un sesgo sistemático.

## Comprobación inicial y validación pendiente

La ejecución inicial procesó los 41 formularios revisados sin errores: 7 casos
superaron los criterios y 34 conservaron el valor supuesto. Los siete están
anotados como aparentemente coplanares, pero no hay una referencia independiente
para afirmar su precisión angular.

Los descartes fueron: apoyo interior insuficiente (20), cubierta ambigua (6),
apoyo casi lineal (2), orientaciones distintas (2), pocos inliers (2), interior
inestable (1) y ausencia de plano interior (1). Son resultados de esta configuración
y muestra dirigida, no tasas de acierto para toda Cantabria.

Conviene revisar aceptados y descartados, ajustar los límites con una parte de
los edificios y evaluar después en otros distintos. Para medir error angular hace
falta una referencia independiente. Aceptar más casos no demuestra una mejora.

Pruebas con geometrías conocidas:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_estimar_planos.py
```

Referencia del método: [RANSAC para planos en Open3D](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#plane-segmentation).
La implementación usa NumPy; no requiere instalar Open3D.
