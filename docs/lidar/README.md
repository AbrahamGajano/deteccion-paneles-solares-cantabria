# LiDAR: inclinación de las detecciones de 2023

El objetivo es estimar la inclinación de paneles compatibles con su cubierta.
La descarga, los cálculos y el visor se documentan aquí, aparte de la memoria
de detección y energía. La versión LaTeX está en [memoria_lidar.tex](memoria_lidar.tex).

## Método

1. Leer los puntos dentro del polígono detectado y ajustar un plano con RANSAC.
2. Comprobar su apoyo, distribución espacial y estabilidad al variar la muestra.
3. Buscar hasta **seis planos exteriores**, retirando los puntos de cada uno
   antes de buscar el siguiente. Parar si ya no hay suficientes puntos de apoyo.
4. Separar los planos en fragmentos y localizar los próximos al interior en 3D.
5. Elegir el fragmento con más puntos vecinos al interior. Si hay superficies
   distintas, exigir al menos el doble de vecinos que las alternativas; en otro
   caso, conservar la ambigüedad.
6. Si falta un fragmento próximo, ampliar la franja de **2 a 4 y hasta 8 m**.
7. Comparar estabilidad, orientación y separación del plano interior y exterior.

| Parámetro | Valor inicial |
|---|---:|
| Distancia perpendicular de inlier | 0,10 m |
| Apoyo interior mínimo | 75 % |
| Inliers mínimos | 10 |
| Dispersión mínima en la dirección horizontal menor | 0,25 m |
| Máximo de planos exteriores por franja | 6 |
| Diferencia angular interior/exterior | 5° |
| Separación máxima, percentil 90 | 0,30 m |
| Variación angular en las pruebas de estabilidad | 3° |

El porcentaje usa todos los puntos interiores: los outliers siguen contando.
Los valores son configurables y experimentales. Se guardan inclinación candidata,
estimada, utilizada y motivo. Sin evidencia suficiente, la estimada queda vacía
y la utilizada toma el supuesto configurable de 30°.

## Naves y cubiertas complejas

Se busca el faldón local, sin reconstruir todo el edificio. Las cubiertas grandes
y de poca pendiente también se procesan; su apariencia en la ortofoto no determina
la inclinación. Una detección que abarca varios faldones puede quedar ambigua.

Si casi todo el faldón está cubierto de paneles, puede faltar cubierta visible
para contrastar. La continuidad de una superficie es evidencia geométrica, pero
no confirma qué objeto produjo los retornos. Un plano equivocado también puede
ser estable. Hace falta revisar naves y edificios distintos para validar el método.

## Ejecución

Descargar las hojas necesarias y conservar sus recortes:

```powershell
.\.venv\Scripts\python.exe -u -m paneles_solares.lidar.descargar_recortar --limite 0 --reserva-gb 5
```

Procesar las 8.671 detecciones disponibles, sin imágenes ni CSV del contexto:

```powershell
.\.venv\Scripts\python.exe -u -m paneles_solares.lidar.estimar_planos --cantabria --solo-resumen --reanudar --salida runs/lidar/cantabria_20261112_v2
```

Consultar el lote en segundo plano:

```powershell
Get-Content runs/lidar/cantabria_20261112_v2/progreso.json
Get-Content runs/lidar/cantabria_20261112_v2/ejecucion.log -Encoding utf8 -Tail 15
```

Abrir los resultados de toda Cantabria:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.lidar.visor_resultados
```

El visor está en http://127.0.0.1:8052. Permite filtrar estimaciones o valores
predeterminados y saltar a un caso aleatorio. Las imágenes se extraen al abrir
cada caso y las aportaciones LAZ se reúnen en memoria. Los huecos de ortofoto se
muestran en negro; si no hay cobertura, se conserva la vista LiDAR.

La muestra anterior sigue disponible con `--muestra --resultados runs/lidar/validacion_100_v2`.
El lote regional guarda `resumen.csv`,
`errores.csv` y un `resumen.json` por detección. La reanudación reutiliza JSON
legibles y no vacíos. Para comparar configuraciones se emplean salidas distintas.

Las salidas de las pruebas sustituidas se pueden retirar. En
los mismos cien casos, la búsqueda anterior aceptaba 13 inclinaciones y la
actual acepta 21; esto mide aprovechamiento, no exactitud. Se conservan la
muestra original, sus formularios y `validacion_100_v2`. El lote regional del
método anterior quedó incompleto y se sustituyó por la ejecución `v2`.

## Archivos del código

- `descargar_recortar.py`: descarga una hoja, recorta y limpia los originales.
- `preparar_revision.py`: prepara imágenes y nubes de una muestra para el visor.
- `ajuste_planos.py`: ajuste, búsqueda exterior y decisión geométrica.
- `estimar_planos.py`: lectura, procesamiento del lote y resultados.
- `visor_resultados.py`: ortofoto, nube, planos y medidas.
