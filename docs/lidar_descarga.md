# Descarga y recorte LiDAR

Desde la raíz del repositorio, procesar todas las hojas pendientes:

```powershell
.\.venv\Scripts\python.exe -u -m paneles_solares.lidar.descargar_recortar --limite 0 --reintentos 3 --reserva-gb 5
```

El programa descarga una hoja cada vez, extrae su LAZ y guarda sus recortes.
Los ZIP y LAZ originales se eliminan al terminar cada intento, también si falla.
Los recortes quedan en `data/lidar/recortes_parciales/`, separados por hoja.
Si un panel cruza dos hojas, conserva sus dos aportaciones por separado.

Cada hoja terminada tiene un `resumen.json`. Al repetir el comando, se comprueba
el plan y el número de puntos declarado en las cabeceras de los recortes para
omitir las hojas completas. Las incompletas se vuelven a procesar.

## Fallos y reanudación

- Los errores temporales de red se intentan hasta tres veces, con diez segundos
  entre intentos. Un 404 se registra sin insistir en esa misma ejecución.
- Si una hoja sigue fallando, se registra el motivo y se pasa a la siguiente.
- Si el disco alcanza la reserva de 5 GiB, el lote se detiene. Se comprueba el
  espacio durante la descarga, antes de extraer y durante el recorte.
- Para interrumpir manualmente, pulsar `Ctrl+C` y esperar a que termine la
  limpieza. Después se puede ejecutar el mismo comando para continuar.
- Un apagado forzado puede dejar archivos incompletos; una hoja sin resumen
  válido se vuelve a descargar y procesar al reiniciar.

Consultar el avance desde otra terminal:

```powershell
Get-Content runs/lidar/descarga/descarga.log -Tail 25
```

Consultar las hojas que siguen dando error:

```powershell
Import-Csv runs/lidar/descarga/errores.csv | Format-Table -Wrap
```

El registro se conserva entre ejecuciones. Una hoja recuperada se retira de
`errores.csv`. Las hojas posteriores a una parada todavía no han sido intentadas.

## Dejar el ordenador trabajando

Mantener el equipo enchufado, la terminal abierta y la conexión activa. En
Windows, el proceso solicita que no se suspenda por inactividad mientras trabaja;
restablece ese estado al terminar. Esto no evita un apagado, un reinicio o la
suspensión provocada al cerrar la tapa.

El espacio final depende de la densidad y la extensión de los recortes. La
reserva permite detener el lote antes de agotar el disco, pero no garantiza que
las 914 hojas terminen ni que todos los enlaces estén disponibles.
