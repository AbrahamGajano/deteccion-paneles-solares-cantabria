# Documentación del proyecto

| Documento | Contenido |
| --- | --- |
| [Historial del proyecto](historial_proyecto.tex) | Etapas del trabajo, decisiones, comparaciones entre modelos y resultados intermedios. |
| [Memoria técnica](memoria_tecnica.tex) | Datos, segmentación, seguimiento, ajuste LiDAR y cálculo de potencia y energía. |
| [Análisis de resultados](analisis_resultados.tex) | Distribución territorial, evolución, calidad de la detección, ángulos LiDAR, curvas de producción y uso del panel. |
| [Guía práctica](guia.md) | Comandos y formatos para las tareas de validación, etiquetado y seguimiento. |
| [LiDAR](lidar/README.md) | Descarga, recortes, visores y [anexo metodológico](lidar/memoria_lidar.pdf). |
| [Revisión cartográfica de 2023](revision_catastro_2023.md) | Selección de edificaciones y parcelas, atribución y comprobaciones. |

`ejemplos/` contiene las entradas pequeñas de la guía. `figuras/` reúne los
mapas y ejemplos visuales del proyecto. La memoria técnica y el anexo LiDAR
comparten la sección de cálculo en `lidar/metodo.tex`.

## Consultar los resultados

| Qué quieres ver | Dónde abrirlo |
| --- | --- |
| **Panel de indicadores** | [panel_indicadores.html](../runs/visualizacion/panel_indicadores.html). En **Mapa y resumen**, la campaña de 2023 muestra todas las máscaras de la pasada regional. Las campañas anteriores permiten consultar las fichas del seguimiento. Activa la ortofoto en el selector del mapa. |
| **Figuras del informe** | [figuras/](figuras/). Incluye el mapa regional, las capturas de Piélagos, la superficie por uso y por campaña, los ángulos y la producción mensual y horaria. |
| **Mapa regional y datos municipales** | [indicadores/unet_actual/](../runs/indicadores/unet_actual/). Abre `detecciones.gpkg` o `municipios.gpkg` en QGIS. `municipios.csv` recoge área detectada y cobertura procesada por municipio. |
| **Seguimiento y estadísticas del panel** | [resumen_indicadores.png](../runs/visualizacion/resumen_indicadores.png), [resumen_municipal.csv](../runs/visualizacion/resumen_municipal.csv) y [atribucion.gpkg](../runs/indicadores/seguimiento_historico/atribucion.gpkg). |
| **Errores del detector** | [Test local de U-Net](../runs/evaluacion/unet_actual/) y [prueba de ortofotos de 20 cm](../runs/evaluacion/validacion_20cm/). Contienen métricas, resultados por tesela y ejemplos de falsos positivos y omisiones. |
| **LiDAR por detección** | [Visor de resultados LiDAR](lidar/README.md), [resumen del ajuste](../runs/lidar/cantabria_20261112_v2/resumen.csv) y [ángulos usados en energía](../runs/visualizacion/factores_energia.csv). |
| **Producción con clima medio** | [Resumen anual](../runs/visualizacion/produccion_estimada.csv) y [detalle por detección](../runs/visualizacion/energia_paneles.csv). |
| **Producción con la meteorología de 2023** | [Series horarias y resúmenes](../runs/visualizacion/series_temporales/inventario_2023_meteo_2023/). Ejemplo de consulta para la nave de [Piélagos](../runs/visualizacion/consulta_pielagos/produccion.html). |

El panel y los datos de `runs/` son salidas locales. Para consultarlos desde otro
equipo, el [README principal](../README.md) describe los archivos de entrega y
los comandos para regenerar el panel y las series.

## Compilar los informes

Con una instalación de LaTeX, abre PowerShell en la raíz del repositorio y ejecuta estos comandos.

```powershell
New-Item -ItemType Directory -Force outputs/documentacion | Out-Null
Push-Location docs
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/documentacion historial_proyecto.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/documentacion memoria_tecnica.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/documentacion analisis_resultados.tex
Pop-Location
```

Ejecuta dos veces cada comando para actualizar el índice y las referencias.
El anexo se compila del mismo modo con `lidar/memoria_lidar.tex`.
Con las salidas de 2023 disponibles, `python docs/generar_figuras_resultados.py`
regenera las gráficas y el mapa regional del análisis. Las imágenes
`panel_pielagos_2020.jpg` y `panel_pielagos_2023.jpg` son capturas del panel
con la ortofoto activada y la agrupación `fv_832b218af089554de012` seleccionada.
