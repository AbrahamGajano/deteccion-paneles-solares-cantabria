# Documentación del proyecto

| Documento | Contenido |
| --- | --- |
| [Historial del proyecto](historial_proyecto.tex) | Etapas del trabajo, decisiones, comparaciones entre modelos y resultados intermedios. |
| [Memoria técnica](memoria_tecnica.tex) | Datos, segmentación, seguimiento, ajuste LiDAR y cálculo de potencia y energía. |
| [Guía práctica](guia.md) | Comandos y formatos para las tareas de validación, etiquetado y seguimiento. |
| [LiDAR](lidar/README.md) | Descarga, recortes, visores y [anexo metodológico](lidar/memoria_lidar.pdf). |
| [Revisión cartográfica de 2023](revision_catastro_2023.md) | Selección de edificaciones y parcelas, atribución y comprobaciones. |

`ejemplos/` contiene las entradas pequeñas de la guía. `figuras/` reúne los
mapas y ejemplos visuales del proyecto. La memoria técnica y el anexo LiDAR
comparten la sección de cálculo en `lidar/metodo.tex`.

## Compilar los informes

Con una instalación de LaTeX, abre PowerShell en la raíz del repositorio y ejecuta estos comandos.

```powershell
New-Item -ItemType Directory -Force outputs/documentacion | Out-Null
Push-Location docs
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/documentacion historial_proyecto.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../outputs/documentacion memoria_tecnica.tex
Pop-Location
```

Ejecuta dos veces cada comando para actualizar el índice y las referencias.
El anexo se compila del mismo modo con `lidar/memoria_lidar.tex`.
