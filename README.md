# Paneles solares en Cantabria

Detección de superficies fotovoltaicas en ortofotos PNOA 2023, evaluación de
U-Net frente a YOLO e indicadores por edificio y municipio. La potencia y la
producción son estimaciones: no se conocen la potencia instalada ni la energía
observada de cada cubierta.

## Entorno y datos

Python 3.10 o posterior. Desde la raíz del repositorio:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Se esperan ortofotos `.tif` en `data/pnoa/`. Los scripts de geografía descargan
Catastro y límites municipales a `data/geografia/` y crean
`data/manifests/teselas.csv`. Los JSON de LabelMe en
`data/pool/annotations/` son las anotaciones originales; los TXT de YOLO y las
máscaras de U-Net se regeneran desde ellos. `weights/unet_resnet34.pt` es el
modelo U-Net actual. YOLO usa el `best.pt` de su entrenamiento.

`data/`, `runs/`, `outputs/`, pesos, entornos y cachés de Python no se versionan.
Las descargas, el etiquetado, entrenamientos, evaluaciones e inferencia completa
pueden tardar bastante.

## Flujo recomendado

Prepara la geografía una vez, o cuando cambien los datos originales:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.geografia.catastro
.\.venv\Scripts\python.exe -m paneles_solares.geografia.municipios
.\.venv\Scripts\python.exe -m paneles_solares.geografia.teselas
```

Para ampliar el conjunto etiquetado, ejecuta los dos pasos alrededor de la
anotación manual en LabelMe, clase `panel_solar`. Una imagen sin paneles
necesita también su JSON vacío. Después decide train/val/test y reconstruye
ambos datasets:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.crear_labeling
# Etiquetar manualmente los PNG en data/labeling/images/
.\.venv\Scripts\python.exe -m paneles_solares.etiquetado.incorporar_labeling
.\.venv\Scripts\python.exe -m paneles_solares.datos.decidir_dataset
.\.venv\Scripts\python.exe -m paneles_solares.datos.sincronizar
```

Entrena y evalúa ambos modelos por separado. U-Net es la fuente actual de los
indicadores; YOLO se conserva para la comparación experimental:

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.entrenar_unet
.\.venv\Scripts\python.exe -m paneles_solares.modelos.evaluar_unet
.\.venv\Scripts\python.exe -m paneles_solares.modelos.entrenar_yolo
.\.venv\Scripts\python.exe -m paneles_solares.modelos.evaluar_yolo
```

Los pesos U-Net se guardan en `weights/unet_resnet34.pt`; su evaluación en
`runs/evaluacion/unet_actual/`. La de YOLO queda en
`runs/evaluacion/cantabria/yolo11s/revision/`. El entrenamiento puede
sobrescribir sus resultados anteriores: revisa las constantes de cada script
antes de lanzarlo.

Calcula los indicadores georreferenciados y, después, la producción. La segunda
etapa consulta PVGIS 5.3 secuencialmente, unas 200 llamadas para los 100
municipios con detecciones actuales. La caché fija `pvgis_municipios.jsonl`
permite continuar sin repetir municipios terminados.

`indicadores_unet.py` realiza la inferencia sobre Cantabria y guarda
`edificios.gpkg` y `municipios.gpkg`. `estimar_energia.py` lee esos GeoPackage y
genera los CSV de producción. `panel_indicadores.py` lee `municipios.gpkg`,
`edificios.gpkg` y el resumen municipal de producción; no repite la inferencia
ni consulta PVGIS.

```powershell
.\.venv\Scripts\python.exe -m paneles_solares.modelos.indicadores_unet
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.estimar_energia
.\.venv\Scripts\python.exe -m paneles_solares.visualizacion.panel_indicadores
```

Prueba limitada de PVGIS, sin escribir resultados completos:

```powershell
.\.venv\Scripts\python.exe -c "from paneles_solares.visualizacion.estimar_energia import prueba; prueba()"
```

El panel se puede regenerar sin repetir inferencia ni consultas PVGIS. El
selector permite ver la producción anual y la potencia pico estimadas junto a
los indicadores de detección. La evaluación U-Net guarda imágenes con errores
del test en `runs/evaluacion/unet_actual/`.

## Salidas y supuestos

`runs/indicadores/unet_actual/` contiene `edificios.gpkg`, `municipios.gpkg`,
`municipios.csv`, `panel_indicadores.html` y sus figuras. La producción añade
`produccion_municipal.csv`, `produccion_mensual.csv`,
`produccion_horaria_representativa.csv`, `produccion_config.json` y
`resumen_produccion.png`. El CSV horario contiene 8.760 horas por municipio;
el HTML utiliza solo resúmenes.

La U-Net usa RGB de 512 × 512, normalización ImageNet y umbral 0,5, como su
evaluación actual (Dice 0,856; sesgo global de superficie −7,54 %). La
superficie detectada es **planimétrica**. La superficie corregida del test es
`superficie_detectada / 0.9246`: una calibración experimental que no sustituye
el valor original ni garantiza precisión por edificio.

La potencia pico estimada es superficie × densidad efectiva: conservador,
superficie detectada × 0,17 kWp/m²; central, superficie corregida × 0,20;
superior, superficie corregida × 0,23. PVGIS usa 30° de inclinación,
orientación 0° (sur) y pérdidas del 14 %. No se conocen aún inclinación,
orientación o sombras reales de cada cubierta, y no se corrige la superficie
por inclinación.

`PVcalc` aporta rendimientos mensuales medios en kWh/kWp. `seriescalc` usa 2019
para dar forma a un perfil horario representativo UTC; cada mes se normaliza a
la media mensual. El perfil no es una medición de producción de 2019. La
[documentación oficial de PVGIS 5.3](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en)
describe los endpoints, parámetros y unidades.

La [memoria técnica](docs/memoria_tecnica.pdf) presenta los resultados, figuras
y limitaciones. Su fuente editable está en `docs/memoria_tecnica.tex`; desde
`docs/` se puede compilar con `latexmk -pdf memoria_tecnica.tex`. Los dos mapas
estáticos se regeneran con:

```powershell
.\.venv\Scripts\python.exe docs/generar_mapas.py
```

## Código

La [validación de otras resoluciones](docs/validacion_resoluciones.md) prepara una
muestra pequeña del test de 2023, permite etiquetarla con LabelMe y evalúa el
modelo congelado sobre otro producto nativo. Su configuración está en
`validaciones.json`; no requiere descarga regional ni reentrenamiento.

- `geografia/`: Catastro, municipios, ortofotos e índice de teselas.
- `etiquetado/` y `datos/`: selección, anotaciones, manifiestos y datasets.
- `modelos/unet.py`: arquitectura, preprocesamiento y carga compartidos.
- `modelos/`: entrenamiento, evaluación e indicadores U-Net y YOLO.
- `visualizacion/`: energía y panel central.
