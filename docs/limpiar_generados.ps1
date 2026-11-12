# Ejecutar desde cualquier carpeta con PowerShell. Solo retira salidas sustituidas.
$ErrorActionPreference = 'Stop'
$raizProyecto = Split-Path -Parent $PSScriptRoot
$rutasRetiradas = @(
    'runs/prueba_lidar_sin_espacio'
    'runs/prueba_visor_lidar'
    'runs/evaluacion/unet_anterior'
    'runs/evaluacion/cantabria/yolo11n'
    'runs/lidar/cantabria_20261112_base'
    'runs/lidar/validacion_100'
    'runs/lidar/validacion_cubierta'
    'runs/lidar/apoyo_revision'
    'runs/lidar/planos'
    'runs/lidar/prueba_cantabria'
    'runs/lidar/prueba_pueblos_20261105'
    'weights/yolo11n-seg.pt'
    'weights/yolo26n.pt'
)

$destinos = @(
    foreach ($ruta in $rutasRetiradas) {
        $destino = Join-Path $raizProyecto $ruta
        if (Test-Path -LiteralPath $destino) { Get-Item -LiteralPath $destino }
    }
    foreach ($carpeta in @('src', 'tests', 'docs')) {
        Get-ChildItem -LiteralPath (Join-Path $raizProyecto $carpeta) -Recurse -Directory |
            Where-Object { $_.Name -eq '__pycache__' }
    }
    foreach ($cache in @('.ruff_cache', '.pytest_cache')) {
        $destino = Join-Path $raizProyecto $cache
        if (Test-Path -LiteralPath $destino) { Get-Item -LiteralPath $destino -Force }
    }
)

# Comprobar todos los destinos antes de borrar el primero.
foreach ($destino in $destinos) {
    $resuelta = (Resolve-Path -LiteralPath $destino.FullName).Path
    if (-not $resuelta.StartsWith($raizProyecto + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Destino fuera del proyecto: $resuelta"
    }
}

foreach ($destino in $destinos) {
    Remove-Item -LiteralPath $destino.FullName -Recurse -Force
    Write-Output "Retirado: $($destino.FullName)"
}
