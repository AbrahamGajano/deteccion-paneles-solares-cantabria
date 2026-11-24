[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'
$raizProyecto = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path

$destinos = @(
    foreach ($carpeta in @('src', 'tests', 'docs')) {
        Get-ChildItem -LiteralPath (Join-Path $raizProyecto $carpeta) -Recurse -Directory |
            Where-Object { $_.Name -eq '__pycache__' }
    }
    foreach ($cache in @('.ruff_cache', '.pytest_cache')) {
        $destino = Join-Path $raizProyecto $cache
        if (Test-Path -LiteralPath $destino) { Get-Item -LiteralPath $destino -Force }
    }
    $extensiones = @('aux', 'log', 'out', 'xdv', 'synctex.gz', 'fdb_latexmk', 'fls', 'toc')
    foreach ($tex in Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File -Filter '*.tex') {
        foreach ($extension in $extensiones) {
            $destino = Join-Path $tex.DirectoryName "$($tex.BaseName).$extension"
            if (Test-Path -LiteralPath $destino -PathType Leaf) { Get-Item -LiteralPath $destino }
        }
    }
)

foreach ($destino in $destinos) {
    $resuelta = (Resolve-Path -LiteralPath $destino.FullName).Path
    if (
        -not $resuelta.StartsWith($raizProyecto + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        ($destino.Attributes -band [IO.FileAttributes]::ReparsePoint)
    ) {
        throw "Destino no válido: $resuelta"
    }
}

foreach ($destino in $destinos) {
    if ($PSCmdlet.ShouldProcess($destino.FullName, 'Retirar caché o auxiliar de compilación')) {
        Remove-Item -LiteralPath $destino.FullName -Recurse -Force
    }
}
