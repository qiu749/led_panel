Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$publishRoot = Join-Path $root "csharp\LedInspectionStation\bin\Release\net8.0-windows\win-x64\publish"
$backendDist = Join-Path $root "dist\LedInspectionBackend"
$stageRoot = Join-Path $root "release\LedInspectionStation_正式版目录"
$zipPath = Join-Path $root "release\LedInspectionStation_正式版目录.zip"

Write-Host "Build C# frontend..."
dotnet publish .\csharp\LedInspectionStation\LedInspectionStation.csproj -c Release -r win-x64 --self-contained true

Write-Host "Build Python backend..."
python -m PyInstaller -y .\LedInspectionBackend.spec

Write-Host "Stage green package..."
if (Test-Path $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
Copy-Item -LiteralPath "$publishRoot\*" -Destination $stageRoot -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $stageRoot "backend") | Out-Null
Copy-Item -LiteralPath "$backendDist\*" -Destination (Join-Path $stageRoot "backend") -Recurse -Force

$readme = Get-ChildItem -Path ".\packaging" -Filter "README_*.txt" | Select-Object -First 1
if ($null -ne $readme) {
    $readmeDest = Join-Path $stageRoot $readme.Name
    if (Test-Path $readmeDest) {
        Remove-Item -LiteralPath $readmeDest -Force
    }
    Copy-Item -LiteralPath $readme.FullName -Destination $readmeDest -Force
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path $stageRoot -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host 'Done:'
Write-Host $stageRoot
Write-Host $zipPath
