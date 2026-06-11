Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$publishRoot = Join-Path $root "csharp\LedInspectionStation\bin\Release\net8.0-windows\win-x64\publish"
$backendDist = Join-Path $root "dist\LedInspectionBackend"
$stageRoot = Join-Path $root "release\LedInspectionStation_green_selfcontained"
$zipPath = Join-Path $root "release\LedInspectionStation_green_selfcontained.zip"

function Invoke-Checked {
    param([Parameter(Mandatory=$true)][scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Build C# frontend (dot panel)..."
Invoke-Checked { dotnet publish .\csharp\LedInspectionStation\LedInspectionStation.csproj -c Release -r win-x64 --self-contained true --ignore-failed-sources }

Write-Host "Build Python backend (dot panel)..."
Invoke-Checked { python -m PyInstaller -y .\csharp_backend.spec }

Write-Host "Stage green package..."
if (Test-Path $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
Copy-Item -Path (Join-Path $publishRoot "*") -Destination $stageRoot -Recurse -Force
New-Item -ItemType Directory -Force -Path (Join-Path $stageRoot "backend") | Out-Null
Copy-Item -Path (Join-Path $backendDist "*") -Destination (Join-Path $stageRoot "backend") -Recurse -Force

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path $stageRoot -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host 'Done!'
Write-Host "Staged directory: $stageRoot"
Write-Host "Zip package: $zipPath"
