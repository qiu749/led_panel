$ErrorActionPreference = "Stop"

$packageName = "LedInspectionStation_green_selfcontained.zip"
$appName = "LedInspectionStation"
$installDir = Join-Path $env:LOCALAPPDATA $appName
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "LedInspectionStation.lnk"
$sourceZip = Join-Path $PSScriptRoot $packageName

if (-not (Test-Path $sourceZip)) {
    throw "Package zip not found: $sourceZip"
}

if (Test-Path $installDir) {
    Remove-Item -LiteralPath $installDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Expand-Archive -LiteralPath $sourceZip -DestinationPath $installDir -Force

$appDir = Join-Path $installDir "LedInspectionStation_green_selfcontained"
$exePath = Join-Path $appDir "LedInspectionStation.exe"
if (-not (Test-Path $exePath)) {
    throw "Installed exe not found: $exePath"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $exePath
$shortcut.WorkingDirectory = $appDir
$shortcut.WindowStyle = 1
$shortcut.Description = "Led Inspection Station"
$shortcut.Save()

Start-Process -FilePath $exePath -WorkingDirectory $appDir
