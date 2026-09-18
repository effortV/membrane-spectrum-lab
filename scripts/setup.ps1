[CmdletBinding()]
param(
    [switch]$SkipDev
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$UvDir = Join-Path $ProjectRoot '.tools\uv'
$UvExe = Join-Path $UvDir 'uv.exe'

if (-not (Test-Path -LiteralPath $UvExe)) {
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    $Archive = Join-Path ([IO.Path]::GetTempPath()) 'xps-agent-uv.zip'
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest `
        -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' `
        -OutFile $Archive
    Expand-Archive -LiteralPath $Archive -DestinationPath $UvDir -Force
}

Set-Location $ProjectRoot
& $UvExe python install 3.12
if ($SkipDev) {
    & $UvExe sync
} else {
    & $UvExe sync --extra dev
}

Write-Host "Environment ready: $ProjectRoot\.venv"
Write-Host "Next: Copy-Item .env.example .env; then edit .env locally."
Write-Host "Check: .\.venv\Scripts\xps-agent.exe doctor"

