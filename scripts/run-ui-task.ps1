[CmdletBinding()]
param(
    [int]$Port = 8501,
    [string]$Address = '127.0.0.1'
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Service = Join-Path $ProjectRoot 'scripts\serve-ui.py'
Set-Location $ProjectRoot
$env:XPS_PROJECT_ROOT = $ProjectRoot

& $Python $Service --address $Address --port $Port
exit $LASTEXITCODE
