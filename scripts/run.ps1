[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AgentArguments
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot '.venv\Scripts\xps-agent.exe'
if (-not (Test-Path -LiteralPath $Executable)) {
    throw 'Environment is missing. Run scripts\setup.ps1 first.'
}
Set-Location $ProjectRoot
& $Executable @AgentArguments

