[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path $ProjectRoot '.env'
$ExamplePath = Join-Path $ProjectRoot '.env.example'

if (-not (Test-Path -LiteralPath $EnvPath)) {
    Copy-Item -LiteralPath $ExamplePath -Destination $EnvPath
}

function Set-DotEnvValue {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return }
    if ($Value.Contains("`r") -or $Value.Contains("`n")) {
        throw "$Name contains a newline and was rejected."
    }
    $Lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $EnvPath)
    $Found = $false
    for ($Index = 0; $Index -lt $Lines.Count; $Index++) {
        if ($Lines[$Index] -match ('^' + [regex]::Escape($Name) + '=')) {
            $Lines[$Index] = "$Name=$Value"
            $Found = $true
            break
        }
    }
    if (-not $Found) { $Lines.Add("$Name=$Value") }
    [IO.File]::WriteAllLines($EnvPath, $Lines, [Text.UTF8Encoding]::new($false))
    Write-Host "$Name updated (value hidden)."
}

function Read-SecretValue {
    param([string]$Prompt)
    $Secure = Read-Host $Prompt -AsSecureString
    $Pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Pointer)
    }
}

Write-Host 'Press Enter for any key you want to leave unchanged.'
Set-DotEnvValue 'SILICONFLOW_API_KEY' (Read-SecretValue 'SiliconFlow API key')
Set-DotEnvValue 'OPENALEX_API_KEY' (Read-SecretValue 'OpenAlex API key')
Set-DotEnvValue 'ELSEVIER_API_KEY' (Read-SecretValue 'Elsevier API key')
Set-DotEnvValue 'ELSEVIER_INSTTOKEN' (Read-SecretValue 'Elsevier institutional token (optional)')

$Mail = Read-Host 'OpenAlex contact email (optional; visible input)'
Set-DotEnvValue 'OPENALEX_MAILTO' $Mail
Write-Host 'Configuration saved. Run .\.venv\Scripts\xps-agent.exe doctor to verify statuses.'

