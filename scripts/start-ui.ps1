[CmdletBinding()]
param(
    [ValidateRange(1024,65535)][int]$Port = 8501,
    [string]$Address = '127.0.0.1',
    [switch]$Scheduled
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$TaskScript = Join-Path $ProjectRoot 'scripts\run-ui-task.ps1'
$TaskName = 'XPSDiscoveryAgentUI'
$Dashboard = Join-Path $ProjectRoot 'streamlit_app.py'

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Python environment is missing. Run scripts\setup.ps1 first.'
}

try {
    $Response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/_stcore/health" -TimeoutSec 2
    if ($Response.StatusCode -eq 200) {
        Write-Host "Streamlit is already running. Open http://127.0.0.1:$Port"
        return
    }
} catch { }

function Start-ForegroundUI {
    $ErrorActionPreference = 'Continue' # Native startup messages on stderr are not fatal in Windows PowerShell.
    Set-Location $ProjectRoot
    $env:XPS_PROJECT_ROOT = $ProjectRoot
    Write-Host "Open http://127.0.0.1:$Port; keep this terminal open. Ctrl+C stops the UI."
    & $Python -m streamlit run $Dashboard --server.address $Address --server.port $Port --server.headless true --browser.gatherUsageStats false
}

if (-not $Scheduled) {
    # Normal Anaconda Prompt/PowerShell startup requires no task-registration privileges.
    Start-ForegroundUI
    exit $LASTEXITCODE
}

$Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$TaskScript`" -Port $Port -Address $Address"
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $Arguments
$UserId = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
$TaskSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

try {
    Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $TaskSettings `
    -Description 'Persistent Streamlit UI for the XPS Discovery Agent' `
        -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
} catch {
    Write-Warning 'Scheduled-task access was denied. Falling back to foreground Streamlit (no administrator required).'
    Start-ForegroundUI
    exit $LASTEXITCODE
}

$Healthy = $false
for ($Attempt = 0; $Attempt -lt 15; $Attempt++) {
    Start-Sleep -Seconds 1
    try {
        $Response = Invoke-WebRequest `
            -UseBasicParsing `
            -Uri "http://127.0.0.1:$Port/_stcore/health" `
            -TimeoutSec 2
        if ($Response.StatusCode -eq 200) {
            $Healthy = $true
            break
        }
    }
    catch { }
}

if (-not $Healthy) {
    $State = (Get-ScheduledTask -TaskName $TaskName).State
    throw "Scheduled task was created but UI health failed (task state: $State)."
}
Write-Host 'Streamlit scheduled task is running (health 200).'
Write-Host "Open http://127.0.0.1:$Port"
