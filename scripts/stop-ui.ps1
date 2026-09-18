[CmdletBinding()]
param(
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$TaskName = 'XPSDiscoveryAgentUI'
$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Task) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Write-Host "Stopped scheduled task $TaskName."
} else {
    Write-Host "Scheduled task $TaskName was not found."
}

# Stop only a remaining listener whose command line belongs to this dashboard.
$Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($Entry in $Listener) {
    $Process = Get-CimInstance Win32_Process -Filter "ProcessId=$($Entry.OwningProcess)"
    if ($Process.CommandLine -like '*streamlit*' -and ($Process.CommandLine -like '*XPS-agent*streamlit_app.py*' -or $Process.CommandLine -like '*XPS-agent*xps_agent*dashboard.py*')) {
        Stop-Process -Id $Entry.OwningProcess
        Write-Host "Stopped remaining dashboard process $($Entry.OwningProcess)."
    }
}
