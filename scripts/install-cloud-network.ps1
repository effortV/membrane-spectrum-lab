# Install the official, checksum-pinned Windows client only.
# Enrollment/login and any paid plan remain separate user-authorized actions.
param([string]$StageRoot = 'D:\data\XPS-agent\workspace\connections\tailscale_20260918')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$version = '1.102.4'
$expectedSha256 = '80EB007E39DFEBE17299FA1A09C79A8E1D934F76E0246C0817EBE3AF675B7EF6'
$client = 'C:\Program Files\Tailscale\tailscale.exe'
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator rights required to install the network client'
}
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {throw 'This pinned installer is for AMD64 only'}
if ((Test-Path -LiteralPath $client) -or (Get-Service -Name Tailscale -ErrorAction SilentlyContinue)) {
    throw 'Tailscale already exists; inspect it instead of modifying an existing network installation'
}
$beforeRoutes = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' | ForEach-Object {'{0}|{1}|{2}' -f $_.InterfaceIndex,$_.NextHop,$_.RouteMetric})
$beforeDns = @(Get-DnsClientServerAddress -AddressFamily IPv4 | ForEach-Object {'{0}|{1}' -f $_.InterfaceIndex,($_.ServerAddresses -join ',')})
$dashboardBefore = (Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -UseBasicParsing -TimeoutSec 5).StatusCode
$stage = [IO.Path]::GetFullPath($StageRoot)
if ($stage -ne 'D:\data\XPS-agent\workspace\connections\tailscale_20260918') {
    throw 'Unexpected install stage; refusing to write outside this deployment directory'
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null
$installer = Join-Path $stage ('tailscale-setup-'+$version+'-amd64.msi')
if (-not (Test-Path -LiteralPath $installer)) {
    $url = 'https://pkgs.tailscale.com/stable/tailscale-setup-'+$version+'-amd64.msi'
    & 'C:\Windows\System32\curl.exe' --fail --location --silent --show-error --connect-timeout 10 --max-time 180 --output $installer $url
    if ($LASTEXITCODE -ne 0) {throw 'Official installer download failed; installation was not started'}
}
$actualHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
if ($actualHash -ne $expectedSha256) {throw 'Installer checksum mismatch; installation was not started'}
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(CN|O)=Tailscale Inc') {
    throw 'Installer publisher signature is not verified; installation was not started'
}
Write-Output 'Official installer checksum and Tailscale publisher signature verified.'
$install = Start-Process -FilePath 'C:\Windows\System32\msiexec.exe' -ArgumentList @('/i',('"'+$installer+'"'),'/qn','/norestart') -WindowStyle Hidden -Wait -PassThru
if ($install.ExitCode -notin @(0,3010)) {throw ('Installer failed with MSI exit code '+$install.ExitCode)}
if (-not (Test-Path -LiteralPath $client)) {throw 'Installed client was not found'}
$afterRoutes = @(Get-NetRoute -DestinationPrefix '0.0.0.0/0' | ForEach-Object {'{0}|{1}|{2}' -f $_.InterfaceIndex,$_.NextHop,$_.RouteMetric})
# A newly created, disconnected adapter is permitted; existing adapter DNS must not change.
$afterDns = @(Get-DnsClientServerAddress -AddressFamily IPv4 | ForEach-Object {'{0}|{1}' -f $_.InterfaceIndex,($_.ServerAddresses -join ',')})
$dnsRemovedOrChanged = @(Compare-Object $beforeDns $afterDns | Where-Object {$_.SideIndicator -eq '<='})
$health = (Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -UseBasicParsing -TimeoutSec 5).StatusCode
$statusRaw = @(& $client status --json 2>$null) -join "`n"
$state = 'not_checked'
if ($statusRaw) {
    try {$state = ($statusRaw | ConvertFrom-Json).BackendState} catch {$state='status_unavailable'}
}
$receipt = @{
    status='client_installed_login_pending'; version=$version; msi_exit_code=$install.ExitCode;
    reboot_required=($install.ExitCode -eq 3010); checksum_verified=$true; publisher_verified=$true;
    default_routes_unchanged=(-not [bool](Compare-Object $beforeRoutes $afterRoutes));
    existing_dns_unchanged=($dnsRemovedOrChanged.Count -eq 0);
    dashboard_health_before=$dashboardBefore; dashboard_health_after=$health;
    backend_state=$state; network_enrollment_performed=$false; cloud_deployed=$false;
    at=(Get-Date).ToUniversalTime().ToString('o')
}
$receipt | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage 'install-receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json -Compress
