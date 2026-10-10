#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs Blue Onyx Prism as a Windows service (BlueOnyxPrismService).
.DESCRIPTION
    - Creates the Application event log source "BlueOnyxPrism"
    - Raises ServicesPipeTimeout to 600000 ms so the service has time to compile models on first start
    - Registers the service (auto start, LocalSystem) pointing at blue-onyx-prism-service.exe
    - Configures automatic restart on failure and a firewall rule for the listening port
.PARAMETER InstallDir
    Directory containing blue-onyx-prism-service.exe (default: the parent of this script's folder).
.PARAMETER Port
    TCP port to open in Windows Firewall (default 32168).
.PARAMETER Account
    Optional "DOMAIN\user" to run the service as, if GPU access fails under LocalSystem.
.PARAMETER NoStart
    Do not start the service after installing.
#>
param(
    [string]$InstallDir = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 32168,
    [string]$Account = "",
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$ServiceName = "BlueOnyxPrismService"
$DisplayName = "Blue Onyx Prism"
$EventSource = "BlueOnyxPrism"

$exe = Join-Path $InstallDir "blue-onyx-prism-service.exe"
if (-not (Test-Path $exe)) {
    throw "blue-onyx-prism-service.exe not found in '$InstallDir'. Pass -InstallDir."
}
if (-not (Test-Path (Join-Path $InstallDir "openvino"))) {
    Write-Warning "No 'openvino' folder next to the service exe. Run 'blue-onyx-prism setup-openvino' first."
}

Write-Host "Creating event log source $EventSource ..."
try {
    if (-not [System.Diagnostics.EventLog]::SourceExists($EventSource)) {
        New-EventLog -LogName Application -Source $EventSource
    }
} catch { Write-Warning "Could not create event log source: $_" }

Write-Host "Setting ServicesPipeTimeout to 600000 ms (takes effect after reboot) ..."
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control" -Name ServicesPipeTimeout -Value 600000 -Type DWord

# Pre-rename service (Blue Onyx OpenVINO): remove it so the two don't fight over the port.
$legacy = Get-Service -Name "BlueOnyxOpenVINOService" -ErrorAction SilentlyContinue
if ($legacy) {
    Write-Host "Removing the old BlueOnyxOpenVINOService ..."
    if ($legacy.Status -ne "Stopped") { Stop-Service "BlueOnyxOpenVINOService" -Force; Start-Sleep -Seconds 2 }
    sc.exe delete "BlueOnyxOpenVINOService" | Out-Null
    Start-Sleep -Seconds 2
}

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Service already exists; stopping and deleting it first ..."
    if ($existing.Status -ne "Stopped") { Stop-Service $ServiceName -Force; Start-Sleep -Seconds 2 }
    sc.exe delete $ServiceName | Out-Null
    Start-Sleep -Seconds 2
}

Write-Host "Registering service $ServiceName ..."
$binPath = "`"$exe`""
if ($Account) {
    $cred = Get-Credential -UserName $Account -Message "Password for $Account"
    $pw = $cred.GetNetworkCredential().Password
    sc.exe create $ServiceName binPath= $binPath start= auto DisplayName= "$DisplayName" obj= $Account password= $pw | Out-Null
} else {
    sc.exe create $ServiceName binPath= $binPath start= auto DisplayName= "$DisplayName" obj= LocalSystem | Out-Null
}
sc.exe config $ServiceName type= own | Out-Null
sc.exe description $ServiceName "Blue Iris / CodeProject.AI compatible object detection (OpenVINO / ONNX Runtime)" | Out-Null
sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/5000/restart/5000 | Out-Null

Write-Host "Adding firewall rule for TCP $Port ..."
$ruleName = "Blue Onyx Prism ($Port)"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
}

if (-not $NoStart) {
    Write-Host "Starting service ..."
    Start-Service $ServiceName
    Start-Sleep -Seconds 3
    Get-Service $ServiceName | Format-Table -AutoSize
    Write-Host "First start compiles the models; watch http://127.0.0.1:$Port/ and the Application event log."
}

Write-Host ""
Write-Host "Manage with:  net start $ServiceName | net stop $ServiceName | sc.exe delete $ServiceName"
Write-Host "Config file:  $(Join-Path $InstallDir 'blue_onyx_prism_config_service.json')"
