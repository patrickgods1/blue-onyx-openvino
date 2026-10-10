#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Removes the Blue Onyx Prism Windows service, its event log source and firewall rule.
#>
param([int]$Port = 32168)

$ErrorActionPreference = "Continue"
$ServiceName = "BlueOnyxPrismService"
$EventSource = "BlueOnyxPrism"

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne "Stopped") {
        Write-Host "Stopping $ServiceName ..."
        Stop-Service $ServiceName -Force
        Start-Sleep -Seconds 2
    }
    Write-Host "Deleting $ServiceName ..."
    sc.exe delete $ServiceName | Out-Null
} else {
    Write-Host "Service $ServiceName not installed."
}

$ruleName = "Blue Onyx Prism ($Port)"
if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) {
    Remove-NetFirewallRule -DisplayName $ruleName
    Write-Host "Removed firewall rule '$ruleName'."
}

try {
    if ([System.Diagnostics.EventLog]::SourceExists($EventSource)) {
        Remove-EventLog -Source $EventSource
        Write-Host "Removed event log source $EventSource."
    }
} catch { Write-Warning "Could not remove event log source: $_" }

Write-Host "Done."
