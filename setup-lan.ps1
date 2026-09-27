# Configure this PC so phones on the LAN can reach the manager.
# Run from an ELEVATED PowerShell window.
#
#   .\setup-lan.ps1                          # auto-pick the connected adapter
#   .\setup-lan.ps1 -AdapterMatch 'Wi-Fi'    # pick by adapter description
#   .\setup-lan.ps1 -HostName llama          # also rename the PC (for llama.local)
#
# What it does:
#   1. marks the adapter's network profile as Private (the firewall rules below
#      only apply on Private networks, and Windows labels a fresh Wi-Fi Public)
#   2. sets the Bonjour service to Automatic and starts it (for <name>.local)
#   3. opens inbound TCP 80 + 8091, Private profile, LocalSubnet only
#   4. optionally renames the computer so Bonjour publishes <name>.local
#
# NOTE: this file is deliberately ASCII-only. Windows PowerShell 5.1 reads a
# BOM-less .ps1 as the system ANSI codepage (GBK on a Chinese install), which
# mangles non-ASCII string literals and fails with TerminatorExpectedAtEndOfString.
param(
    [string]$AdapterMatch = '',
    [string]$HostName = ''
)

$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated PowerShell window.'
}

$adapter = Get-NetAdapter -Physical | Where-Object {
    $_.Status -eq 'Up' -and (-not $AdapterMatch -or $_.InterfaceDescription -match $AdapterMatch)
} | Select-Object -First 1
if (-not $adapter) {
    if ($AdapterMatch) {
        throw "No connected physical adapter matches '$AdapterMatch'. Nothing was changed."
    }
    throw 'No connected physical adapter found. Nothing was changed.'
}

$ipConfig = Get-NetIPConfiguration -InterfaceIndex $adapter.ifIndex
if (-not $ipConfig.IPv4DefaultGateway) {
    throw "Adapter '$($adapter.Name)' has no IPv4 default gateway. Nothing was changed."
}

$profile = Get-NetConnectionProfile -InterfaceIndex $adapter.ifIndex
if ($profile.NetworkCategory -ne 'Private') {
    Set-NetConnectionProfile -InterfaceIndex $adapter.ifIndex -NetworkCategory Private
}

# Bonjour is optional: without it <name>.local will not resolve, but the IP works.
$bonjour = Get-Service -Name 'Bonjour Service' -ErrorAction SilentlyContinue
if ($bonjour) {
    Set-Service -Name $bonjour.Name -StartupType Automatic
    if ($bonjour.Status -ne 'Running') { Start-Service -Name $bonjour.Name }
} else {
    Write-Host 'Bonjour Service not found - skipping.'
    Write-Host 'Use the IP address below instead, or install Apple Bonjour for <name>.local.'
}

foreach ($port in 80, 8091) {
    $ruleName = "llm-manager-$port-lan"
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $port -Profile Private -RemoteAddress LocalSubnet |
        Out-Null
}

# rule left over from an older port layout
Get-NetFirewallRule -DisplayName 'llm-manager-8090-lan' -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule

$renamePending = $false
if ($HostName) {
    $renamePending = $env:COMPUTERNAME -ine $HostName
    if ($renamePending) { Rename-Computer -NewName $HostName -Force }
}

Write-Host "Adapter : $($adapter.Name) [$($adapter.InterfaceDescription)]"
Write-Host "Address : $($ipConfig.IPv4Address.IPAddress)  gateway $($ipConfig.IPv4DefaultGateway.NextHop)"
if ($bonjour) { Write-Host 'Bonjour : set to Automatic and running.' }
Write-Host 'Firewall: TCP 80/8091, Private profile, LocalSubnet only (old 8090 rule removed).'
if ($renamePending) {
    Write-Host "Computer renamed to '$HostName'. Reboot, then Bonjour publishes $HostName.local."
} elseif ($HostName) {
    Write-Host "Computer name is already '$HostName'."
}
