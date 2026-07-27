# Installs the built .ofx.bundle directories where Resolve looks for them.
#
# Resolve scans C:\Program Files\Common Files\OFX\Plugins, which is not
# writable without elevation, so run this from an elevated PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File scripts\install-ofx.ps1
#
# Resolve only scans at startup, so restart it after installing.

[CmdletBinding()]
param(
    [string]$BuildDir,
    [string]$PluginDir = 'C:\Program Files\Common Files\OFX\Plugins',
    # The Phase 0 probes are instrumentation, not product. Installing them
    # alongside the plugin means a second ONNX Runtime brought up in the same
    # process for no reason, so they are left out unless asked for.
    [switch]$IncludeProbes
)

$ProbeBundles = @('iw3probe.ofx.bundle', 'iw3ort.ofx.bundle')

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not populated while param() defaults are evaluated under
# Windows PowerShell 5.1, so the default is resolved here instead.
$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $scriptDir) { $scriptDir = (Get-Location).Path }

if (-not $BuildDir) {
    $BuildDir = Join-Path $scriptDir '..\ofx\build\bundles'
}
if (-not (Test-Path $BuildDir)) {
    Write-Error "Build output not found at $BuildDir - build the plugins first."
}
$BuildDir = (Resolve-Path $BuildDir).Path

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Not elevated. Re-run this script from an Administrator PowerShell."
}

if (-not (Test-Path $PluginDir)) {
    New-Item -ItemType Directory -Path $PluginDir -Force | Out-Null
}

$bundles = Get-ChildItem $BuildDir -Directory -Filter '*.ofx.bundle'
if (-not $IncludeProbes) {
    $bundles = $bundles | Where-Object { $ProbeBundles -notcontains $_.Name }
    # Remove any previously installed probe, so an old build does not linger.
    foreach ($name in $ProbeBundles) {
        $stale = Join-Path $PluginDir $name
        if (Test-Path $stale) {
            Remove-Item $stale -Recurse -Force
            Write-Output "removed $name (pass -IncludeProbes to keep the probes)"
        }
    }
}
if ($bundles.Count -eq 0) {
    Write-Error "No .ofx.bundle directories to install under $BuildDir - build first."
}

foreach ($bundle in $bundles) {
    $target = Join-Path $PluginDir $bundle.Name
    if (Test-Path $target) {
        Remove-Item $target -Recurse -Force
    }
    Copy-Item $bundle.FullName $target -Recurse -Force
    Write-Output "installed $($bundle.Name)"
}

Write-Output ""
Write-Output "Installed to $PluginDir. Restart DaVinci Resolve to pick them up."
