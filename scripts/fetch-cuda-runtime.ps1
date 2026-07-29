<#
.SYNOPSIS
    Fetches the CUDA runtime libraries iw3 Stereo's GPU path needs.

.DESCRIPTION
    The plugin does not ship these. They are about a gigabyte -- four times the
    rest of the bundle -- and most people who already have a CUDA install do not
    need them, so they are fetched once instead of shipped every time.

    Two libraries are involved and they fail differently, which is worth knowing
    if this goes wrong:

      cuBLAS  is a static import of ONNX Runtime's CUDA provider. Without it the
              provider will not load at all, and the log says
              'cublasLt64_13.dll ... is missing'. That is the error most people
              hit first.
      cuDNN   is loaded by name at the moment the first convolution runs. Without
              it the provider loads happily and then every Conv node fails with
              'cuDNN is unavailable'. That looks like a broken plugin rather than
              a missing file, so if you see it, this script is the answer.

    Both come from NVIDIA's own packages on PyPI, which are plain zip files --
    nothing is installed, no Python is involved, and pip is not used. Versions
    are resolved from PyPI at run time rather than hardcoded here, so the URLs
    cannot rot.

    You do NOT need the CUDA Toolkit. Installing it would not be enough anyway:
    cuDNN is a separate product that the Toolkit installer does not include.

.PARAMETER BundleDirectory
    The installed plugin bundle's Contents\Win64 folder.

    Not usually needed. This script ships inside the bundle, so by default it
    works out where it is and targets the bundle it came from -- which also means
    it works when the plugin is installed somewhere other than the standard
    directory. Only pass this if you are running a copy from outside a bundle
    and the standard location is not where the plugin lives.

    It does not matter what directory you run this from.

.PARAMETER IncludeAdv
    Also fetch cudnn_adv64_9.dll (about 100 MB). Skipped by default: it holds
    cuDNN's recurrent-network and fused-attention kernels, and the models that
    ship with this plugin use neither -- verified by removing it and running the
    full render path. Use this if a future model needs it, or if the log
    complains about cuDNN when everything else here succeeded.

.PARAMETER Force
    Re-download and overwrite files that are already present.

.EXAMPLE
    From an elevated PowerShell, from any directory:

    powershell -ExecutionPolicy Bypass -File "C:\Program Files\Common Files\OFX\Plugins\iw3stereo.ofx.bundle\fetch-cuda-runtime.ps1"

.NOTES
    These libraries are NVIDIA's and are covered by NVIDIA's licence terms, not
    this project's. Each package's License.txt is written into the target
    directory beside the DLLs.
#>

[CmdletBinding()]
param(
    [string] $BundleDirectory,
    [switch] $IncludeAdv,
    [switch] $Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Work out which bundle to fill, so that running this needs no arguments and no
# particular working directory.
#
# $PSScriptRoot is deliberately resolved here and not in the param() default
# above: under Windows PowerShell 5.1 it is not populated while parameter
# defaults are being evaluated, which would silently leave this empty. Same trap
# and same workaround as install-ofx.ps1.
if (-not $BundleDirectory) {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }

    # Shipped inside the bundle: the sibling of this script is Contents\Win64.
    # Preferring this over the standard path is what makes a non-default install
    # location work without anyone having to notice.
    $beside = if ($scriptDir) { Join-Path $scriptDir "Contents\Win64" } else { $null }
    if ($beside -and (Test-Path (Join-Path $beside "ort"))) {
        $BundleDirectory = $beside
    }
    else {
        $BundleDirectory =
            "C:\Program Files\Common Files\OFX\Plugins\iw3stereo.ofx.bundle\Contents\Win64"
    }
}

# The runtime lives in a subdirectory of the bundle so that its provider DLLs
# resolve from there rather than from Resolve's application directory. These go
# in beside them, and the plugin pre-loads whatever it finds.
$target = Join-Path $BundleDirectory "ort"

if (-not (Test-Path $target)) {
    Write-Error @"
Not found: $target

That should be the 'ort' folder inside the installed plugin bundle. Install the
plugin first (scripts\install-ofx.ps1), or pass -BundleDirectory pointing at
...\iw3stereo.ofx.bundle\Contents\Win64.
"@
}

# Writing into Program Files needs elevation; say so before downloading a
# gigabyte rather than after.
$probe = Join-Path $target ".write-probe"
try {
    New-Item -ItemType File -Path $probe -ErrorAction Stop | Out-Null
    Remove-Item $probe -Force
}
catch {
    Write-Error @"
Cannot write to $target

Run this from an elevated PowerShell (right-click > Run as administrator).
"@
}

# What to take from each package. The wheels carry more than this -- headers,
# import libraries, nvblas -- and none of it is needed at run time.
$packages = @(
    @{
        Name    = "nvidia-cublas"
        Why     = "cuBLAS -- without it the CUDA provider will not load"
        Include = @("cublas64_13.dll", "cublasLt64_13.dll")
    },
    @{
        Name    = "nvidia-cudnn-cu13"
        Why     = "cuDNN -- without it convolutions fail at run time"
        Include = @(
            "cudnn64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_engines_tensor_ir64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_ext64_9.dll"
        )
    }
)
if ($IncludeAdv) {
    $packages[1].Include += "cudnn_adv64_9.dll"
}

$scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("iw3-cuda-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratch | Out-Null

$written = New-Object System.Collections.Generic.List[string]
$totalBytes = 0L

try {
    foreach ($package in $packages) {
        $name = $package.Name
        Write-Host ""
        Write-Host "== $name" -ForegroundColor Cyan
        Write-Host "   $($package.Why)"

        # Resolve the newest Windows wheel from PyPI rather than hardcoding a URL.
        $meta = Invoke-RestMethod -Uri "https://pypi.org/pypi/$name/json" -UseBasicParsing
        $wheel = $meta.urls | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
        if ($null -eq $wheel) {
            Write-Error "No Windows wheel for $name. Report this -- the packaging upstream has changed."
        }

        $missing = @($package.Include | Where-Object {
            $Force -or -not (Test-Path (Join-Path $target $_))
        })
        if ($missing.Count -eq 0) {
            Write-Host "   already present, skipping" -ForegroundColor DarkGray
            continue
        }

        $mb = [math]::Round($wheel.size / 1MB, 1)
        Write-Host "   version $($meta.info.version), downloading $mb MB..."
        $archive = Join-Path $scratch ($wheel.filename -replace "\.whl$", ".zip")

        # A wheel is a zip. Expand-Archive needs the extension to say so.
        #
        # Progress display off during the transfer: with it on, Invoke-WebRequest
        # spends more time redrawing the bar than moving bytes, and 400 MB takes
        # minutes rather than seconds.
        $previous = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        try {
            Invoke-WebRequest -Uri $wheel.url -OutFile $archive -UseBasicParsing
        }
        finally {
            $ProgressPreference = $previous
        }

        $extractDir = Join-Path $scratch ("x-" + $name)
        Expand-Archive -LiteralPath $archive -DestinationPath $extractDir -Force
        Remove-Item $archive -Force

        foreach ($dll in $package.Include) {
            $source = Get-ChildItem -Path $extractDir -Filter $dll -Recurse -File |
                      Select-Object -First 1
            if ($null -eq $source) {
                Write-Warning "   $dll not in this release; skipping"
                continue
            }
            Copy-Item -LiteralPath $source.FullName -Destination (Join-Path $target $dll) -Force
            $written.Add($dll)
            $totalBytes += $source.Length
            Write-Host ("   {0,-42} {1,7:N1} MB" -f $dll, ($source.Length / 1MB))
        }

        # The licence travels with the binaries. These are NVIDIA's terms, not
        # this project's, and someone looking at the folder should be able to
        # find them without going back to PyPI.
        $licence = Get-ChildItem -Path $extractDir -Filter "License.txt" -Recurse -File |
                   Select-Object -First 1
        if ($null -ne $licence) {
            Copy-Item -LiteralPath $licence.FullName `
                      -Destination (Join-Path $target "LICENSE-$name.txt") -Force
        }

        Remove-Item $extractDir -Recurse -Force
    }
}
finally {
    if (Test-Path $scratch) { Remove-Item $scratch -Recurse -Force }
}

Write-Host ""
if ($written.Count -eq 0) {
    Write-Host "Nothing to do -- everything was already in place." -ForegroundColor Green
    Write-Host "Use -Force to re-download."
    return
}

# Verify rather than assume. A truncated download that Expand-Archive accepted
# would otherwise show up much later as a plugin that does not work.
$bad = @()
foreach ($dll in $written) {
    $path = Join-Path $target $dll
    if (-not (Test-Path $path)) { $bad += "$dll (missing)"; continue }
    # 'MZ' -- enough to catch a truncated or HTML-error-page download, which is
    # the realistic failure and one Expand-Archive would not have complained
    # about on its own.
    $header = New-Object byte[] 2
    $stream = [System.IO.File]::OpenRead($path)
    try { $stream.Read($header, 0, 2) | Out-Null } finally { $stream.Close() }
    if ($header[0] -ne 0x4D -or $header[1] -ne 0x5A) { $bad += "$dll (not a DLL)" }
}
if ($bad.Count -gt 0) {
    Write-Error ("These files did not come out intact: " + ($bad -join ", ") +
                 ". Re-run with -Force.")
}

Write-Host ("Installed {0} libraries, {1:N0} MB, into:" -f $written.Count, ($totalBytes / 1MB)) `
    -ForegroundColor Green
Write-Host "  $target"
Write-Host ""
Write-Host "Restart DaVinci Resolve. If the GPU path still does not start, the log at"
Write-Host "  %LOCALAPPDATA%\iw3probe\probe.log"
Write-Host "lists every library the plugin pre-loaded and any it could not."
