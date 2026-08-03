$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $PSScriptRoot
$frontendDir = Join-Path $rootDir "frontend"
$frontendIndex = Join-Path $frontendDir "dist\index.html"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js/npm on this Windows development computer and try again."
}

Push-Location $frontendDir
try {
    & npm ci --prefer-offline --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed with exit code $LASTEXITCODE." }

    & npm run check
    if ($LASTEXITCODE -ne 0) { throw "npm run check failed with exit code $LASTEXITCODE." }

    & npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $frontendIndex -PathType Leaf)) {
    throw "Frontend build completed without creating $frontendIndex."
}

Write-Host ""
Write-Host "Frontend build complete: $frontendIndex"
Write-Host "Run scripts\build_windows.ps1 to compile the Vulkan backend, or scripts\run_windows.ps1 if it is already built."
