[CmdletBinding()]
param(
    [switch]$SkipFrontend,
    [switch]$SkipNcnn,
    [switch]$Clean,
    [ValidateRange(1, 32)]
    [int]$Jobs = 8
)

$ErrorActionPreference = "Stop"

$rootDir = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$frontendDir = Join-Path $rootDir "frontend"
$dependencyDir = Join-Path $rootDir ".deps"
$sourceParent = Join-Path $dependencyDir "src"
$ncnnSource = Join-Path $sourceParent "ncnn-20260526"
$ncnnBuild = Join-Path $dependencyDir "build\ncnn-vulkan"
$ncnnInstall = Join-Path $dependencyDir "install\ncnn-vulkan"
$runnerBuild = Join-Path $rootDir "build\windows"
$runnerExe = Join-Path $runnerBuild "Release\inference_runner_windows.exe"

function Assert-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE." }
}

function Find-VisualStudioCMake {
    $command = Get-Command cmake -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { return $null }
    $installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installation) { return $null }
    $candidate = Join-Path $installation "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    return $null
}

function Remove-ProjectDirectory([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    $rootPrefix = $rootDir.TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a directory outside the repository: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "This build script supports 64-bit Windows only."
}

$cmake = Find-VisualStudioCMake
if (-not $cmake) { throw "CMake from Visual Studio C++ Build Tools was not found. Run scripts/setup_windows.ps1." }
$ctest = Join-Path (Split-Path -Parent $cmake) "ctest.exe"
if (-not (Test-Path -LiteralPath $ctest -PathType Leaf)) { throw "ctest.exe was not found beside $cmake." }

if ($Clean) {
    Remove-ProjectDirectory $runnerBuild
    if (-not $SkipNcnn) {
        Remove-ProjectDirectory $ncnnBuild
        Remove-ProjectDirectory $ncnnInstall
    }
}

if (-not $SkipFrontend) {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
    if (-not $npm) { throw "npm was not found. Run scripts/setup_windows.ps1." }
    Push-Location $frontendDir
    try {
        & $npm.Source ci --prefer-offline --no-audit --no-fund
        Assert-Success "npm ci"
        & $npm.Source run check
        Assert-Success "frontend type-check"
        & $npm.Source run build
        Assert-Success "frontend production build"
    } finally {
        Pop-Location
    }
}

$frontendIndex = Join-Path $frontendDir "dist\index.html"
if (-not (Test-Path -LiteralPath $frontendIndex -PathType Leaf)) {
    throw "The production frontend is missing at $frontendIndex. Build it or omit -SkipFrontend."
}

if (-not $SkipNcnn) {
    $ncnnProject = Join-Path $ncnnSource "CMakeLists.txt"
    $glslangProject = Join-Path $ncnnSource "glslang\CMakeLists.txt"
    if (-not (Test-Path -LiteralPath $ncnnProject -PathType Leaf) -or -not (Test-Path -LiteralPath $glslangProject -PathType Leaf)) {
        $git = Get-Command git.exe -ErrorAction SilentlyContinue
        if (-not $git) { $git = Get-Command git -ErrorAction SilentlyContinue }
        if (-not $git) { throw "Git was not found. It is required to fetch NCNN and glslang." }
        Remove-ProjectDirectory $ncnnSource
        New-Item -ItemType Directory -Path $sourceParent -Force | Out-Null
        Write-Host "Downloading pinned NCNN 20260526 source..." -ForegroundColor Cyan
        & $git.Source -c http.sslBackend=schannel clone --branch 20260526 --depth 1 --recurse-submodules --shallow-submodules https://github.com/Tencent/ncnn.git $ncnnSource
        Assert-Success "NCNN source checkout"
    }

    Write-Host "Configuring NCNN with Vulkan compute..." -ForegroundColor Cyan
    $ncnnConfigure = @(
        "-S", $ncnnSource,
        "-B", $ncnnBuild,
        "-A", "x64",
        "-DCMAKE_INSTALL_PREFIX=$ncnnInstall",
        "-DNCNN_VULKAN=ON",
        "-DNCNN_SIMPLEVK=ON",
        "-DNCNN_SYSTEM_GLSLANG=OFF",
        "-DNCNN_BUILD_TOOLS=OFF",
        "-DNCNN_BUILD_BENCHMARK=OFF",
        "-DNCNN_BUILD_EXAMPLES=OFF",
        "-DNCNN_BUILD_TESTS=OFF",
        "-DNCNN_PYTHON=OFF",
        "-DNCNN_SHARED_LIB=OFF",
        "-DNCNN_OPENMP=ON"
    )
    & $cmake @ncnnConfigure
    Assert-Success "NCNN configure"
    & $cmake --build $ncnnBuild --config Release --target install --parallel $Jobs
    Assert-Success "NCNN Vulkan build"
}

$ncnnConfig = Join-Path $ncnnInstall "lib\cmake\ncnn\ncnnConfig.cmake"
if (-not (Test-Path -LiteralPath $ncnnConfig -PathType Leaf)) {
    throw "The Vulkan NCNN installation is missing at $ncnnInstall. Build it or omit -SkipNcnn."
}
if (-not (Select-String -LiteralPath $ncnnConfig -Pattern "set\(NCNN_VULKAN ON\)" -Quiet)) {
    throw "The selected NCNN installation was not built with Vulkan enabled."
}

Write-Host "Configuring Inference Runner Windows..." -ForegroundColor Cyan
$runnerConfigure = @(
    "-S", (Join-Path $rootDir "backend"),
    "-B", $runnerBuild,
    "-A", "x64",
    "-Dncnn_DIR=$(Join-Path $ncnnInstall 'lib\cmake\ncnn')",
    "-DRUNNER_WITH_NCNN=ON",
    "-DRUNNER_REQUIRE_VULKAN=ON",
    "-DBUILD_TESTING=ON",
    "-DRUNNER_FRONTEND_DIST=$(Join-Path $frontendDir 'dist')"
)
& $cmake @runnerConfigure
Assert-Success "runner configure"
& $cmake --build $runnerBuild --config Release --parallel $Jobs
Assert-Success "runner build"
& $ctest --test-dir $runnerBuild -C Release --output-on-failure
Assert-Success "native tests"

if (-not (Test-Path -LiteralPath $runnerExe -PathType Leaf)) {
    throw "Build succeeded without creating $runnerExe."
}

Write-Host ""
Write-Host "Build complete: $runnerExe" -ForegroundColor Green
Write-Host "Run: powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1"
