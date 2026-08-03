[CmdletBinding()]
param(
    [switch]$SetGpuPreference,
    [switch]$InstallModelRuntimes,
    [switch]$InstallTensorRT,
    [string]$Executable
)

$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $PSScriptRoot
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "Inference Runner Windows must be set up on Windows."
}

if (-not $Executable) {
    $Executable = Join-Path $rootDir "build\windows\Release\inference_runner_windows.exe"
}
$Executable = [IO.Path]::GetFullPath($Executable)

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

$issues = [System.Collections.Generic.List[string]]::new()
$git = Get-Command git.exe -ErrorAction SilentlyContinue
if (-not $git) { $git = Get-Command git -ErrorAction SilentlyContinue }
if (-not $git) { $issues.Add("Git is missing; it is required to fetch NCNN and its pinned glslang submodule.") }
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { $npm = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $npm) { $issues.Add("Node.js/npm is missing.") }
$python = Join-Path $rootDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { $pythonCommand = Get-Command python -ErrorAction SilentlyContinue }
    if ($pythonCommand) { $python = $pythonCommand.Source } else { $python = $null }
}
if (-not $python) { $issues.Add("Python 3.10-3.12 is missing; it is required for direct model formats.") }

$cmake = Find-VisualStudioCMake
if (-not $cmake) { $issues.Add("Visual Studio C++ Build Tools with CMake is missing.") }

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) { $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue }
if (-not $nvidiaSmi) { $issues.Add("nvidia-smi was not found; install or repair the NVIDIA laptop driver.") }

$vulkanInfo = Get-Command vulkaninfo.exe -ErrorAction SilentlyContinue
if (-not $vulkanInfo) { $vulkanInfo = Get-Command vulkaninfo -ErrorAction SilentlyContinue }
if (-not $vulkanInfo) { $issues.Add("vulkaninfo was not found; install or repair the NVIDIA Vulkan runtime.") }

Write-Host ""
Write-Host "Inference Runner Windows preflight" -ForegroundColor Cyan
Write-Host "Repository : $rootDir"

$computer = Get-CimInstance Win32_ComputerSystem
$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$memoryGiB = [Math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
Write-Host "Laptop     : $($computer.Manufacturer) $($computer.Model)"
Write-Host "CPU        : $($processor.Name.Trim()) ($($processor.NumberOfCores)C/$($processor.NumberOfLogicalProcessors)T)"
Write-Host "Memory     : $memoryGiB GiB"
Write-Host "Power plan : $((powercfg /getactivescheme) -join ' ')"

if ($nvidiaSmi) {
    Write-Host ""
    Write-Host "NVIDIA GPU" -ForegroundColor Cyan
    & $nvidiaSmi.Source --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    if ($LASTEXITCODE -ne 0) { $issues.Add("nvidia-smi could not query an NVIDIA GPU.") }
}

if ($vulkanInfo) {
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $vulkanSummary = & $vulkanInfo.Source --summary 2>&1
    } finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    $vulkanDevices = $vulkanSummary | Select-String -Pattern "deviceName\s*="
    Write-Host ""
    Write-Host "Vulkan devices" -ForegroundColor Cyan
    if ($vulkanDevices) {
        $vulkanDevices | ForEach-Object { Write-Host $_.Line.Trim() }
    } else {
        $issues.Add("vulkaninfo did not report a compute device.")
    }
}

if ($npm) { Write-Host "npm        : $(& $npm.Source --version)" }
if ($git) { Write-Host "Git        : $(& $git.Source --version)" }
if ($cmake) { Write-Host "CMake      : $cmake" }
if ($python) {
    Write-Host "Python     : $(& $python --version)"
    & $python -c "import torch, ultralytics; assert torch.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -ne 0 -and -not ($InstallModelRuntimes -or $InstallTensorRT)) {
        $issues.Add("The PyTorch CUDA/Ultralytics runtime is missing. Re-run with -InstallModelRuntimes.")
    }
}

if ($InstallModelRuntimes -or $InstallTensorRT) {
    & (Join-Path $PSScriptRoot "setup_model_runtimes.ps1") -BasePython $python -InstallTensorRT:$InstallTensorRT
    if ($LASTEXITCODE -ne 0) { $issues.Add("The model runtime setup failed.") }
}

if ($SetGpuPreference) {
    $preferenceKey = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
    if (-not (Test-Path -LiteralPath $preferenceKey)) { New-Item -Path $preferenceKey -Force | Out-Null }
    New-ItemProperty -Path $preferenceKey -Name $Executable -Value "GpuPreference=2;" -PropertyType String -Force | Out-Null
    Write-Host "GPU policy : high-performance preference registered for $Executable" -ForegroundColor Green
}

if ($issues.Count -gt 0) {
    throw "Preflight failed:`n - $($issues -join "`n - ")"
}

Write-Host ""
Write-Host "Windows, MSVC, NVIDIA, Vulkan, and model runtime prerequisites are ready." -ForegroundColor Green
