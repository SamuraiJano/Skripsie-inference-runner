[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [ValidateRange(1, 65535)]
    [int]$ModelServicePort = 8091,
    [string]$DetectorModelId = "",
    [string]$SegmenterModelId = "",
    [switch]$MaxPerformance
)

$ErrorActionPreference = "Stop"

$rootDir = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$executable = Join-Path $rootDir "build\windows\Release\inference_runner_windows.exe"
$dataDir = Join-Path $rootDir "data\models"
$frontendDir = Join-Path $rootDir "frontend\dist"
$pipelineConfig = Join-Path $rootDir "config\pipeline.json"
$modelService = Join-Path $rootDir "backend\python\model_service.py"
$python = Join-Path $rootDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { $pythonCommand = Get-Command python -ErrorAction SilentlyContinue }
    if (-not $pythonCommand) { throw "Python was not found. Run scripts/setup_windows.ps1 -InstallModelRuntimes." }
    $python = $pythonCommand.Source
}
if (-not (Test-Path -LiteralPath $pipelineConfig -PathType Leaf)) {
    $pipelineConfig = Join-Path $rootDir "config\pipeline.example.json"
}

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "The runner has not been built. Run npm run build:windows first."
}
if (-not (Test-Path -LiteralPath $frontendDir -PathType Container)) {
    throw "The frontend bundle is missing. Run npm run build:windows first."
}
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
$logDir = Join-Path $rootDir "data\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) { $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue }
if (-not $nvidiaSmi) { throw "nvidia-smi was not found; the RTX runtime cannot be verified." }
& $nvidiaSmi.Source --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
if ($LASTEXITCODE -ne 0) { throw "The NVIDIA driver did not report an available GPU." }

$arguments = @(
    "--host", $HostAddress,
    "--port", $Port.ToString(),
    "--data", $dataDir,
    "--static", $frontendDir,
    "--pipeline-config", $pipelineConfig,
    "--python-backend", "http://127.0.0.1:$ModelServicePort"
)
if ($DetectorModelId) { $arguments += @("--detector-model", $DetectorModelId) }
if ($SegmenterModelId) { $arguments += @("--segmenter-model", $SegmenterModelId) }

$previousScheme = $null
$powerChanged = $false
if ($MaxPerformance) {
    $activeText = (powercfg /getactivescheme) -join " "
    if ($activeText -match "([0-9a-fA-F-]{36})") { $previousScheme = $Matches[1] }
    $performanceLine = powercfg /list | Select-String -Pattern "\((Performance|High performance|Ultimate Performance)\)" | Select-Object -First 1
    if ($performanceLine -and $performanceLine.Line -match "([0-9a-fA-F-]{36})") {
        $performanceScheme = $Matches[1]
        if ($performanceScheme -ne $previousScheme) {
            powercfg /setactive $performanceScheme
            if ($LASTEXITCODE -eq 0) { $powerChanged = $true }
        }
    }

    $preferenceKey = "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences"
    if (-not (Test-Path -LiteralPath $preferenceKey)) { New-Item -Path $preferenceKey -Force | Out-Null }
    New-ItemProperty -Path $preferenceKey -Name $executable -Value "GpuPreference=2;" -PropertyType String -Force | Out-Null
}

Write-Host "Dashboard: http://$HostAddress`:$Port" -ForegroundColor Cyan
Write-Host "Pipeline : $pipelineConfig"
Write-Host "Models   : $dataDir"
Write-Host "Backends : NCNN/Vulkan + Ultralytics/CUDA"

$quotedArguments = $arguments | ForEach-Object {
    $value = [string]$_
    if ($value.Contains(' ')) { '"' + $value.Replace('"', '\"') + '"' } else { $value }
}

$exitCode = 1
$process = $null
$modelProcess = $null

function Stop-ProcessTree([int]$RootProcessId) {
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $RootProcessId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) { Stop-ProcessTree ([int]$child.ProcessId) }
    Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

try {
    $modelArguments = @(
        ('"' + $modelService + '"'),
        "--host", "127.0.0.1",
        "--port", $ModelServicePort.ToString(),
        "--models-root", ('"' + $dataDir + '"')
    )
    $modelProcess = Start-Process -FilePath $python -ArgumentList $modelArguments -WorkingDirectory $rootDir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "model-service.out.log") `
        -RedirectStandardError (Join-Path $logDir "model-service.err.log") -PassThru
    $modelHealthUrl = "http://127.0.0.1:$ModelServicePort/health"
    $modelHealth = $null
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if ($modelProcess.HasExited) {
            $details = Get-Content -Raw (Join-Path $logDir "model-service.err.log") -ErrorAction SilentlyContinue
            throw "The model service stopped during startup. $details"
        }
        try {
            $modelHealth = Invoke-RestMethod -Uri $modelHealthUrl -TimeoutSec 2
            break
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $modelHealth) { throw "The model service did not become ready at $modelHealthUrl." }
    if (-not $modelHealth.cudaAvailable) { throw "The Python runtime cannot use the RTX GPU. Run scripts/setup_windows.ps1 -InstallModelRuntimes." }
    Write-Host "CUDA     : PyTorch $($modelHealth.torch), CUDA $($modelHealth.cudaVersion), $($modelHealth.gpuDevice)" -ForegroundColor Green

    $process = Start-Process -FilePath $executable -ArgumentList $quotedArguments -WorkingDirectory $rootDir -NoNewWindow -PassThru
    if ($MaxPerformance) {
        try { $process.PriorityClass = [Diagnostics.ProcessPriorityClass]::High } catch { Write-Warning "Could not set high process priority: $($_.Exception.Message)" }
    }
    while (-not $process.WaitForExit(500)) { }
    $exitCode = $process.ExitCode
} finally {
    if ($process) { Stop-ProcessTree $process.Id }
    if ($modelProcess) { Stop-ProcessTree $modelProcess.Id }
    if ($powerChanged -and $previousScheme) { powercfg /setactive $previousScheme | Out-Null }
}

exit $exitCode
