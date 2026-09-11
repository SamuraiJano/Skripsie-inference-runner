[CmdletBinding()]
param(
    [string]$ModelPath = "",
    [int]$ApiPort = 18080,
    [int]$ModelServicePort = 18091
)

$ErrorActionPreference = "Stop"
$rootDir = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (-not $ModelPath) { $ModelPath = Join-Path $rootDir "models\yolo26n\best.pt" }
$ModelPath = [IO.Path]::GetFullPath($ModelPath)
$python = Join-Path $rootDir ".venv\Scripts\python.exe"
$runner = Join-Path $rootDir "build\windows\Release\inference_runner_windows.exe"
$frontend = Join-Path $rootDir "frontend\dist"
$pipeline = Join-Path $rootDir "config\pipeline.example.json"
$testRoot = Join-Path $rootDir "build\integration-data"
$modelsRoot = Join-Path $testRoot "models"
$stateRoot = Join-Path $testRoot "state"
$imagePath = Join-Path $testRoot "blank.jpg"

foreach ($required in @($ModelPath, $python, $runner)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file is missing: $required" }
}
$resolvedTestRoot = [IO.Path]::GetFullPath($testRoot)
$buildPrefix = [IO.Path]::GetFullPath((Join-Path $rootDir "build")).TrimEnd('\') + '\'
if (-not $resolvedTestRoot.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a test directory outside the repository build directory."
}
if (Test-Path -LiteralPath $resolvedTestRoot) { Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force }
New-Item -ItemType Directory -Path $modelsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
Copy-Item -LiteralPath $ModelPath -Destination (Join-Path $modelsRoot ([IO.Path]::GetFileName($ModelPath)))

Add-Type -AssemblyName System.Drawing
$bitmap = New-Object System.Drawing.Bitmap 640,640
try {
    $graphics = [Drawing.Graphics]::FromImage($bitmap)
    try { $graphics.Clear([Drawing.Color]::Black) } finally { $graphics.Dispose() }
    $bitmap.Save($imagePath, [Drawing.Imaging.ImageFormat]::Jpeg)
} finally {
    $bitmap.Dispose()
}

function Wait-Json([string]$Url, [Diagnostics.Process]$Process) {
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        if ($Process.HasExited) { throw "Process $($Process.Id) exited while waiting for $Url." }
        try { return Invoke-RestMethod -Uri $Url -TimeoutSec 2 } catch { Start-Sleep -Milliseconds 250 }
    }
    throw "Timed out waiting for $Url."
}

function Stop-ProcessTree([int]$RootProcessId) {
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $RootProcessId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) { Stop-ProcessTree ([int]$child.ProcessId) }
    Stop-Process -Id $RootProcessId -Force -ErrorAction SilentlyContinue
}

$modelProcess = $null
$runnerProcess = $null
try {
    $serviceArguments = @(
        ('"' + (Join-Path $rootDir "backend\python\model_service.py") + '"'),
        "--host", "127.0.0.1", "--port", $ModelServicePort.ToString(),
        "--models-root", ('"' + $modelsRoot + '"'),
        "--state-root", ('"' + $stateRoot + '"')
    )
    $modelProcess = Start-Process -FilePath $python -ArgumentList $serviceArguments -WorkingDirectory $rootDir -NoNewWindow -PassThru
    $modelHealth = Wait-Json "http://127.0.0.1:$ModelServicePort/health" $modelProcess

    $runnerArguments = @(
        "--host", "127.0.0.1", "--port", $ApiPort.ToString(),
        "--data", ('"' + $stateRoot + '"'), "--models", ('"' + $modelsRoot + '"'),
        "--static", ('"' + $frontend + '"'),
        "--pipeline-config", ('"' + $pipeline + '"'),
        "--python-backend", "http://127.0.0.1:$ModelServicePort"
    )
    $runnerProcess = Start-Process -FilePath $runner -ArgumentList $runnerArguments -WorkingDirectory $rootDir -NoNewWindow -PassThru
    $health = Wait-Json "http://127.0.0.1:$ApiPort/api/health" $runnerProcess

    $detected = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/api/models" -TimeoutSec 120
    if ($detected.Count -ne 1) { throw "Expected one automatically detected model, found $($detected.Count)." }
    $selected = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$ApiPort/api/models/$($detected[0].id)/activate" -TimeoutSec 300

    $inferenceJson = & curl.exe --silent --show-error --fail-with-body `
        -F "modelId=$($selected.id)" -F "confidence=0.25" -F "iou=0.5" -F "image=@$imagePath" `
        "http://127.0.0.1:$ApiPort/api/infer"
    if ($LASTEXITCODE -ne 0) { throw "Inference request failed: $inferenceJson" }
    $inference = $inferenceJson | ConvertFrom-Json

    [pscustomobject]@{
        ApiEngine = $health.engine
        GPU = $health.gpuDevice
        CUDA = $modelHealth.cudaAvailable
        AvailableFormats = (($health.formats | Where-Object available).format -join ", ")
        DetectedFormat = $selected.format
        DetectedPrecision = $selected.precision
        DetectedQuantization = $selected.quantization.label
        SelectedBackend = $selected.backend
        InferenceTask = $inference.task
        InferenceMilliseconds = [Math]::Round($inference.timing.backendTotalMs, 2)
    } | Format-List
} finally {
    if ($runnerProcess) { Stop-ProcessTree $runnerProcess.Id }
    if ($modelProcess) { Stop-ProcessTree $modelProcess.Id }
}
