[CmdletBinding()]
param(
    [string]$BasePython = "",
    [switch]$InstallTensorRT
)

$ErrorActionPreference = "Stop"
$rootDir = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$venvDir = Join-Path $rootDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $rootDir "requirements-windows.txt"

if (-not $BasePython) {
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $command) { $command = Get-Command python -ErrorAction SilentlyContinue }
    if (-not $command) { throw "Python 3.10-3.12 was not found." }
    $BasePython = $command.Source
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating the local Python runtime..." -ForegroundColor Cyan
    & $BasePython -m venv --system-site-packages $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Could not create $venvDir." }
}

& $venvPython -m pip install --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Model runtime package installation failed." }
if ($InstallTensorRT) {
    & $venvPython -m pip install --disable-pip-version-check --upgrade tensorrt-cu12
    if ($LASTEXITCODE -ne 0) { throw "TensorRT CUDA 12 installation failed." }
    & $venvPython -c "import tensorrt as trt; print(f'TensorRT    : {trt.__version__}')"
    if ($LASTEXITCODE -ne 0) { throw "TensorRT installed but its Windows runtime could not be loaded." }
}

& $venvPython -c @"
import torch
import ultralytics
import onnxruntime as ort
import sys
sys.path.insert(0, r'$rootDir\backend\python')
from model_service import format_capabilities
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'
providers = ', '.join(ort.get_available_providers())
onnx_capability = next(item for item in format_capabilities() if item['format'] == 'onnx')
onnx_device = onnx_capability['accelerator']
print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}')
print(f'Ultralytics : {ultralytics.__version__}')
print(f'CUDA        : {torch.cuda.is_available()} ({torch.version.cuda})')
print(f'GPU         : {gpu_name}')
print(f'ONNX        : {ort.__version__} ({providers})')
print(f'ONNX device : {onnx_device}')
if not torch.cuda.is_available():
    raise SystemExit('A CUDA-enabled PyTorch build is required for direct .pt GPU inference.')
if not onnx_capability['accelerated']:
    raise SystemExit('ONNX Runtime could not create a CUDA execution session.')
"@
if ($LASTEXITCODE -ne 0) { throw "The local model runtime did not pass CUDA validation." }

Write-Host "Model runtimes are ready in $venvDir" -ForegroundColor Green
