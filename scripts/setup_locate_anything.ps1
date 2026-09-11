[CmdletBinding()]
param(
    [string]$BasePython = ""
)

$ErrorActionPreference = "Stop"
$rootDir = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$venvPython = Join-Path $rootDir ".venv\Scripts\python.exe"
$requirements = Join-Path $rootDir "requirements-locate-anything-windows.txt"
$revision = "bb860a7efa22bb40cd4bb77959512375b92db934"
$workerSha256 = "021F26B07220B9DCB090023286EF52F0A55392D0C43EC1A0490645A2C5C165A4"
$sourceDir = Join-Path $rootDir ".deps\src\locate-anything-$revision"
$workerFile = Join-Path $sourceDir "locateanything_worker.py"
$modelRevision = "c32291ca5e996f5a7a485845b4f57a233936bba0"
$modelDir = Join-Path $rootDir "models\LocateAnything-3B"

if (-not $BasePython) {
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $BasePython = $venvPython
    } else {
        $command = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $command) { $command = Get-Command python -ErrorAction SilentlyContinue }
        if (-not $command) { throw "Python 3.10-3.12 was not found. Run npm run setup:windows first." }
        $BasePython = $command.Source
    }
}

if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "LocateAnything requirements are missing at $requirements."
}

New-Item -ItemType Directory -Path $sourceDir -Force | Out-Null
$downloadWorker = $true
if (Test-Path -LiteralPath $workerFile -PathType Leaf) {
    $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $workerFile).Hash
    $downloadWorker = $existingHash -ne $workerSha256
}
if ($downloadWorker) {
    $temporaryWorker = Join-Path $sourceDir ".locateanything_worker.download"
    $workerUrl = "https://raw.githubusercontent.com/NVlabs/Eagle/$revision/Embodied/locateanything_worker.py"
    Write-Host "Downloading the pinned LocateAnything worker from NVlabs/Eagle..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $workerUrl -UseBasicParsing -OutFile $temporaryWorker
    $downloadHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporaryWorker).Hash
    if ($downloadHash -ne $workerSha256) {
        Remove-Item -LiteralPath $temporaryWorker -Force
        throw "LocateAnything worker integrity check failed. Expected $workerSha256, received $downloadHash."
    }
    Move-Item -LiteralPath $temporaryWorker -Destination $workerFile -Force
}

@(
    "NVlabs/Eagle",
    "https://github.com/NVlabs/Eagle/tree/$revision/Embodied",
    $revision
) | Set-Content -LiteralPath (Join-Path $sourceDir "SOURCE.txt") -Encoding UTF8

& $BasePython -m pip install --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) { throw "LocateAnything runtime package installation failed." }

& $BasePython -c @"
import sys
from importlib.metadata import version
from pathlib import Path
source = Path(r'$sourceDir')
sys.path.insert(0, str(source))
import decord
import torch
import transformers
from huggingface_hub import snapshot_download
from locateanything_worker import LocateAnythingWorker
assert LocateAnythingWorker
decord_version = version('decord')
if transformers.__version__ != '4.57.1':
    raise SystemExit(f'LocateAnything requires transformers 4.57.1, found {transformers.__version__}')
if not torch.cuda.is_available():
    raise SystemExit('A CUDA-enabled PyTorch build is required for LocateAnything.')
if decord_version != '0.6.0':
    raise SystemExit(f'LocateAnything requires decord 0.6.0, found {decord_version}')
model = Path(r'$modelDir')
required = (
    'config.json',
    'model.safetensors.index.json',
    'model-00001-of-00002.safetensors',
    'model-00002-of-00002.safetensors',
    'processing_locateanything.py',
    'tokenizer_config.json',
)
if not all((model / name).is_file() for name in required):
    print(f'Downloading the pinned NVIDIA checkpoint to {model} ...', flush=True)
    snapshot_download(
        repo_id='nvidia/LocateAnything-3B',
        revision='$modelRevision',
        local_dir=model,
        max_workers=2,
    )
missing = [name for name in required if not (model / name).is_file()]
if missing:
    raise SystemExit(f'LocateAnything local checkpoint is incomplete: {missing}')
print(f'LocateAnything source : {source}')
print(f'Local checkpoint     : {model}')
print(f'Transformers          : {transformers.__version__}')
print(f'Decord               : {decord_version}')
print(f'CUDA GPU              : {torch.cuda.get_device_name(0)}')
"@
if ($LASTEXITCODE -ne 0) { throw "LocateAnything installed but its worker could not be loaded." }

Write-Host "LocateAnything is ready. Dashboard imports now load the local checkpoint without downloading." -ForegroundColor Green
