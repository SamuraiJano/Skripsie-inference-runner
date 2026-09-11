[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$rootDir = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$venvPython = Join-Path $rootDir ".venv\Scripts\python.exe"
$requirements = Join-Path $rootDir "requirements-windows.txt"
$modelDir = Join-Path $rootDir "models\GroundingDINO-Tiny"
$modelRevision = "a2bb814dd30d776dcf7e30523b00659f4f141c71"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "The project Python environment is missing. Run npm run setup:windows first."
}

& $venvPython -m pip install --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Grounding DINO runtime package installation failed." }

& $venvPython -c @"
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor

if transformers.__version__ != '4.57.1':
    raise SystemExit(f'Grounding DINO requires transformers 4.57.1, found {transformers.__version__}')
if not torch.cuda.is_available():
    raise SystemExit('A CUDA-enabled PyTorch build is required for Grounding DINO Tiny.')

model = Path(r'$modelDir')
required = (
    'added_tokens.json',
    'config.json',
    'model.safetensors',
    'preprocessor_config.json',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.txt',
)
if not all((model / name).is_file() for name in required):
    print(f'Downloading the pinned Grounding DINO Tiny checkpoint to {model} ...', flush=True)
    snapshot_download(
        repo_id='IDEA-Research/grounding-dino-tiny',
        revision='$modelRevision',
        local_dir=model,
        allow_patterns=list(required),
        max_workers=4,
    )
missing = [name for name in required if not (model / name).is_file()]
if missing:
    raise SystemExit(f'Grounding DINO Tiny local checkpoint is incomplete: {missing}')
config = AutoConfig.from_pretrained(str(model), local_files_only=True)
processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
if config.model_type != 'grounding-dino' or processor is None:
    raise SystemExit('The downloaded snapshot is not a Grounding DINO checkpoint.')
print(f'Local checkpoint : {model}')
print(f'Model revision   : $modelRevision')
print(f'Transformers     : {transformers.__version__}')
print(f'CUDA GPU         : {torch.cuda.get_device_name(0)}')
"@
if ($LASTEXITCODE -ne 0) { throw "Grounding DINO Tiny installed but its local checkpoint could not be validated." }

Write-Host "Grounding DINO Tiny is ready. Add it from the dashboard without another download." -ForegroundColor Green
