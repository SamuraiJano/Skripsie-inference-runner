# Inference Runner Windows

Inference Runner Windows is the Windows/RTX edition of the local inference app. It keeps the React dashboard, still-image workflow, benchmarks, and live conveyor pipeline while replacing the Raspberry Pi deployment profile with a capability-driven Windows model layer. A model is routed to PyTorch/CUDA, ONNX Runtime CUDA, TensorRT, another installed Ultralytics runtime, or the native NCNN/Vulkan engine according to its format.

The application is local-first: model files, frames, predictions, benchmark samples, and conveyor results stay on this computer.

## Target laptop

The repository is tuned for the detected development machine:

- Windows x86-64 on an ASUS TUF Gaming A15 FA507NV;
- NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM;
- AMD Ryzen 7 7735HS with 8 cores / 16 logical processors;
- 32 GB system memory;
- NVIDIA Vulkan 1.4 driver;
- PyTorch CUDA 12.4 and ONNX Runtime CUDA for direct model artifacts;
- eight native helper threads and a 30 FPS browser submission ceiling by default.

Existing NCNN `.param`/`.bin` artifacts continue to use Vulkan. Direct `.pt`, TorchScript, and ONNX artifacts use the project-local Python model service and select the RTX 4060 through CUDA. The dashboard reports both format availability and the accelerator selected by each installed runtime; it does not label a CPU fallback as GPU execution.

## Quick start

Run these commands in Windows PowerShell from the repository root:

```powershell
npm run setup:windows
npm run build:windows
npm run start:windows
```

Open <http://127.0.0.1:8080>. To temporarily select the Windows Performance power plan, register the executable as a high-performance GPU application, and launch it with high process priority:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 -MaxPerformance
```

The launcher restores the previous Windows power plan after the process exits.

TensorRT `.engine` support is optional because engines are version- and GPU-specific. Install and validate its CUDA 12 runtime when needed:

```powershell
npm run setup:tensorrt
```

## Windows scripts

- `npm run setup:windows` checks Python, PyTorch CUDA, Node/npm, Visual Studio C++ Build Tools, the NVIDIA driver, Vulkan devices, CPU/RAM, and the active power plan, then creates `.venv` and installs the direct-model runtimes. The underlying `scripts/setup_windows.ps1` also accepts `-SetGpuPreference` to register the native executable as a high-performance GPU application.
- `scripts/setup_model_runtimes.ps1` creates the isolated Python environment and validates PyTorch, CUDA, Ultralytics, and ONNX Runtime providers.
- `scripts/build_windows.ps1` builds the React bundle, downloads the pinned NCNN 20260526 source when needed, builds NCNN with Vulkan enabled, compiles the C++ service with MSVC/AVX2, and runs the native tests.
- `scripts/run_windows.ps1` starts the private loopback model service and the C++ application together, then stops both on exit.
- `scripts/status_windows.ps1` reports NVIDIA utilization, VRAM, temperature, power draw, Windows power mode, process state, and API health.

Build only part of the project when iterating:

```powershell
# Frontend only
npm run check
npm run build

# Reuse already-built NCNN and skip npm work
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 -SkipFrontend -SkipNcnn
```

The production executable is written to:

```text
build/windows/Release/inference_runner_windows.exe
```

## Models and automatic backend selection

The Add model drawer has two import paths:

- **Model file** accepts a compatible Ultralytics detection, instance-segmentation, or classification artifact and generates its local manifest automatically.
- **NCNN package** retains the verified `manifest.json` + `model.param` + `model.bin` contract.

Availability is runtime-driven and returned by `GET /api/capabilities`. On this laptop the base setup provides:

| Artifact | Runtime | Expected device |
| --- | --- | --- |
| `.pt` | PyTorch | RTX 4060 / CUDA |
| `.torchscript` | PyTorch | RTX 4060 / CUDA |
| `.onnx` | ONNX Runtime GPU | RTX 4060 / CUDA |
| `.engine` | TensorRT, when installed | RTX 4060 / CUDA |
| `.param` + `.bin` + manifest | native NCNN | RTX 4060 / Vulkan |
| OpenVINO/SavedModel/Paddle export directory as `.zip`, `.pb`, `.tflite`, `.mnn`, `.pte` | corresponding optional runtime | shown in the UI when installed |

“Compatible” means that the artifact is an Ultralytics-compatible model for one of the three tasks the application can render: detection, instance segmentation, or classification. Hardware-specific formats still require their runtime and must have been exported for a compatible device/runtime version. For example, TensorRT engines are not portable across every GPU and TensorRT version.

PyTorch-style executable checkpoints (`.pt`, `.torchscript`, and `.pte`) require an explicit trusted-source confirmation before import. Do not load untrusted checkpoints.

The native NCNN manifest retains explicit Windows GPU selection:

```json
{
  "runtime": {
    "name": "ncnn",
    "version": "1.0.20260526",
    "threads": 8,
    "useVulkanCompute": true,
    "gpuDevice": 0,
    "useInt8Inference": false,
    "useInt8Storage": false,
    "useFp16Storage": true,
    "useFp16Arithmetic": true
  }
}
```

Use `gpuDevice: -1` to let NCNN choose its highest-scoring Vulkan device. On this laptop, device `0` is the discrete RTX 4060 and device `1` is the integrated AMD GPU. The example manifest pins device `0` so model inference cannot accidentally move to the iGPU.

The complete NCNN example is in `config/model-manifest.example.json`. Every imported artifact is copied into `data/models`, loaded, and warmed before being admitted to the model library. The active model ID is persisted in `data/models/active-model.txt`.

## Video object detection

Open the **Video detection** screen from the top navigation, choose a local video, select an installed object-detection model, and click **Run detection**. Only models imported with the object-detection task appear on this screen. The browser decodes the video and submits one frame at a time, so requests cannot build up when model inference is slower than video playback.

The screen accepts any video format that Microsoft Edge can decode. The selected model can use any compatible artifact format listed above; backend and GPU selection remain automatic.

## Live conveyor pipeline

Copy the example before changing deployment values:

```powershell
Copy-Item .\config\pipeline.example.json .\config\pipeline.json
```

The browser accepts a webcam or video file, JPEG-encodes frames locally, and sends at most one frame request at a time. The configured pipeline performs:

1. YOLO26n fruit detection;
2. ByteTrack identity assignment;
3. four-view crop acquisition between the acquisition and classification lines;
4. bounded asynchronous defect segmentation;
5. conservative multi-view grading;
6. continued tracking with the grade attached to the original track ID.

Windows performance settings live under `windows` in the pipeline JSON:

```json
"windows": {
  "cpu_threads": 8,
  "live_max_fps": 30
}
```

Completed and failed records are saved under `data/results` and appended to `data/logs/tracks.jsonl`. Enable `output.debug_save_images` only when crop/mask inspection is needed.

## Command-line options

```powershell
.\build\windows\Release\inference_runner_windows.exe `
  --data .\data\models `
  --static .\frontend\dist `
  --pipeline-config .\config\pipeline.json `
  --detector-model YOUR_DETECTOR_ID `
  --segmenter-model YOUR_SEGMENTER_ID `
  --python-backend http://127.0.0.1:8091 `
  --host 127.0.0.1 `
  --port 8080
```

Binding to `127.0.0.1` is the safe default. Use `--host 0.0.0.0` only when another trusted device on the local network must reach the dashboard, and allow the port through Windows Firewall yourself.

## API

The React app and native service share port `8080`:

- `GET /api/health` — CUDA/Vulkan/runtime capabilities, GPU, core count, uptime, and active model;
- `GET /api/capabilities` — installed model formats and their selected accelerators;
- `GET /api/models` — model library;
- `POST /api/models` — multipart model import;
- `POST /api/models/:id/activate` — activate a model;
- `DELETE /api/models/:id` — remove a model and its stored files;
- `POST /api/infer` — still-image inference;
- `POST /api/benchmark` — warm-up plus repeated sequential inference through the selected backend;
- `GET /api/pipeline` — live tracks and timing snapshot;
- `POST /api/pipeline/frame` — one live frame;
- `POST /api/pipeline/reset` — clear live tracking state.

Benchmark output includes per-stage mean/median/p95/p99 latency, effective FPS, model size, CPU helper count, peak process working set, and the resolved Vulkan GPU. Use `npm run status:windows` beside a benchmark to observe GPU utilization, VRAM, temperature, and power draw through `nvidia-smi`.

## UI-only development

Run the mock API and Vite in separate PowerShell terminals:

```powershell
npm run mock
npm run dev
```

Vite proxies `/api` calls to the local mock/native service. The frontend does not require cloud services.

## Repository layout

```text
backend/        C++17 API/pipeline, native NCNN engine, and private multi-format Python service
config/         Windows pipeline and GPU model manifest examples
docs/           runtime architecture and implementation decisions
frontend/       React/TypeScript dashboard
models/         source/export model artifacts retained from the app
scripts/        Windows setup, build, launch, and diagnostics
tools/          local UI mock API
```

Raspberry Pi build scripts, ARM compiler flags, Linux service units, sysfs telemetry, and Pi-specific UI/configuration have intentionally been removed from this repository.
