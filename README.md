# Inference Runner Windows

Inference Runner Windows is the Windows/RTX edition of the local inference app. It keeps the React dashboard, still-image workflow, benchmarks, and live conveyor pipeline while replacing the Raspberry Pi deployment profile with a capability-driven Windows model layer. A model is routed to PyTorch/CUDA, ONNX Runtime CUDA, TensorRT, another installed Ultralytics runtime, or the native NCNN/Vulkan engine according to its format.

The application is local-first: model files, frames, predictions, benchmark samples, and conveyor results stay on this computer.

For a source-by-source explanation of the Python algorithms, React/C++ HTTP bridge, live tracking, 20/40/60/80 capture scheduler, grading state machine, and a recommended reconstruction order, see [Rebuilding Inference Runner from scratch](docs/rebuild-from-scratch.md).

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
- `npm run setup:grounding-dino` downloads and validates the pinned Grounding DINO Tiny checkpoint for offline CUDA inference.
- `scripts/build_inference_runner_windows.ps1` builds the React bundle, downloads the pinned NCNN 20260526 source when needed, builds NCNN with Vulkan enabled, compiles the C++ service with MSVC/AVX2, and runs the native tests.
- `scripts/run_windows.ps1` starts the private loopback model service and the C++ application together, then stops both on exit.
- `scripts/status_windows.ps1` reports NVIDIA utilization, VRAM, temperature, power draw, Windows power mode, process state, and API health.

Build only part of the project when iterating:

```powershell
# Frontend only
npm run check
npm run build

# Reuse already-built NCNN and skip npm work
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_inference_runner_windows.ps1 -SkipFrontend -SkipNcnn
```

The production executable is written to:

```text
build/windows/Release/inference_runner_windows.exe
```

## Models and automatic backend selection

The project-level `models/` directory is the model library. The runner scans it recursively at startup, whenever the dashboard refreshes, and when **Rescan folder** is selected. There is no upload or add-model step: copy a trusted model file or complete export directory below `models/` and refresh the library.

Packages are collapsed into one library item. For example, a Hugging Face directory containing multiple Safetensors shards is one model, as is an NCNN directory containing its `.param`, `.bin`, and `metadata.yaml` files. Cache, asset, and sidecar files are not shown as models.

Availability is runtime-driven and returned by `GET /api/capabilities`. On this laptop the base setup provides:

| Artifact | Runtime | Expected device |
| --- | --- | --- |
| `.pt` | PyTorch | RTX 4060 / CUDA |
| `.torchscript` | PyTorch | RTX 4060 / CUDA |
| `.onnx` | ONNX Runtime GPU | RTX 4060 / CUDA |
| `.engine` | TensorRT, when installed | RTX 4060 / CUDA |
| NCNN export directory with `.param` + `.bin` | native NCNN | RTX 4060 / Vulkan |
| Grounding DINO Hugging Face directory | Hugging Face Transformers | RTX 4060 / CUDA |
| LocateAnything Hugging Face directory | NVIDIA Transformers worker | RTX 4060 / CUDA + CPU offload |
| OpenVINO/SavedModel/Paddle export directory, `.zip`, `.pb`, `.tflite`, `.mnn`, `.pte` | corresponding optional runtime | shown in the UI when installed |

“Compatible” means that the artifact maps to a task the application can render, including detection, instance segmentation, classification, and visual grounding. Hardware-specific formats still require their runtime and must have been exported for a compatible device/runtime version. For example, TensorRT engines are not portable across every GPU and TensorRT version.

The library reports the detected on-disk weight format separately from the runtime. PyTorch precision is inspected from checkpoint tensors, Safetensors precision from tensor headers, and NCNN quantization from package metadata. Mixed packages are retained as such (for example, `INT8 + FP32 hybrid`) rather than being flattened to a generic INT8 label. Formats without reliable embedded precision metadata are marked `Unknown` instead of being guessed.

The `models/` directory is a trust boundary. PyTorch-style checkpoints (`.pt`, `.torchscript`, and `.pte`) can contain executable content, and LocateAnything uses trusted local model code. Place only models from sources you trust in this directory. Discovery does not load a model onto the GPU; activation performs runtime validation and warm-up.

Automatically detected NCNN models use the highest-scoring Vulkan device by default and derive their runtime precision flags from `metadata.yaml`. The equivalent explicit runtime configuration is:

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

Use `gpuDevice: -1` to let NCNN choose its highest-scoring Vulkan device. On this laptop, device `0` is the discrete RTX 4060 and device `1` is the integrated AMD GPU. Model files remain in `models/`; only the active model ID is persisted in `data/models/active-model.txt`.

## Grounding DINO Tiny

The runner includes the official Transformers implementation of [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny). Install its pinned checkpoint once:

```powershell
npm run setup:grounding-dino
npm run start:windows
```

The setup command downloads only the Safetensors checkpoint and processor files (about 657 MiB) to `models/GroundingDINO-Tiny`. It is detected automatically after startup or a folder rescan. Activations remain local-only and cannot fetch files from the model hub.

Grounding DINO accepts comma-separated categories or short referring phrases and returns scored bounding boxes. The workbench and video screen expose its box-confidence threshold; the text threshold remains pinned at `0.25`. The integration runs the checkpoint's native FP32 weights on CUDA and pins the model repository to revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. The checkpoint is Apache-2.0 licensed.

## NVIDIA Locate Anything

The runner includes an optional integration for [NVlabs/Eagle LocateAnything](https://github.com/NVlabs/Eagle/tree/main/Embodied). Install its pinned worker, Python dependencies, and checkpoint once:

```powershell
npm run setup:locate-anything
npm run start:windows
```

The setup command installs the pinned `nvidia/LocateAnything-3B` snapshot (about 7.8 GB) in `models/LocateAnything-3B`, where it is detected automatically. The dashboard never downloads checkpoint files; activation uses the project-local snapshot with Hugging Face offline loading.

The single-image workbench supports:

- category detection;
- single- and multi-instance phrase grounding;
- scene-text detection and specific-text grounding;
- GUI box/point grounding;
- general point localization;
- fast, hybrid, and slow decoding modes.

LocateAnything returns generated locations rather than calibrated detector confidences. Its boxes are therefore shown as `BOX`, and point results are rendered as crosshairs. The raw token answer remains available below the structured results for research inspection.

The integration pins the GitHub worker to commit `bb860a7efa22bb40cd4bb77959512375b92db934` and the model repository to revision `c32291ca5e996f5a7a485845b4f57a233936bba0`. The RTX 4060 profile reserves GPU headroom and automatically offloads overflow layers to system RAM. Close other GPU-heavy applications before activation. NVIDIA officially lists Linux as the supported operating system for this checkpoint, so this Windows path is best-effort and should be validated against your skripsie dataset.

Important: the released LocateAnything-3B checkpoint is licensed for non-commercial academic and non-profit research use, not general commercial use. The model also executes pinned custom code; only proceed if those terms and the NVIDIA source are acceptable for your project.

## Video object detection and grounding

Open the **Video detection** screen from the top navigation, choose a local video, and select an installed object-detection or visual-grounding model. The browser submits only one frame at a time, so requests cannot build up. Conventional detectors follow video playback; Grounding DINO and LocateAnything pause playback and seek through the file at the selected sample interval so every grounding result stays aligned with the frame shown.

For fruit detection, enter `fruit`, or use a more explicit comma-separated list such as `apple, banana, orange, pear`. LocateAnything exposes all eight grounding tasks plus fast, hybrid, and slow decoding; Grounding DINO provides its focused named-category detector and scored boxes. The screen accepts any video format that Microsoft Edge can decode; backend and GPU selection remain automatic.

## Live conveyor pipeline

Copy the example before changing deployment values:

```powershell
Copy-Item .\config\pipeline.example.json .\config\pipeline.json
```

The browser accepts a webcam or video file, JPEG-encodes frames locally, and sends at most one frame request at a time. The configured pipeline performs:

1. YOLO26n fruit detection;
2. normalized top-to-bottom ROI and bounding-box plausibility filtering;
3. ByteTrack association with separate low, high, and new-track confidence thresholds;
4. tentative confirmation from repeated high-confidence hits and downward motion;
5. one crop per confirmed fruit at 20%, 40%, 60%, and 80% of the acquisition route; during a detector dropout, ByteTrack's projected box continues to advance and supply these crops under the same track ID until reacquisition or the configured miss buffer expires;
6. bounded asynchronous defect segmentation and conservative multi-view grading;
7. one-time completion/counting at the lower classification boundary.

The `zones` block controls the normalized left/right/top/bottom ROI plus entry and exit-zone heights. The `tracking` block controls minimum box ratios, confirmation hits, motion/stationary tolerances, optional strict entry gating, and debug diagnostics. Keep `tracking.debug_diagnostics` off during normal operation; when enabled, tentative/rejected candidates and their filter reasons are exposed in the dashboard snapshot.

Windows performance settings live under `windows` in the pipeline JSON:

```json
"windows": {
  "cpu_threads": 8,
  "live_max_fps": 30
}
```

Completed records and confirmed tracks that later fail classification are saved under `data/results` and appended to `data/logs/tracks.jsonl`. Rejected tentative flickers are not written as fruit results. Enable `output.debug_save_images` only when crop/mask inspection is needed.

## Command-line options

```powershell
.\build\windows\Release\inference_runner_windows.exe `
  --data .\data\models `
  --models .\models `
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
- `GET /api/models` — rescan and return the automatically detected model library;
- `POST /api/models/rescan` — explicit folder rescan;
- `POST /api/models/:id/activate` — activate a model;
- `POST /api/infer` — still-image inference; visual-grounding requests additionally accept `prompt`, `groundingMode`, and `generationMode` multipart fields;
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
models/         recursively scanned source/export model library
scripts/        Windows setup, build, launch, and diagnostics
tools/          local UI mock API
```

Raspberry Pi build scripts, ARM compiler flags, Linux service units, sysfs telemetry, and Pi-specific UI/configuration have intentionally been removed from this repository.
