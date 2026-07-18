# Edge Runner

Edge Runner is a Raspberry Pi 4-first model inference application. It uses a small React dashboard for model management and results, while a persistent C++17 service performs image decode, preprocessing, NCNN inference, and postprocessing.

The production Pi runs one process. The C++ executable serves the compiled React assets and the JSON API, so Node.js is only needed while building the frontend.

## Architecture

```text
Browser / React dashboard
        │  same-origin HTTP + multipart uploads
        ▼
C++ edge_runner service
  ├─ model store (disk)
  ├─ one active model (RAM)
  ├─ one inference queue
  ├─ stb image decode + letterbox
  └─ NCNN CPU runtime (ARM NEON / OpenMP / INT8-ready)
```

Why this split:

- React stays out of the inference hot path.
- C++ avoids a Python interpreter and large training-framework dependencies on the Pi.
- NCNN is designed for mobile and embedded CPUs, including ARM NEON and multi-core execution.
- Only the active model is resident, which matters on 2 GB and 4 GB Pi 4 boards.
- Requests are serialized so simultaneous browser requests do not create competing inference thread pools.
- HTTP connections are admitted one at a time; each model may use up to four NCNN compute threads.

The default model input is `640×640`, but input resolution is a model property rather than a hidden performance switch.

## Supported model profiles

The UI imports an NCNN `.param` + `.bin` pair and an explicit profile:

| Profile | Output behavior |
| --- | --- |
| YOLO detection | Decodes common Ultralytics `[4 + classes, predictions]` or `[predictions, 4 + classes]` outputs, then performs class-aware NMS |
| Classification | Returns configurable top-K scores, with optional softmax |
| Raw tensors | Returns each output shape and the first 64 float values for graph inspection |

YOLO detection requires the exact class-label list because the label count defines the output layout. Instance-mask, pose, OBB, and custom output parsing should be added only against a concrete exported model; the runner intentionally does not guess those schemas.

The full profile format is shown in [config/model-manifest.example.json](config/model-manifest.example.json).

## Local UI development on Windows

Install the frontend once:

```powershell
cd frontend
npm install
```

Run these in two terminals from the repository root:

```powershell
npm run mock
```

```powershell
npm run dev
```

Open `http://127.0.0.1:5173`. The mock server is development-only; production inference always uses the C++ service.

Validate the frontend:

```powershell
npm run check
npm run build
```

## Raspberry Pi 4 setup

Use Raspberry Pi OS 64-bit. Good cooling and a stable power supply are important for sustained inference because thermal throttling changes latency more than many small code tweaks.

From the repository root on the Pi:

```bash
bash scripts/setup_pi.sh
bash scripts/build_pi.sh
```

The build script pins NCNN `20260526`, disables Vulkan by default, builds only the CPU runtime, enables OpenMP, and compiles the runner with Cortex-A72 tuning. Vulkan remains an experiment on the Pi 4; benchmark it with your exact model before preferring it over the CPU path.

Run directly:

```bash
./build/pi/edge_runner --data ./data/models
```

Then open `http://PI_ADDRESS:8080`.

Install it as a boot service:

```bash
bash scripts/install_service.sh
```

Useful service commands:

```bash
systemctl status edge-runner
journalctl -u edge-runner -f
sudo systemctl restart edge-runner
```

## Exporting an Ultralytics model

Export on the training computer, not on the Pi:

```bash
yolo export model=best.pt format=ncnn imgsz=640 batch=1
```

Upload the generated `.param` and `.bin` files. Standard Ultralytics NCNN exports normally use `in0` and `out0`; confirm the graph rather than assuming if your exporter or model is custom.

Quantization can reduce model size and sometimes improve CPU latency, but it can also change accuracy and operator behavior. Benchmark FP32 and quantized exports on the actual Pi, then validate both against the same evaluation set before selecting one.

## Runtime API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Engine, architecture, core count, uptime, active model |
| `GET` | `/api/models` | Stored model profiles |
| `POST` | `/api/models` | Import multipart `manifest`, `param`, and `bin` fields |
| `POST` | `/api/models/:id/activate` | Load one model and release the previous model |
| `DELETE` | `/api/models/:id` | Remove a model from disk |
| `POST` | `/api/infer` | Infer multipart `image`, `modelId`, `confidence`, and `iou` fields |

Model uploads are capped at 256 MB and input images at 25 MB. Models are executable compute graphs: only import trusted files, especially on a device reachable from your LAN.

## Performance knobs that matter

- Start with four inference threads on a Pi 4, then benchmark two versus four for the specific graph.
- Keep batch size at one for interactive camera or image latency.
- Use a smaller model architecture before reducing input size when spatial detail matters.
- Keep only one active model. Switching is intentionally explicit.
- Use 64-bit Raspberry Pi OS so the runtime can use the Pi 4's AArch64 path.
- Measure end-to-end latency. The dashboard reports decode, preprocessing, model compute, postprocessing, and total time separately.
- Prefer wired Ethernet or local display if model uploads are large; network speed does not affect inference after import.

## Source layout

```text
frontend/        React + TypeScript dashboard
backend/         C++ HTTP service and NCNN engine
config/          model-profile example
deploy/          systemd unit template
scripts/         Pi setup, build, and installation
tools/           development-only mock API
```
