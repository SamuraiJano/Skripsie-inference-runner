# Windows multi-backend architecture

## Decision

The public application remains a C++17 service with the existing React/TypeScript dashboard and conveyor pipeline. Model execution is selected per artifact:

- detected NCNN export packages run in-process through the native Vulkan engine;
- compatible artifacts discovered below `models/` run through a private Ultralytics service bound only to `127.0.0.1`;
- pinned Grounding DINO and LocateAnything directories run through dedicated Transformers backends in the private service;
- the private service delegates to PyTorch, ONNX Runtime, TensorRT, or another installed export runtime and reports the actual accelerator.

This preserves the low-level NCNN path while allowing a normal `.pt` model—or another supported export—to be used directly from the filesystem without conversion or an import copy.

## Hardware mapping

| Laptop resource | Application use |
| --- | --- |
| RTX 4060 Laptop GPU / CUDA | PyTorch `.pt`, TorchScript, ONNX Runtime GPU, Grounding DINO, LocateAnything, and optional TensorRT |
| RTX 4060 Laptop GPU / Vulkan | native NCNN models |
| Ryzen 7 7735HS | image decode, tracking, native post-processing, HTTP, file I/O, and optional CPU runtimes |
| 32 GB RAM | loaded models, decoded frames, queues, and result data |
| Windows power plan and GPU preference | optional `-MaxPerformance` launcher policy |
| `nvidia-smi`, `vulkaninfo`, PowerShell/CIM | setup validation and live diagnostics |

No CUDA Toolkit is needed for the tested PyTorch and ONNX wheels. TensorRT has its own versioned CUDA requirements and is exposed only when its runtime imports successfully.

## Request path

```text
Webcam / video / image
        |
        v
React capture and grouped discovered-model library
        |
        v
C++ HTTP API + persistent pipeline state
        |
        +-- NCNN package ----------> native NCNN / Vulkan --------+
        |                                                       |
        +-- direct artifact -----> loopback model service       |
                                      |                         |
                                      +-> PyTorch / CUDA         |
                                      +-> ONNX Runtime / CUDA    |
                                      +-> TensorRT / CUDA        |
                                      +-> Transformers grounding |
                                      +-> optional CPU runtime   |
                                                                v
                                         normalized result JSON
```

Both engines implement the same C++ `InferenceEngine` result contract. Detection boxes, class scores, segmentation masks, and timing therefore continue into the same workbench, tracker, crop scheduler, multi-view grader, and benchmark endpoints.

## Capability and discovery contract

The private service reports installed modules and execution providers through `/health`; the C++ API merges those with native NCNN support in `/api/capabilities`. It recursively discovers supported single-file artifacts and complete export directories below `models/`, skipping hidden caches and collapsing package sidecars and checkpoint shards into one record.

Discovery metadata is cached by artifact size and modification time. PyTorch tensors, Safetensors headers, export metadata, and NCNN metadata are inspected to derive the task, input dimensions, labels, on-disk precision, and quantization mode. Unknown precision is reported honestly. Stable IDs are derived from relative paths, so active-model persistence survives restarts and file timestamps changing.

The C++ process accepts only discovered paths that resolve inside the configured model root. Activation then loads and warms the selected artifact on its chosen device. The `models/` directory is therefore the executable-checkpoint trust boundary; the public API has no upload or delete operation.

## Process and concurrency model

`run_windows.ps1` starts the model service first, waits for CUDA/runtime health, then starts the public C++ service. The model process is hidden, loopback-only, and stopped with the application. The C++ model store serializes inference across active and pipeline engines, so both native and delegated models preserve the pipeline’s detector-priority ordering.

Model artifacts remain under `models/`. The active model, results, and other runtime state remain under `data/`, while logs from the private service are under `data/logs/`. Raspberry Pi services, ARM flags, sysfs telemetry, and Linux deployment files are intentionally absent.
