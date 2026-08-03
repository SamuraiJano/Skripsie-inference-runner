# Windows multi-backend architecture

## Decision

The public application remains a C++17 service with the existing React/TypeScript dashboard and conveyor pipeline. Model execution is selected per artifact:

- verified NCNN packages run in-process through the native Vulkan engine;
- compatible direct artifacts run through a private Ultralytics service bound only to `127.0.0.1`;
- the private service delegates to PyTorch, ONNX Runtime, TensorRT, or another installed export runtime and reports the actual accelerator.

This preserves the low-level NCNN path while allowing a normal `.pt` model—or another supported export—to be imported without conversion.

## Hardware mapping

| Laptop resource | Application use |
| --- | --- |
| RTX 4060 Laptop GPU / CUDA | PyTorch `.pt`, TorchScript, ONNX Runtime GPU, and optional TensorRT |
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
React capture and model library
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
                                      +-> optional CPU runtime   |
                                                                v
                                         normalized result JSON
```

Both engines implement the same C++ `InferenceEngine` result contract. Detection boxes, class scores, segmentation masks, and timing therefore continue into the same workbench, tracker, crop scheduler, multi-view grader, and benchmark endpoints.

## Capability and import contract

The private service reports installed modules and execution providers through `/health`; the C++ API merges those with native NCNN support in `/api/capabilities`. The frontend derives its accepted file extensions from that response.

Direct artifacts use generated schema-version-3 metadata. Import is transactional:

1. copy into a temporary model directory;
2. atomically move it into local storage;
3. resolve the format and runtime;
4. load and warm the artifact on the selected device;
5. persist detected task, labels, format, runtime, and accelerator;
6. remove the stored copy if validation fails.

Directory-based exports are uploaded as ZIP archives. Extraction rejects absolute paths, parent traversal, more than 4,096 entries, and expanded content above 4 GB. The model service accepts paths only beneath its configured model root. Executable checkpoint formats require a trusted-source acknowledgement.

Verified NCNN packages continue to use schema version 2, provenance hashes, precision flags, named blobs, and the original three-file import contract.

## Process and concurrency model

`run_windows.ps1` starts the model service first, waits for CUDA/runtime health, then starts the public C++ service. The model process is hidden, loopback-only, and stopped with the application. The C++ model store serializes inference across active and pipeline engines, so both native and delegated models preserve the pipeline’s detector-priority ordering.

Persistent models and result state remain under `data/`, while logs from the private service are under `data/logs/`. Raspberry Pi services, ARM flags, sysfs telemetry, and Linux deployment files are intentionally absent.
