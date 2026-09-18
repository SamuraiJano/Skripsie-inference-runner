# Rebuilding Inference Runner from scratch

This guide explains the design decisions, application-owned algorithms, model-runtime adapters, HTTP contracts, and React integration in this repository. It is intended to be read beside the source rather than as a replacement for it.

## 1. The most important architectural fact

React does not run or import Python. The processes communicate over HTTP, and the current application has two server layers:

```text
Browser / React
    |
    | same-origin HTTP: /api/*
    v
C++ service on 127.0.0.1:8080
    |-- owns model selection and the public API
    |-- owns conveyor tracking, capture scheduling, queues, and grading
    |-- runs native NCNN/TFLite/TensorRT engines where configured
    |
    | private HTTP: /models/*
    v
Python model service on 127.0.0.1:8091
    |-- discovers and inspects model artifacts
    |-- loads PyTorch/Ultralytics/Transformers runtimes
    |-- runs one-image inference
    `-- converts backend-specific results to one JSON schema
```

In development, Vite serves React on port 5173 and proxies `/api` to the C++ service. In production, the C++ service serves the compiled React files and `/api` on the same port. The browser therefore never needs the Python address and never needs CORS for the normal application.

Relevant source:

- [`frontend/vite.config.ts`](../frontend/vite.config.ts): development proxy;
- [`frontend/src/api.ts`](../frontend/src/api.ts): browser HTTP client;
- [`backend/src/main.cpp`](../backend/src/main.cpp): public HTTP routes;
- [`backend/src/EngineFactory.cpp`](../backend/src/EngineFactory.cpp): selects native or Python-backed execution;
- [`backend/src/PythonEngine.cpp`](../backend/src/PythonEngine.cpp): private C++-to-Python bridge;
- [`backend/python/model_service.py`](../backend/python/model_service.py): Python HTTP/model service;
- [`scripts/run_windows.ps1`](../scripts/run_windows.ps1): starts Python, waits for it, then starts C++.

### Why it was split this way

- The browser stays a presentation and image-capture client. It does not own durable tracker state.
- C++ receives frames in a strict order and owns the real-time conveyor state machine.
- Python makes it practical to use PyTorch, Ultralytics and Transformers without embedding their runtimes in C++.
- Native models can still avoid Python while returning the same `InferenceResult` contract.
- The public server is the single trust boundary. Python is private and loopback-only.

If reproducing the architecture exactly is not a requirement, a simpler first version can use React -> FastAPI -> Python directly. The tracker, capture scheduler and grading state must then be ported from C++ into that Python service; they must not be kept in React state.

## 2. Build order that minimizes rework

Implement the system in these stages:

1. Define the JSON types and return hard-coded responses.
2. Build single-image React upload and overlay rendering.
3. Implement one Python model loader and `/infer` route.
4. Add model discovery and activation only after inference works.
5. Add video sampling with one request in flight.
6. Add a stateful tracker using synthetic detection sequences.
7. Add route progress and bounded dense per-track capture.
8. Add crop quality gates and the bounded segmentation queue.
9. Add background spheroid fitting, texture alignment and surface fusion.
10. Add grading, persistence and timeout behavior.
11. Optimize or add more runtime formats last.

This order keeps model/runtime problems separate from tracker/state-machine problems.

## 3. The shared data contract

All engines normalize their output to the interfaces in [`frontend/src/types.ts`](../frontend/src/types.ts). TypeScript does not validate server data at runtime, so the Python/C++ serializers and TypeScript interfaces must change together.

The core result is conceptually:

```ts
interface InferenceResult {
  modelId: string
  task: 'classification' | 'yolo-detect' | 'yolo-segment' | 'vision-grounding' | 'raw'
  sourceWidth: number
  sourceHeight: number
  detections: Array<{
    classId: number
    label: string
    confidence: number
    x: number
    y: number
    width: number
    height: number
    trackId: number | null
    mask: MaskRle | null
  }>
  points: Array<{ classId: number; label: string; x: number; y: number }>
  classifications: Array<{ classId: number; label: string; score: number }>
  timing: TimingBreakdown
}
```

Coordinates are source-image pixels. Boxes use `(x, y, width, height)`, where `(x, y)` is the top-left corner. Keeping this invariant prevents every React overlay from needing backend-specific coordinate logic.

## 4. Python algorithms and adapters

The neural-network mathematics is primarily implemented by Ultralytics, PyTorch and Transformers. The algorithms below are the application-owned Python logic that surrounds those libraries.

### 4.1 Model format detection

Location: `detect_format()` in [`model_service.py`](../backend/python/model_service.py).

Algorithm:

1. For a file, map its extension to a runtime format (`.pt`, `.pth`, `.onnx`, `.engine`, `.tflite`, and so on).
2. For a directory, inspect `config.json` and well-known marker files such as `saved_model.pb`, `.xml`, `.param`, and `.pdmodel`.
3. Treat model ZIP files as archives that must be expanded before loading.
4. Reject artifacts that cannot be identified instead of guessing a runtime.

The important design choice is that a directory export is one logical model even when it has multiple files.

### 4.2 Recursive model discovery

Locations: `discover_model_candidates()`, `_model_files()`, `_candidate_fingerprint()`, `inspect_discovered_model()`, and `discover_models()`.

Algorithm:

1. Walk the configured `models/` tree in a deterministic sorted order.
2. Skip hidden directories, Python caches, and offload caches.
3. When a complete directory export is detected, add that directory and stop descending into it.
4. Inspect single-file artifacts by supported suffix.
5. Create a stable model ID from a slug plus the first eight hexadecimal digits of `SHA-256(lowercase relative path)`.
6. Cache inspection metadata using `(sum of relevant file sizes, latest modification time)`.
7. Re-read runtime availability even on a metadata-cache hit.
8. Sort the UI list by family, display name, and relative path.

This gives stable IDs across restarts while avoiding repeated parsing of multi-gigabyte artifacts.

### 4.3 Metadata and precision inspection

Locations: `_precision_details()`, `_safetensors_dtypes()`, `_yaml_metadata()`, `_inspect_*()`.

The service inspects each format differently:

| Format | Inspection method |
| --- | --- |
| PyTorch `.pt` | Load checkpoint on CPU, inspect model arguments, names and parameter dtypes |
| D-FINE `.pth` | Read the tensor-only state dict, validate Nano topology, derive labels and dtypes |
| TorchScript | Load on CPU and read the optional embedded `config.txt` |
| ONNX | Read graph initializers and metadata without external tensor data |
| TensorRT engine | Read the exporter metadata prefix embedded by the export format |
| Safetensors packages | Read the length-prefixed JSON headers and sum bytes by dtype |
| NCNN | Parse YAML metadata plus `.param` input/output blob names |
| Other directory exports | Read conservative metadata/filename hints and report unknowns honestly |

Precision selection is based on the tensor type that occupies the most bytes. Explicit 4-bit or 8-bit quantization configuration takes priority. Integer plus floating weights are reported as a hybrid; the code does not pretend a mixed model is entirely INT8.

Security note: general PyTorch inspection uses `torch.load(..., weights_only=False)` because Ultralytics checkpoints may contain model objects. A `.pt` file is executable/trusted input in this application. Only place trusted artifacts inside `models/`.

### 4.4 Safe archive extraction

Location: `extract_archive()`.

Algorithm and limits:

- at most 4,096 entries;
- at most 4 GiB expanded size;
- resolve every target and verify that it remains inside the staging directory;
- extract into a staging directory;
- rename staging to the final cache only after success;
- write a `.ready` marker so the next scan can reuse it.

This prevents `../` path traversal and avoids treating a half-extracted model as valid.

### 4.5 Runtime capability probing

Locations: `format_capabilities()` and `onnx_cuda_available()`.

Importing a module does not prove that it can execute on the GPU. For ONNX Runtime, the service:

1. verifies that `CUDAExecutionProvider` is listed;
2. builds a one-node ONNX Identity graph in memory;
3. creates a session with CUDA first and CPU second;
4. checks that CUDA is the actual first provider.

The capability table is cached because probing runtimes and GPU providers is relatively expensive.

### 4.6 Model registry, reference counting and serialization

Location: `ModelRegistry`.

The registry maps `modelId -> LoadedModel` and protects the map and model calls with an `RLock`.

- `load()` validates IDs and paths, selects the runtime, loads the model and performs warm-up inference.
- Loading the same ID/path increments a reference count instead of allocating a second model.
- `unload()` decrements the count and releases Python/CUDA memory only at zero.
- `infer()` runs while holding the same lock, avoiding races in runtime objects and GPU memory.

Every requested model path must resolve inside `models_root`. The API cannot use an arbitrary filesystem path.

Warm-up is also validation: activation fails immediately if the artifact/task/runtime combination is incompatible rather than failing later in a live conveyor run.

### 4.7 Generic Ultralytics inference

Location: `ModelRegistry.infer()`.

Algorithm:

1. Decode encoded bytes with Pillow and convert to RGB NumPy HWC data.
2. Clamp confidence and IoU to `[0, 1]`.
3. Call `model.predict(source=image, device=..., imgsz=(height, width), conf=..., iou=...)`.
4. Convert `xyxy` tensors to source-pixel `(x, y, width, height)` objects.
5. Attach a mask to the matching detection for segmentation tasks.
6. For classification, return the top five probabilities.
7. Copy Ultralytics preprocess/inference/postprocess timing into the common timing object.

Ultralytics owns resizing, neural inference, task-specific decoding and its NMS in this path. This application owns input validation and result normalization.

### 4.8 Segmentation-mask run-length encoding

Location: `rle_mask()` in Python and `MaskOverlay` in [`InferenceWorkbench.tsx`](../frontend/src/components/InferenceWorkbench.tsx).

Sending a full JSON boolean matrix is wasteful. The service instead:

1. resizes the probability mask to the original image dimensions with nearest-neighbor interpolation;
2. thresholds it to `0/1`;
3. flattens pixels in row-major order;
4. emits alternating run lengths, always starting with a background run.

Example:

```text
pixels:  0 0 1 1 1 0 1
counts:  [2, 3, 1, 1]
meaning: 2 background, 3 foreground, 1 background, 1 foreground
```

If the first pixel is foreground, the first background count is `0`. React alternates the `foreground` flag after every count and paints only foreground runs into a transparent `ImageData` canvas.

### 4.9 D-FINE checkpoint compatibility conversion

Location: [`backend/python/dfine_checkpoint.py`](../backend/python/dfine_checkpoint.py).

This module is a compatibility adapter, not a new detection algorithm:

1. Load tensor weights using `weights_only=True`.
2. Prefer EMA weights, then `model`, then `state_dict`.
3. Validate that the checkpoint matches the supported D-FINE Nano topology.
4. Rename official training-code parameter keys to Transformers parameter keys with regex rules.
5. Split fused attention `in_proj_weight`/`in_proj_bias` into separate query, key and value projections.
6. duplicate shared decoder heads under the names registered by Transformers;
7. load with `strict=True` so missing or unexpected parameters fail activation;
8. use `RTDetrImageProcessor` for 640 x 640 preprocessing and source-size postprocessing.

During inference it uses `torch.inference_mode()`, runs the model, calls `post_process_object_detection()`, clamps boxes to source bounds, and maps label IDs to names.

### 4.10 Grounding DINO

Locations: `_load_grounding_dino()` and `_infer_grounding_dino()`.

- The checkpoint and revision are pinned and loaded with `local_files_only=True`.
- Only named-category detection is supported.
- Input text is split on commas or full stops, whitespace-normalized, lowercased, limited to 64 categories and 1,000 characters.
- The processor jointly encodes image and text.
- `post_process_grounded_object_detection()` applies both a box threshold and a text threshold and restores source-pixel boxes.
- Returned text labels receive stable class IDs within that request.

The actual open-vocabulary transformer algorithm belongs to the Grounding DINO implementation; this code defines the safe input policy and normalizes its output.

### 4.11 LocateAnything

Locations: `_load_locate_anything_worker()`, `_load_locate_anything()`, `_infer_locate_anything()`, and `parse_locate_anything_answer()`.

The model/source revisions and required local files are pinned. Depending on available VRAM, the loader either keeps the model on CUDA or uses Accelerate CPU offload with memory limits. Inference supports detection, single/multiple grounding, scene text, GUI boxes/points, and pointing modes.

The model returns tagged text rather than this application's JSON boxes. The parser:

1. finds `<ref>label</ref>` and following box/point tags with regular expressions;
2. reads model coordinates normalized to the integer range `0..1000`;
3. converts them to source pixels with `x = token_x / 1000 * width` and `y = token_y / 1000 * height`;
4. clamps coordinates and orders box corners;
5. emits the same detection/point schema as every other backend.

### 4.12 Python HTTP service

Location: `ServiceHandler` and `main()`.

Private routes:

| Method | Route | Body | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | none | runtime and accelerator capabilities |
| GET | `/models/discover` | none | inspected model records |
| POST | `/models/load` | JSON | load/warm a model and increment its reference |
| POST | `/models/unload` | JSON | decrement/release a model |
| POST | `/models/{id}/infer?...` | raw encoded image bytes | run one inference |

Request bodies have explicit size limits and responses use compact JSON with non-finite numbers rejected. `ThreadingHTTPServer` can accept concurrent HTTP requests, while the model registry lock serializes model execution.

Python's standard `http.server` is suitable here only because the service is private and bound to loopback. For a new direct React -> Python application, use a production-oriented ASGI stack such as FastAPI/Uvicorn.

## 5. Conveyor algorithms: these are C++, not Python

ByteTrack identity, dense view collection, dropout prediction, crop quality gates, segmentation queues and final naartjie grade are implemented in [`backend/src/Pipeline.cpp`](../backend/src/Pipeline.cpp), with configuration in [`config/pipeline.example.json`](../config/pipeline.example.json). Numerical surface reconstruction is kept in one module, [`backend/python/surface_mapping.py`](../backend/python/surface_mapping.py), and runs through the existing private Python worker.

### 5.1 Detection filtering and new-track gate

Before tracking, the pipeline rejects:

- the wrong class ID;
- non-finite or zero-area boxes;
- boxes outside a tolerance-expanded ROI;
- boxes below maintenance size or above maximum size.

A detection may maintain an existing track inside the relaxed boundary. A new track is eligible only when it passes strict ROI/size checks and, if configured, is in the entry zone. This prevents a partially visible fruit near the exit from appearing as a new fruit.

### 5.2 Live ByteTrack-style association and prediction

The live pipeline currently uses `ByteTracker` in `Pipeline.cpp`. It is a deliberately small ByteTrack-style implementation, not a byte-for-byte copy of the official tracker:

1. Before association, predict each box from its last real observation and smoothed velocity.
2. Split detections into high and low confidence groups.
3. Greedily associate the maximum-IoU pairs against high-confidence detections.
4. Repeat association for unmatched tracks using low-confidence detections. This is the central ByteTrack idea: low-score boxes can recover real occluded objects.
5. Smooth velocity with

   ```text
   v := 0.60 * previous_v + 0.40 * measured_v
   measured_v := (new_position - last_real_position) / frame_gap
   ```

6. If unmatched, increment `missed` but return the predicted box while `missed <= max_missed_frames`.
7. Start new tracks only from unmatched high detections above `new_track_threshold` and allowed by the entry gate.
8. Delete a track only after its miss buffer expires.

That predicted output is the mechanism that preserves one track ID through a detector dropout. `has_observation` remains false for predicted frames, so predictions can continue route progress and produce a scheduled crop without being mistaken for real confirmation evidence.

The repository also contains a fuller reusable `ByteTrack` in [`backend/src/ByteTrack.cpp`](../backend/src/ByteTrack.cpp), used by `TrackingPipeline`. It uses Hungarian minimum-cost matching with

```text
cost = iou_weight * (1 - IoU(predicted_box, detection_box))
     + (1 - iou_weight) * normalized_centroid_distance
```

plus class-awareness, direction gates, confidence stages, velocity drag and track age/buffer limits. Do not accidentally assume this second implementation is the one instantiated by `MultiViewPipeline`; the live surface-mapping flow uses the smaller `ByteTracker` described above.

### 5.3 Track confirmation and false-positive rejection

A tentative track becomes confirmed only after:

- enough real high-confidence detections; and
- net downward motion over a bounded centroid-history window.

Sustained upward motion is rejected. Nearly unchanged vertical position increments a stationary counter, and long-lived stationary features are rejected. Predicted positions never increase confirmation hits or update the real-motion history.

### 5.4 Route progress

The acquisition route excludes the entry and exit zones. Using the tracked box center `cy`:

```text
start = roi_top + entry_zone_height
end   = roi_bottom - exit_zone_height
progress = clamp((cy - start) / (end - start), 0, 1)
```

The variable names in the helper use `x`, but the live conveyor passes vertical `center_y` values. Route direction is top to bottom.

### 5.5 Dense bounded view collection

When `surface_mapping.enabled` is true, a confirmed fruit is sampled every `candidate_frame_stride` eligible frames while it is inside the acquisition ROI. `maximum_candidate_views` is a hard memory limit, and `maximum_mapping_views` limits the views entering the numerical search. Completion never depends on reaching a configured view count; it depends on estimated angular coverage, gap size, observed area and alignment confidence.

The former route-percentage scheduler remains only as a compatibility path when surface mapping is disabled.

Configuration:

```json
"capture": {
  "target_capture_percentages": [0.2, 0.4, 0.6, 0.8],
  "target_views": 4
}
```

In that disabled path, `next_capture_target()` scans targets in order and returns the first uncaptured target for which `progress >= target`.

A capture is accepted only if:

- the minimum frame gap since the previous view has elapsed;
- the predicted or detected box is large enough;
- overlap with another tracked fruit is below the configured fraction;
- the clamped crop encodes successfully;
- the crop dimensions remain large enough; and
- Laplacian variance is above the sharpness threshold.

If the detector is temporarily lost, the tracker-predicted box follows the same bounded collector. The saved view is marked `usedTrackingPrediction: true`. When detection resumes, association restores the observation to the same track instead of starting a new sequence.

The guarantee is therefore “at most one image per configured target and continued opportunities during a bounded tracking loss,” not “always exactly four images under every physical condition.” A track can still have fewer than four if it is never confirmed, expires longer than the miss buffer, moves too far between submitted frames, or every eligible crop fails overlap/sharpness/size/queue constraints.

For enabled mapping, each accepted record instead retains frame/time, the original box, crop-to-frame translation, native-aspect-ratio JPEG, confidence, focus score, segmentation instances and whether it used a tracker prediction. Rejected candidate reasons are retained in result metadata.

### 5.6 Cropping and sharpness

`clamp_crop()` expands a box by `crop_margin`, floors the top/left, ceils bottom/right, clamps all sides to the image, and prevents negative dimensions.

Sharpness is the variance of a four-neighbour Laplacian over grayscale pixels:

```text
gray = 0.299 R + 0.587 G + 0.114 B
L(x,y) = left + right + top + bottom - 4 * center
sharpness = mean(L^2) - mean(L)^2
```

Low variance indicates a flat/blurred crop. The crop is JPEG encoded and persisted before it is submitted for segmentation.

### 5.7 Bounded asynchronous segmentation

Detection/tracking remains on the frame path. Accepted crops become jobs containing generation, track, frame and view IDs. A bounded FIFO queue feeds segmentation workers.

- Bounded capacity prevents unbounded RAM growth.
- IDs allow late results to be routed to the correct view.
- A generation ID makes results from a reset obsolete.
- Duplicate/stale results are ignored.
- Queue overflow marks that view as dropped instead of blocking the detector forever.

### 5.8 Surface fitting, alignment and fusion

After a track reaches the classification boundary (or a confirmed track times out), a dedicated surface worker sends paths and segmentation RLEs to the private Python `/surface-map` endpoint. This is separate from both the detector loop and segmentation workers.

The mapper performs these stages:

1. Run GrabCut on every native-resolution crop; retain the largest component, fill holes and close small breaks.
2. Reject boundary-touching, implausible, disconnected, blurred, low-confidence or geometrically discontinuous views.
3. Fit covariance ellipses. Use `A = mean(2 sqrt(lambda_max))` and `B = min(2 sqrt(lambda_min))`; switch to a sphere when `(A-B)/A` is small.
4. Downsample the green channel, subtract a Gaussian blur (`sigma=1.25`), select strong safe-interior texture points, and search constrained Rodrigues vectors `(rx, ry, 0)`.
5. Refine the discrete minimum with a local quadratic only when its Hessian is positive definite and well conditioned. Textureless or weak pairs remain unreliable. One bad intermediate view may be bypassed only inside the maximum angular gap.
6. Accumulate `G[i+1] = R[i->i+1] @ G[i]`, normalize drift with SVD, propagate the reference fruit basis, and calculate unwrapped centre longitudes.
7. Use vectorized NumPy and `cv2.remap()` to project RGB, per-class probability, confidence and observed masks to a latitude-limited longitude map. At most the two bracketing views contribute to a map point.
8. Weight defect areas by the spheroid surface Jacobian. A three-width connected-component tile joins regions across the longitude seam without counting duplicated pixels.

The geometry follows [Albiol et al. 2021](https://doi.org/10.3390/s21062232) and the fusion follows [Albiol et al. 2022](https://doi.org/10.3390/s22145452). A black RGB pixel is not automatically unknown; the authoritative indicator is `surface_observed_mask.png`. Unknown map locations are never mirrored or inpainted.

### 5.9 Final grading

Each successful view produces a class and confidence. The aggregator:

1. rejects failed, invalid, non-finite or below-threshold view results;
2. requires the configured minimum number of confident views;
3. selects the maximum zero-based class index as the conservative worst grade;
4. averages confidence only among views that selected that worst class;
5. records incomplete evidence when timeout occurs before all target views/results exist.

The mapping “higher class index means worse fruit” is a domain invariant. Change the aggregation rule if your class order is different.

### 5.10 Track state machine

```text
TENTATIVE
   | enough real hits + downward motion
   v
CONFIRMED -> ACQUIRING -> WAITING_FOR_SEGMENTATION
                    \-> WAITING_FOR_CLASSIFICATION -> COMPLETED

Any nonterminal operational state may be wrapped as LOST while predictions
continue. A matching real detection restores the previous operational state.
Invalid motion, insufficient evidence, or policy failures can end in REJECTED.
```

## 6. How React is linked to the backend

### 6.1 API wrapper

[`frontend/src/api.ts`](../frontend/src/api.ts) is the only general HTTP layer. `request<T>()`:

1. calls browser `fetch()`;
2. chooses JSON or text from `Content-Type`;
3. converts non-2xx responses to `ApiError`;
4. returns the payload as the caller's TypeScript type.

Image calls use `FormData`:

```ts
const body = new FormData()
body.append('modelId', modelId)
body.append('confidence', String(confidence))
body.append('iou', String(iou))
body.append('image', image, 'frame.jpg')

const result = await request<InferenceResult>('/api/infer', {
  method: 'POST',
  body,
})
```

Do not set the `Content-Type` header yourself. The browser generates `multipart/form-data; boundary=...`, and the boundary is required by the server parser.

### 6.2 Still images

[`InferenceWorkbench.tsx`](../frontend/src/components/InferenceWorkbench.tsx):

1. accepts a local image and creates a preview with `URL.createObjectURL()`;
2. revokes the old object URL in effect cleanup;
3. posts the original file to `/api/infer`;
4. draws boxes/points as an SVG using the source dimensions as `viewBox`;
5. reconstructs RLE masks into a same-sized transparent canvas;
6. displays the server stage timings plus client HTTP round-trip time.

Keeping image, mask canvas and SVG as separate aligned layers makes rendering independent of the displayed CSS size.

### 6.3 Recorded video

[`VideoDetection.tsx`](../frontend/src/components/VideoDetection.tsx):

1. draws the current `<video>` frame to a hidden `<canvas>`;
2. scales the longest edge down to a configured maximum;
3. encodes the canvas to a JPEG `Blob` with `toBlob()`;
4. awaits inference before requesting another frame;
5. copies the exact submitted pixels to a processed canvas;
6. draws the returned overlays above that frozen canvas.

Only one request is ever in flight. This is client-side backpressure and prevents an unbounded queue if inference is slower than playback. Normal detection schedules the next iteration with `requestAnimationFrame()`. Expensive grounding pauses playback and explicitly seeks by `sampleInterval`, keeping output and media time aligned.

An incrementing loop token invalidates async work after stop/reset/file changes. Without it, an old response can overwrite the result of a new run.

### 6.4 Live conveyor

[`LiveConveyor.tsx`](../frontend/src/components/LiveConveyor.tsx):

1. gets webcam frames with `getUserMedia()` or plays a selected local video;
2. captures/encodes a frame to JPEG;
3. increments a strictly monotonic `frameId`;
4. posts it to `/api/pipeline/frame`;
5. awaits the complete stateful pipeline response;
6. freezes the submitted pixels and overlays the returned track boxes;
7. schedules the next request according to `live_max_fps` after accounting for request time;
8. refreshes the capture library when returned capture IDs change.

Again, only one frame request is active. Concurrent requests could arrive out of order and corrupt the meaning of velocity, missed frames and percentage crossings.

React's cleanup stops all webcam tracks and revokes local video object URLs.

### 6.5 Development and production routing

Development (`frontend/vite.config.ts`):

```ts
server: {
  port: 5173,
  proxy: { '/api': 'http://127.0.0.1:8080' },
}
```

Production:

1. `vite build` writes static assets to `frontend/dist`.
2. CMake embeds/configures that directory as `RUNNER_FRONTEND_DIST`.
3. The C++ server serves the SPA files and owns `/api/*`.
4. `PythonEngine.cpp` forwards only model lifecycle/inference work to port 8091.

## 7. Public API routes used by React

| Method | Route | React use |
| --- | --- | --- |
| GET | `/api/health` | runtime status and active model |
| GET | `/api/models` | model library |
| POST | `/api/models/rescan` | explicit rescan |
| POST | `/api/models/{id}/activate` | load/select a model |
| POST | `/api/infer` | still image or sampled video frame |
| GET | `/api/pipeline/config` | overlay geometry and policies |
| GET | `/api/pipeline` | current pipeline snapshot |
| POST | `/api/pipeline/frame` | ordered live frame update |
| POST | `/api/pipeline/reset` | new tracking generation |
| GET | `/api/pipeline/captures` | persisted crop metadata |
| GET | `/api/pipeline/captures/{id}.jpg` | exact saved crop |
| GET | `/api/pipeline/results/{trackId}/{asset}` | saved surface map, mask, overlay, confidence or metadata |

Inspect the actual multipart validation and route handlers in [`backend/src/main.cpp`](../backend/src/main.cpp).

## 8. Suggested greenfield project structure

```text
my-runner/
  frontend/
    src/
      api.ts
      types.ts
      components/
        StillImage.tsx
        VideoInference.tsx
        LiveConveyor.tsx
        DetectionOverlay.tsx
  backend/
    app.py                 # FastAPI if omitting C++
    contracts.py
    model_registry.py
    inference_adapters.py
    tracking.py
    capture_scheduler.py
    grading.py
  config/
    pipeline.json
  tests/
```

For a close reproduction, retain the repository's `frontend/`, `backend/src/`, `backend/python/` split instead.

## 9. Tests to write before using a real camera

- RLE round-trip including first-pixel foreground and all-zero/all-one masks.
- IoU association with high-score and low-score recovery.
- Two nearby fruit crossing without ID reuse.
- Detector missing frames during dense route sampling and across the legacy 20/40/60/80 targets.
- Reacquisition after a predicted interval retains the original track ID.
- Candidate sampling obeys the stride and bounded-memory limits without duplicate frames.
- A fast jump crossing multiple legacy targets captures them in order when surface mapping is disabled.
- Blur, too-small crop and overlapping-fruit rejection.
- Queue overflow and reset-generation stale results.
- Timeout or route exit with too little reliable surface evidence falls back explicitly.
- Synthetic sphere and oblate-spheroid projection, rotation recovery and seam-connected defects.
- Complete fused evidence uses the existing conservative grade rule; incomplete evidence cannot silently claim a fused grade.
- Stop/restart video while an HTTP request is still in flight.

The existing C++ regression cases are in [`backend/tests/RunnerTests.cpp`](../backend/tests/RunnerTests.cpp). Python discovery/RLE tests are in [`backend/python/test_model_service.py`](../backend/python/test_model_service.py), and mapping geometry/fusion tests are in [`backend/python/test_surface_mapping.py`](../backend/python/test_surface_mapping.py).

## 10. Common reconstruction mistakes

- Calling Python directly from React while also expecting the C++ tracker to update.
- Sending frames concurrently or allowing responses to update UI out of order.
- Using CSS display dimensions for boxes rather than source-image coordinates and an SVG `viewBox`.
- Setting a multipart `Content-Type` manually and omitting its boundary.
- Treating a tracker prediction as a real high-confidence hit during confirmation.
- Starting a new track anywhere in the ROI instead of at the entry gate.
- Advancing the capture target only when a crop fails; a failed quality gate should keep that target eligible.
- Using frame count as route percentage. Progress is geometric position between acquisition lines.
- Assuming all ByteTrack implementations use identical motion/association math.
- Assuming four crops are physically guaranteed even when every candidate is blurred or the track expires.
- Changing the Python result JSON without changing C++ parsing and TypeScript interfaces.

## 11. Authoritative learning resources

React and browser integration:

- [React `useEffect`](https://react.dev/reference/react/useEffect) for synchronization and cleanup;
- [React `useRef`](https://react.dev/reference/react/useRef) for DOM/media handles and mutable loop tokens;
- [MDN: Using the Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch);
- [MDN: FormData](https://developer.mozilla.org/en-US/docs/Web/API/FormData);
- [MDN: Using FormData objects](https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest_API/Using_FormData_Objects), including the multipart boundary warning;
- [MDN: Canvas `drawImage()`](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage);
- [MDN: Canvas `toBlob()`](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob);
- [MDN: `getUserMedia()`](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia);
- [Vite getting started](https://vite.dev/guide/) and [production build guide](https://vite.dev/guide/build).

Python and model runtimes:

- [FastAPI file uploads](https://fastapi.tiangolo.com/tutorial/request-files/) for a simpler direct Python API;
- [Python `http.server`](https://docs.python.org/3/library/http.server.html), including its production warning;
- [Python `zipfile`](https://docs.python.org/3/library/zipfile.html);
- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict);
- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html);
- [PyTorch `inference_mode`](https://docs.pytorch.org/docs/stable/generated/torch.inference_mode.html);
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/);
- [ONNX Runtime CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html);
- [Transformers D-FINE](https://huggingface.co/docs/transformers/model_doc/d_fine);
- [Transformers Grounding DINO](https://huggingface.co/docs/transformers/model_doc/grounding-dino);
- [NVIDIA LocateAnything source](https://github.com/NVlabs/Eagle);
- [NCNN](https://github.com/Tencent/ncnn).

Tracking and C++ HTTP:

- [Official ByteTrack repository and paper links](https://github.com/FoundationVision/ByteTrack);
- [cpp-httplib](https://github.com/yhirose/cpp-httplib).

## 12. The shortest viable reproduction

For the first working version:

1. Use Vite + React + TypeScript.
2. Copy the `InferenceResult`/`Detection` contracts.
3. Implement `api.infer()` with `FormData`.
4. Build a FastAPI `UploadFile` endpoint that decodes one image and calls one Ultralytics model.
5. Normalize boxes to source-pixel `(x, y, width, height)` JSON.
6. Render boxes with an SVG `viewBox`.
7. Add a sequential canvas-to-JPEG video loop.
8. Port the `ByteTracker`, `acquisition_progress()`, `next_capture_target()`, crop quality gates and grading logic into backend modules.
9. Keep tracker state keyed by a run/session ID on the server.
10. Add persistence and multiple model formats only after synthetic tracker tests pass.

That path reproduces the behavior with fewer moving parts. Add the C++ gateway later only if native runtimes, deterministic state ownership, or deployment performance justify it.
