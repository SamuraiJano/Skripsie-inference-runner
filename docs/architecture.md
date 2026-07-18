# Runtime architecture decision

## Decision

Use TypeScript/React for the browser interface and C++17 with NCNN for the production inference service.

## Options considered

| Runtime | Pi 4 strengths | Main cost | Decision |
| --- | --- | --- | --- |
| C++ + NCNN | Direct access to NCNN's ARM NEON, packing, OpenMP, INT8, allocators, and memory-mapped loading; small production runtime | More explicit model preprocessing and output parsing | Selected |
| C++ + ONNX Runtime | Broad `.onnx` compatibility and strong graph optimization/thread controls | Larger runtime and generally more deployment weight for a CPU-only Pi | Keep as a future optional engine if an actual model cannot be exported reliably to NCNN |
| Python + Ultralytics/ONNX Runtime | Fastest model experimentation and broad parser support | Interpreter, wheels, framework overhead, slower startup, and more RAM pressure | Training/export only, not the production hot path |
| Rust + bindings | Strong memory safety and good native performance | Inference bindings still wrap a C/C++ runtime, adding integration work without improving the kernels | Not selected for this runner |
| Browser JavaScript/WASM | Simplest single-surface deployment | Browser memory limits, cross-origin isolation for WASM threading, extra copies, and less predictable sustained Pi performance | UI only |

## Pi-specific choices

- Production is CPU-only by default. NCNN's own Raspberry Pi build guidance describes Vulkan on Raspberry Pi as something to experiment with rather than a guaranteed faster path.
- The build targets a 64-bit Cortex-A72 and disables unused NCNN tools, examples, tests, and Vulkan code.
- `ncnn::Net` remains loaded between requests.
- The newest pinned NCNN runtime enables memory-mapped model loading to avoid an additional large model-file copy during load.
- HTTP connections are admitted one at a time. NCNN receives up to four compute threads without a competing HTTP worker pool.
- Letterbox and normalization happen in C++ and feed the NCNN tensor directly. OpenCV is intentionally not a runtime dependency.
- Postprocessing filters by confidence before NMS, caps candidates, and caps final detections.
- The React production bundle is static and is served by the C++ process.

## Model contract

Model execution is only reliable when four things agree with training/export:

1. input width and height;
2. RGB preprocessing, mean, and normalization;
3. input/output blob names;
4. output parser and class-label count.

That contract is stored beside each imported model in `manifest.json`. Runtime settings such as thread count remain separate from model input resolution.

## References

- [Tencent NCNN](https://github.com/Tencent/ncnn) documents its native ARM NEON, multi-core, INT8, packing, and low-memory runtime features.
- [NCNN build guidance](https://github.com/Tencent/ncnn/wiki/how-to-build) covers Raspberry Pi builds and Vulkan's experimental nature on Pi hardware.
- [Ultralytics NCNN export](https://docs.ultralytics.com/integrations/ncnn) documents embedded/Raspberry Pi deployment and the `.param`/`.bin` export path.
- [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) documents sequential execution, graph optimization, and intra-op thread tuning for the alternative ONNX path.
- [Raspberry Pi OS](https://www.raspberrypi.com/documentation/computers/os.html) recommends the 64-bit edition for Pi 4-class hardware and compute-heavy tasks.
