"""Local multi-format inference service for Inference Runner Windows.

The native C++ process keeps ownership of the public API, model library, and
conveyor pipeline.  This private loopback service supplies the model runtimes
that are impractical to embed in C++ (PyTorch, ONNX Runtime, TensorRT, and the
other export backends exposed by Ultralytics).
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_FILES = 4096
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
_FORMAT_CAPABILITIES: list[dict[str, Any]] | None = None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_format(path: Path) -> str:
    """Return the Ultralytics backend name implied by an artifact path."""
    name = path.name.lower()
    if path.is_dir():
        directory_markers = (
            ("_openvino_model", "openvino"),
            ("_saved_model", "saved_model"),
            ("_paddle_model", "paddle"),
            ("_ncnn_model", "ncnn"),
        )
        for marker, format_name in directory_markers:
            if marker in name:
                return format_name
        if any(path.glob("*.xml")):
            return "openvino"
        if (path / "saved_model.pb").is_file():
            return "saved_model"
        if any(path.glob("*.ncnn.param")) or any(path.glob("*.param")):
            return "ncnn"
        if any(path.glob("*.pdmodel")):
            return "paddle"
        raise ValueError(f"could not identify the exported model directory '{path.name}'")

    suffix_map = {
        ".pt": "pt",
        ".torchscript": "torchscript",
        ".onnx": "onnx",
        ".engine": "engine",
        ".pb": "pb",
        ".tflite": "edgetpu" if name.endswith("_edgetpu.tflite") else "litert",
        ".mnn": "mnn",
        ".pte": "executorch",
        ".xml": "openvino",
    }
    if path.suffix.lower() == ".zip":
        return "archive"
    try:
        return suffix_map[path.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported model artifact '{path.name}'") from error


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def extract_archive(archive: Path) -> Path:
    """Safely extract a directory-based export and return its model path."""
    destination = archive.parent / "expanded"
    ready_marker = destination / ".ready"
    if ready_marker.is_file():
        return discover_model(destination)

    temporary = archive.parent / ".expanded-upload"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            if len(entries) > MAX_ARCHIVE_FILES:
                raise ValueError("model archive contains too many files")
            if sum(item.file_size for item in entries) > MAX_ARCHIVE_BYTES:
                raise ValueError("expanded model archive exceeds 4 GB")
            for item in entries:
                target = temporary / item.filename
                if not _within(target, temporary):
                    raise ValueError("model archive contains an unsafe path")
            source.extractall(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        ready_marker.write_text("ok", encoding="utf-8")
        return discover_model(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def discover_model(root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in [root, *root.rglob("*")]:
        try:
            format_name = detect_format(path)
        except ValueError:
            continue
        if format_name == "archive" or path.name == ".ready":
            continue
        priority = 0 if path.is_dir() else 1
        candidates.append((priority, path))
    if not candidates:
        raise ValueError("the archive does not contain a supported model export")
    candidates.sort(key=lambda item: (item[0], len(item[1].parts), str(item[1])))
    return candidates[0][1]


def resolve_artifact(path: Path) -> tuple[Path, str]:
    format_name = detect_format(path)
    if format_name == "archive":
        path = extract_archive(path)
        format_name = detect_format(path)
    return path, format_name


def onnx_cuda_available(onnxruntime: Any) -> bool:
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers() or not module_available("onnx"):
        return False
    try:
        import onnx
        from onnx import TensorProto, helper

        input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
        output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
        graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "cuda-probe", [input_value], [output_value])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        return bool(session.get_providers()) and session.get_providers()[0] == "CUDAExecutionProvider"
    except Exception:
        return False


def format_capabilities() -> list[dict[str, Any]]:
    global _FORMAT_CAPABILITIES
    if _FORMAT_CAPABILITIES is not None:
        return [dict(item) for item in _FORMAT_CAPABILITIES]
    import torch

    cuda = torch.cuda.is_available()
    ort_providers: list[str] = []
    if module_available("onnxruntime"):
        import onnxruntime

        ort_providers = onnxruntime.get_available_providers()
    cuda_ort = module_available("onnxruntime") and onnx_cuda_available(onnxruntime)
    specs = [
        ("pt", ".pt", True, "PyTorch", cuda, "CUDA" if cuda else "CPU"),
        ("torchscript", ".torchscript", True, "PyTorch", cuda, "CUDA" if cuda else "CPU"),
        ("onnx", ".onnx", module_available("onnxruntime"), "ONNX Runtime", cuda_ort, "CUDA" if cuda_ort else "CPU"),
        ("engine", ".engine", module_available("tensorrt") and cuda, "TensorRT", cuda, "CUDA"),
        ("openvino", ".zip", module_available("openvino"), "OpenVINO", False, "CPU"),
        ("saved_model", ".zip", module_available("tensorflow"), "TensorFlow", False, "CPU"),
        ("pb", ".pb", module_available("tensorflow"), "TensorFlow", False, "CPU"),
        ("paddle", ".zip", module_available("paddle"), "PaddlePaddle", cuda, "CUDA" if cuda else "CPU"),
        ("litert", ".tflite", module_available("ai_edge_litert") or module_available("tensorflow"), "LiteRT", False, "CPU"),
        ("edgetpu", ".tflite", False, "Edge TPU", False, "Edge TPU"),
        ("mnn", ".mnn", module_available("MNN"), "MNN", False, "CPU"),
        ("executorch", ".pte", module_available("executorch"), "ExecuTorch", False, "CPU"),
    ]
    _FORMAT_CAPABILITIES = [
        {
            "format": format_name,
            "extensions": [extension],
            "available": available,
            "runtime": runtime,
            "accelerated": accelerated,
            "accelerator": accelerator,
        }
        for format_name, extension, available, runtime, accelerated, accelerator in specs
    ]
    return [dict(item) for item in _FORMAT_CAPABILITIES]


def task_to_ultralytics(task: str) -> str | None:
    return {
        "yolo-detect": "detect",
        "yolo-segment": "segment",
        "classification": "classify",
        "auto": None,
        "": None,
    }.get(task)


def task_from_ultralytics(task: str | None, fallback: str) -> str:
    return {
        "detect": "yolo-detect",
        "segment": "yolo-segment",
        "classify": "classification",
    }.get(task or "", fallback if fallback != "auto" else "yolo-detect")


def rle_mask(mask: Any, width: int, height: int, threshold: float) -> dict[str, Any]:
    import torch

    tensor = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")
    flat = (tensor.squeeze() >= threshold).to(torch.uint8).cpu().numpy().reshape(-1)
    counts: list[int] = []
    current = 0
    count = 0
    for value in flat:
        bit = int(value)
        if bit == current:
            count += 1
        else:
            counts.append(count)
            count = 1
            current = bit
    counts.append(count)
    return {"encoding": "rle-row-major", "width": width, "height": height, "counts": counts}


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    path: Path
    format_name: str
    task: str
    device: str
    input_width: int
    input_height: int
    labels: list[str]
    runtime: str
    accelerator: str
    accelerated: bool
    references: int = 1


class ModelRegistry:
    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root.resolve()
        self.models: dict[str, LoadedModel] = {}
        self.lock = threading.RLock()

    def _checked_path(self, raw_path: str) -> Path:
        path = Path(raw_path).resolve()
        if not _within(path, self.models_root) or not path.exists():
            raise ValueError("model path must exist inside the configured model store")
        return path

    def load(self, request: dict[str, Any]) -> dict[str, Any]:
        from ultralytics import YOLO

        model_id = str(request.get("modelId", ""))
        if not model_id or not all(ch.isalnum() or ch in "-_" for ch in model_id):
            raise ValueError("invalid model id")
        path, format_name = resolve_artifact(self._checked_path(str(request.get("path", ""))))
        if format_name == "ncnn":
            raise ValueError("NCNN exports must use the native manifest + .param + .bin import path")
        task_hint = str(request.get("task", "auto"))
        input_width = max(32, min(4096, int(request.get("inputWidth", 640))))
        input_height = max(32, min(4096, int(request.get("inputHeight", 640))))
        available = {item["format"]: item for item in format_capabilities()}
        capability = available.get(format_name)
        if not capability or not capability["available"]:
            runtime = capability["runtime"] if capability else format_name
            raise RuntimeError(f"{runtime} is not installed for {format_name} models")

        device = "0" if capability["accelerated"] else "cpu"
        task = task_to_ultralytics(task_hint)
        with self.lock:
            existing = self.models.get(model_id)
            if existing is not None and existing.path == path:
                existing.references += 1
                return self._metadata(existing)
            model = YOLO(str(path), task=task)
            # Run a warm-up/validation image so activation cannot succeed with an
            # incompatible artifact and then fail later on the conveyor.
            import numpy as np

            blank = np.zeros((input_height, input_width, 3), dtype=np.uint8)
            results = model.predict(blank, device=device, imgsz=(input_height, input_width), verbose=False)
            actual_task = task_from_ultralytics(getattr(model, "task", None), task_hint)
            if actual_task not in {"yolo-detect", "yolo-segment", "classification"}:
                raise ValueError(f"model task '{actual_task}' is not supported by this application")
            names_value = getattr(results[0], "names", {}) if results else {}
            names = [str(names_value[key]) for key in sorted(names_value)] if isinstance(names_value, dict) else list(names_value)
            self.models[model_id] = LoadedModel(
                model_id=model_id,
                model=model,
                path=path,
                format_name=format_name,
                task=actual_task,
                device=device,
                input_width=input_width,
                input_height=input_height,
                labels=names,
                runtime=capability["runtime"],
                accelerator=capability["accelerator"],
                accelerated=capability["accelerated"],
            )
            return self._metadata(self.models[model_id])

    @staticmethod
    def _metadata(loaded: LoadedModel) -> dict[str, Any]:
        return {
            "modelId": loaded.model_id,
            "format": loaded.format_name,
            "task": loaded.task,
            "labels": loaded.labels,
            "device": loaded.device,
            "runtime": loaded.runtime,
            "accelerator": loaded.accelerator,
            "accelerated": loaded.accelerated,
        }

    def unload(self, model_id: str) -> None:
        with self.lock:
            loaded = self.models.get(model_id)
            if loaded is None:
                return
            loaded.references -= 1
            if loaded.references <= 0:
                del self.models[model_id]

    def infer(self, model_id: str, encoded_image: bytes, confidence: float, iou: float, mask_threshold: float) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(encoded_image)) as source:
                image = np.asarray(source.convert("RGB"))
        except Exception as error:
            raise ValueError(f"could not decode image: {error}") from error
        decoded = time.perf_counter()
        with self.lock:
            loaded = self.models.get(model_id)
            if loaded is None:
                raise ValueError("model is not loaded")
            results = loaded.model.predict(
                source=image,
                device=loaded.device,
                imgsz=(loaded.input_height, loaded.input_width),
                conf=max(0.0, min(1.0, confidence)),
                iou=max(0.0, min(1.0, iou)),
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        if not results:
            raise RuntimeError("model returned no result")
        result = results[0]
        height, width = (int(value) for value in result.orig_shape)
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        masks = result.masks.data if result.masks is not None else None
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            for index, (coordinates, class_value, score) in enumerate(zip(xyxy, classes, scores)):
                class_id = int(class_value)
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                label = result.names.get(class_id, str(class_id)) if isinstance(result.names, dict) else str(class_id)
                detection = {
                    "classId": class_id,
                    "label": str(label),
                    "confidence": float(score),
                    "x": x1,
                    "y": y1,
                    "width": max(0.0, x2 - x1),
                    "height": max(0.0, y2 - y1),
                    "trackId": None,
                    "mask": None,
                }
                if masks is not None and index < len(masks):
                    detection["mask"] = rle_mask(masks[index], width, height, mask_threshold)
                detections.append(detection)

        classifications: list[dict[str, Any]] = []
        if result.probs is not None:
            probabilities = result.probs.data.detach().cpu()
            top_count = min(5, int(probabilities.numel()))
            values, indices = probabilities.topk(top_count)
            for class_id, score in zip(indices.tolist(), values.tolist()):
                label = result.names.get(class_id, str(class_id)) if isinstance(result.names, dict) else str(class_id)
                classifications.append({"classId": int(class_id), "label": str(label), "score": float(score)})

        speed = result.speed or {}
        preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
        inference_ms = float(speed.get("inference", 0.0) or 0.0)
        postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)
        return {
            "modelId": model_id,
            "task": loaded.task,
            "sourceWidth": width,
            "sourceHeight": height,
            "detections": detections,
            "classifications": classifications,
            "tensors": [],
            "timing": {
                "decodeMs": (decoded - started) * 1000.0,
                "preprocessMs": preprocess_ms,
                "inferenceMs": inference_ms,
                "detectionNmsMs": postprocess_ms,
                "maskReconstructionMs": 0.0,
                "trackingMs": 0.0,
                "queueWaitMs": 0.0,
                "gradingMs": 0.0,
                "backendTotalMs": elapsed_ms,
            },
        }


class ServiceHandler(BaseHTTPRequestHandler):
    registry: ModelRegistry
    health_payload: dict[str, Any]
    server_version = "InferenceRunnerModelService/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stdout.write("[model-service] " + format_string % args + "\n")
        sys.stdout.flush()

    def send_json(self, status: int, value: Any) -> None:
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def read_body(self, maximum: int) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > maximum:
            raise ValueError(f"request body must be between 1 and {maximum} bytes")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("incomplete request body")
        return body

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            if urlparse(self.path).path != "/health":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
                return
            self.send_json(HTTPStatus.OK, self.health_payload)
        except Exception as error:  # pragma: no cover - defensive service boundary
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/models/load":
                request = json.loads(self.read_body(1024 * 1024))
                self.send_json(HTTPStatus.OK, self.registry.load(request))
                return
            if parsed.path == "/models/unload":
                request = json.loads(self.read_body(1024 * 1024))
                self.registry.unload(str(request.get("modelId", "")))
                self.send_json(HTTPStatus.OK, {"status": "unloaded"})
                return
            if parsed.path.startswith("/models/") and parsed.path.endswith("/infer"):
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 3:
                    raise ValueError("invalid inference path")
                model_id = unquote(parts[1])
                query = parse_qs(parsed.query)
                confidence = float(query.get("confidence", ["0.25"])[0])
                iou = float(query.get("iou", ["0.5"])[0])
                mask_threshold = float(query.get("maskThreshold", ["0.5"])[0])
                image = self.read_body(MAX_IMAGE_BYTES)
                self.send_json(HTTPStatus.OK, self.registry.infer(model_id, image, confidence, iou, mask_threshold))
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            traceback.print_exc()
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private multi-format model service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--models-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the model service may only bind to loopback")
    args.models_root.mkdir(parents=True, exist_ok=True)
    import torch
    import ultralytics

    ServiceHandler.registry = ModelRegistry(args.models_root)
    ServiceHandler.health_payload = {
        "status": "ok",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaVersion": torch.version.cuda,
        "gpuDevice": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "formats": format_capabilities(),
    }
    server = ThreadingHTTPServer((args.host, args.port), ServiceHandler)
    print(f"Model service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
