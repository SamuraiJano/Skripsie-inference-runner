"""Local multi-format inference service for Inference Runner Windows.

The native C++ process keeps ownership of the public API, model library, and
conveyor pipeline.  This private loopback service supplies the model runtimes
that are impractical to embed in C++ (PyTorch, ONNX Runtime, TensorRT, and the
other export backends exposed by Ultralytics).
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_SURFACE_REQUEST_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_FILES = 4096
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
_FORMAT_CAPABILITIES: list[dict[str, Any]] | None = None
LOCATE_ANYTHING_MODEL = "nvidia/LocateAnything-3B"
LOCATE_ANYTHING_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
LOCATE_ANYTHING_REQUIRED_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "processing_locateanything.py",
    "tokenizer_config.json",
)
LOCATE_ANYTHING_SOURCE_REVISION = "bb860a7efa22bb40cd4bb77959512375b92db934"
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUNDING_DINO_REQUIRED_FILES = (
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
_DISCOVERY_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_DISCOVERABLE_FILE_SUFFIXES = {
    ".pt",
    ".pth",
    ".torchscript",
    ".onnx",
    ".engine",
    ".pb",
    ".tflite",
    ".mnn",
    ".pte",
    ".zip",
}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_format(path: Path) -> str:
    """Return the Ultralytics backend name implied by an artifact path."""
    name = path.name.lower()
    if name.endswith(".locateanything.json"):
        return "locate-anything"
    if name.endswith(".groundingdino.json"):
        return "grounding-dino"
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            try:
                model_type = str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", ""))
            except (OSError, json.JSONDecodeError):
                model_type = ""
            if model_type == "locateanything":
                return "locate-anything"
            if model_type == "grounding-dino":
                return "grounding-dino"
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
        ".pth": "dfine",
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


def _model_files(path: Path, format_name: str) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = {
        "locate-anything": ("*.safetensors", "*.safetensors.index.json", "config.json"),
        "grounding-dino": ("*.safetensors", "config.json"),
        "ncnn": ("*.param", "*.bin", "metadata.yaml", "metadata.yml"),
        "openvino": ("*.xml", "*.bin", "metadata.yaml", "metadata.yml"),
        "saved_model": ("*.pb", "variables/*", "metadata.yaml", "metadata.yml"),
        "paddle": ("*.pdmodel", "*.pdiparams", "metadata.yaml", "metadata.yml"),
    }.get(format_name, ("*",))
    files: set[Path] = set()
    for pattern in patterns:
        files.update(candidate for candidate in path.glob(pattern) if candidate.is_file())
    return sorted(files)


def _candidate_fingerprint(path: Path, format_name: str) -> tuple[int, int]:
    files = _model_files(path, format_name)
    if not files:
        return (0, 0)
    stats = [candidate.stat() for candidate in files]
    return (sum(item.st_size for item in stats), max(item.st_mtime_ns for item in stats))


def discover_model_candidates(root: Path) -> list[Path]:
    """Return one path per supported model or exported model package.

    Exported runtimes such as NCNN and OpenVINO are directories containing
    several sidecar files.  Once one of those directories is recognized we
    stop descending into it, so the UI receives one model record instead of a
    record for every shard/configuration file.
    """
    root = root.resolve()
    candidates: list[Path] = []
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(
            name for name in directories
            if not name.startswith(".") and name.lower() not in {"offload", "__pycache__"}
        )
        directory = Path(current)
        if directory != root:
            try:
                directory_format = detect_format(directory)
            except ValueError:
                directory_format = ""
            if directory_format:
                candidates.append(directory)
                directories.clear()
                continue
        for filename in sorted(filenames):
            path = directory / filename
            if path.suffix.lower() not in _DISCOVERABLE_FILE_SUFFIXES:
                continue
            try:
                detect_format(path)
            except ValueError:
                continue
            candidates.append(path)
    return candidates


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value[:48] or "model"


def _task_name(value: Any) -> str:
    return {
        "detect": "yolo-detect",
        "segment": "yolo-segment",
        "classify": "classification",
        "classification": "classification",
    }.get(str(value or "").lower(), "raw")


def _labels(value: Any) -> list[str]:
    if isinstance(value, dict):
        def key(item: Any) -> tuple[int, str]:
            text = str(item)
            return (int(text), "") if text.isdigit() else (sys.maxsize, text)
        return [str(value[item]) for item in sorted(value, key=key)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _image_size(value: Any, default_width: int = 640, default_height: int = 640) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and value:
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[-1]), int(value[-2])
    return default_width, default_height


def _precision_details(dtypes: dict[str, int], source: str, quantization_config: Any = None) -> dict[str, Any]:
    """Describe stored precision from tensor byte counts and quantizer metadata.

    This is model introspection, not numeric conversion.  Mixed integer and
    floating tensors are reported as hybrid rather than guessed to be purely
    quantized.
    """
    config = quantization_config if isinstance(quantization_config, dict) else {}
    method = str(config.get("quant_method") or config.get("quantization_method") or "").upper()
    bits = config.get("bits")
    if bits in {4, "4"}:
        label = f"{method + ' ' if method else ''}INT4"
        return {"precision": "int4", "quantization": label, "quantized": True, "quantizationMethod": source}
    if bits in {8, "8"}:
        label = f"{method + ' ' if method else ''}INT8"
        return {"precision": "int8", "quantization": label, "quantized": True, "quantizationMethod": source}

    normalized = {name.lower().replace("torch.", ""): size for name, size in dtypes.items() if size > 0}
    has_int8 = any(name in normalized for name in ("int8", "qint8", "uint8", "quint8"))
    floating = [
        ("fp8", "float8"),
        ("bf16", "bfloat16"),
        ("fp16", "float16"),
        ("fp32", "float32"),
        ("fp64", "float64"),
    ]
    present_floating = [(precision, dtype) for precision, dtype in floating if dtype in normalized]
    if has_int8 and present_floating:
        return {
            "precision": "int8",
            "quantization": "INT8 + floating-point hybrid",
            "quantized": True,
            "quantizationMethod": source,
        }
    if has_int8:
        return {"precision": "int8", "quantization": "INT8", "quantized": True, "quantizationMethod": source}
    if present_floating:
        precision, dtype = max(present_floating, key=lambda item: normalized[item[1]])
        label = {"fp8": "FP8", "bf16": "BF16", "fp16": "FP16", "fp32": "FP32", "fp64": "FP64"}[precision]
        return {
            "precision": "fp32" if precision == "fp64" else precision,
            "quantization": f"Not quantized ({label} weights)",
            "quantized": False,
            "quantizationMethod": source,
        }
    return {"precision": "unknown", "quantization": "Unknown", "quantized": False, "quantizationMethod": source}


def _safetensors_dtypes(paths: list[Path]) -> dict[str, int]:
    dtypes: dict[str, int] = {}
    for path in paths:
        with path.open("rb") as source:
            header_size = int.from_bytes(source.read(8), "little")
            if header_size <= 0 or header_size > 128 * 1024 * 1024:
                raise ValueError(f"invalid Safetensors header in {path.name}")
            header = json.loads(source.read(header_size))
        for name, tensor in header.items():
            if name == "__metadata__" or not isinstance(tensor, dict):
                continue
            offsets = tensor.get("data_offsets", [0, 0])
            size = max(0, int(offsets[1]) - int(offsets[0])) if len(offsets) == 2 else 1
            dtype = str(tensor.get("dtype", "")).upper()
            mapped = {
                "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
                "I8": "int8", "U8": "uint8", "F8_E4M3": "float8", "F8_E5M2": "float8",
            }.get(dtype, dtype.lower())
            dtypes[mapped] = dtypes.get(mapped, 0) + size
    return dtypes


def _yaml_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {}
    task = re.search(r"(?m)^task:\s*([^#\r\n]+)", text)
    if task:
        result["task"] = task.group(1).strip(" '\"")
    image_size = re.search(r"(?ms)^imgsz:\s*(?:\[\s*(\d+)\s*,\s*(\d+)\s*\]|\r?\n\s*-\s*(\d+)\s*\r?\n\s*-\s*(\d+))", text)
    if image_size:
        values = [int(value) for value in image_size.groups() if value]
        result["imgsz"] = values
    names_match = re.search(r"(?ms)^names:\s*\r?\n(?P<body>(?:^[ \t]+[^\r\n]+\r?\n?)*)", text)
    if names_match:
        names: dict[int, str] = {}
        for key, value in re.findall(r"(?m)^\s+(\d+):\s*(.+?)\s*$", names_match.group("body")):
            names[int(key)] = value.strip(" '\"")
        result["names"] = names
    quantization = re.search(r"(?ms)^quantization:\s*\r?\n(?P<body>(?:^[ \t]+[^\r\n]+\r?\n?)*)", text)
    if quantization:
        body = quantization.group("body")
        for field in ("precision", "mode", "method"):
            match = re.search(rf"(?m)^\s+{field}:\s*(.+?)\s*$", body)
            if match:
                result[f"quantization_{field}"] = match.group(1).strip(" '\"")
    quantize = re.search(r"(?m)^\s+quantize:\s*([^#\r\n]+)", text)
    if quantize:
        result["quantize"] = quantize.group(1).strip()
    return result


def _ncnn_blobs(param_path: Path) -> tuple[str, list[str]]:
    input_blob = "in0"
    output_blobs: set[str] = set()
    for line in param_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0] == "Input":
            input_count = int(parts[2])
            output_count = int(parts[3])
            if input_count == 0 and output_count > 0:
                input_blob = parts[4]
        output_blobs.update(token for token in parts if re.fullmatch(r"out\d+", token))
    return input_blob, sorted(output_blobs) or ["out0"]


def _inspect_pytorch(path: Path) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    dtypes: dict[str, int] = {}
    for parameter in parameters:
        name = str(parameter.dtype)
        dtypes[name] = dtypes.get(name, 0) + parameter.numel() * parameter.element_size()
    args = getattr(model, "args", {})
    args = args if isinstance(args, dict) else {}
    result = {
        **_precision_details(dtypes, "PyTorch checkpoint tensors"),
        "task": _task_name(args.get("task") or getattr(model, "task", None)),
        "labels": _labels(getattr(model, "names", {})),
    }
    result["inputWidth"], result["inputHeight"] = _image_size(args.get("imgsz"))
    del parameters, model, checkpoint
    gc.collect()
    return result


def _inspect_dfine(path: Path) -> dict[str, Any]:
    from dfine_checkpoint import checkpoint_labels, checkpoint_state, validate_dfine_nano

    state = checkpoint_state(path)
    validate_dfine_nano(state)
    dtypes: dict[str, int] = {}
    for tensor in state.values():
        if not hasattr(tensor, "dtype") or not hasattr(tensor, "numel"):
            continue
        name = str(tensor.dtype)
        dtypes[name] = dtypes.get(name, 0) + tensor.numel() * tensor.element_size()
    result = {
        **_precision_details(dtypes, "D-FINE checkpoint tensors"),
        "task": "yolo-detect",
        "labels": checkpoint_labels(state),
        "inputWidth": 640,
        "inputHeight": 640,
        "inputBlob": "pixel_values",
        "outputBlobs": ["logits", "pred_boxes"],
    }
    del state
    gc.collect()
    return result


def _structured_metadata(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return value


def _inspection_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: _structured_metadata(value) for key, value in metadata.items()}
    args = metadata.get("args", {})
    args = args if isinstance(args, dict) else {}
    width, height = _image_size(metadata.get("imgsz", args.get("imgsz")))
    return {
        "task": _task_name(metadata.get("task", args.get("task"))),
        "labels": _labels(metadata.get("names", {})),
        "inputWidth": width,
        "inputHeight": height,
        "inputBlob": "images",
        "outputBlobs": ["auto"],
    }


def _inspect_torchscript(path: Path) -> dict[str, Any]:
    import torch

    extra_files: dict[str, Any] = {"config.txt": ""}
    model = torch.jit.load(str(path), map_location="cpu", _extra_files=extra_files)
    dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype)
        dtypes[name] = dtypes.get(name, 0) + parameter.numel() * parameter.element_size()
    metadata: dict[str, Any] = {}
    config_value = extra_files.get("config.txt", "")
    if isinstance(config_value, bytes):
        config_value = config_value.decode("utf-8", errors="replace")
    if config_value:
        parsed = _structured_metadata(config_value)
        if isinstance(parsed, dict):
            metadata = parsed
    result = {
        **_precision_details(dtypes, "TorchScript tensors"),
        **_inspection_from_metadata(metadata),
    }
    del model
    gc.collect()
    return result


def _inspect_onnx(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    dtype_map = {
        onnx.TensorProto.FLOAT: ("float32", 4),
        onnx.TensorProto.FLOAT16: ("float16", 2),
        onnx.TensorProto.BFLOAT16: ("bfloat16", 2),
        onnx.TensorProto.DOUBLE: ("float64", 8),
        onnx.TensorProto.INT8: ("int8", 1),
        onnx.TensorProto.UINT8: ("uint8", 1),
    }
    for name in ("FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ"):
        value = getattr(onnx.TensorProto, name, None)
        if value is not None:
            dtype_map[value] = ("float8", 1)
    dtypes: dict[str, int] = {}
    for tensor in model.graph.initializer:
        dtype, item_size = dtype_map.get(tensor.data_type, (str(tensor.data_type), 1))
        elements = 1
        for dimension in tensor.dims:
            elements *= max(1, int(dimension))
        dtypes[dtype] = dtypes.get(dtype, 0) + elements * item_size
    metadata = {item.key: item.value for item in model.metadata_props}
    return {
        **_precision_details(dtypes, "ONNX initializer tensors"),
        **_inspection_from_metadata(metadata),
    }


def _inspect_tensorrt_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        metadata_size = int.from_bytes(source.read(4), "little")
        if metadata_size <= 0 or metadata_size > 16 * 1024 * 1024:
            raise ValueError("TensorRT engine has no readable embedded metadata")
        metadata = json.loads(source.read(metadata_size))
    args = metadata.get("args", {}) if isinstance(metadata, dict) else {}
    args = args if isinstance(args, dict) else {}
    if args.get("int8"):
        details = {"precision": "int8", "quantization": "INT8", "quantized": True, "quantizationMethod": "TensorRT embedded metadata"}
    elif args.get("half"):
        details = {"precision": "fp16", "quantization": "Not quantized (FP16 weights)", "quantized": False, "quantizationMethod": "TensorRT embedded metadata"}
    else:
        details = {"precision": "fp32", "quantization": "Not quantized (FP32 weights)", "quantized": False, "quantizationMethod": "TensorRT embedded metadata"}
    return {**details, **_inspection_from_metadata(metadata)}


def _inspect_ncnn(path: Path) -> dict[str, Any]:
    param_paths = sorted([*path.glob("*.ncnn.param"), *path.glob("*.param")])
    bin_paths = sorted([*path.glob("*.ncnn.bin"), *path.glob("*.bin")])
    if not param_paths or not bin_paths:
        raise ValueError("NCNN package requires matching .param and .bin files")
    metadata_path = next(iter(sorted([*path.glob("metadata.yaml"), *path.glob("metadata.yml")])), None)
    metadata = _yaml_metadata(metadata_path) if metadata_path else {}
    precision_text = str(metadata.get("quantization_precision", ""))
    mode = str(metadata.get("quantization_mode", "")).lower()
    quantize = str(metadata.get("quantize", "")).lower()
    if "int8" in precision_text.lower() or quantize == "8":
        hybrid = mode == "hybrid" or "mixed" in precision_text.lower()
        details = {
            "precision": "int8",
            "quantization": "INT8 + FP32 hybrid" if hybrid else "INT8",
            "quantized": True,
            "quantizationMethod": str(metadata.get("quantization_method", "metadata.yaml")),
        }
    else:
        details = {
            "precision": "fp32",
            "quantization": "Not quantized (FP32 weights)",
            "quantized": False,
            "quantizationMethod": "NCNN package metadata",
        }
    input_blob, output_blobs = _ncnn_blobs(param_paths[0])
    width, height = _image_size(metadata.get("imgsz"))
    return {
        **details,
        "task": _task_name(metadata.get("task")),
        "labels": _labels(metadata.get("names", {})),
        "inputWidth": width,
        "inputHeight": height,
        "inputBlob": input_blob,
        "outputBlobs": output_blobs,
        "primaryPath": str(param_paths[0].resolve()),
        "secondaryPath": str(bin_paths[0].resolve()),
    }


def _inspect_huggingface(path: Path, format_name: str) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    details = _precision_details(
        _safetensors_dtypes(sorted(path.glob("*.safetensors"))),
        "Safetensors tensor headers",
        config.get("quantization_config"),
    )
    if format_name == "locate-anything":
        return {
            **details, "task": "vision-grounding", "labels": [],
            "inputWidth": 1280, "inputHeight": 1280,
        }
    return {
        **details, "task": "vision-grounding", "labels": [],
        "inputWidth": 1333, "inputHeight": 800,
    }


def _framework(format_name: str) -> str:
    if format_name in {"pt", "torchscript", "dfine"}:
        return "pytorch"
    if format_name == "onnx":
        return "onnx"
    if format_name == "engine":
        return "tensorrt"
    if format_name in {"saved_model", "pb", "litert", "edgetpu"}:
        return "tensorflow"
    if format_name in {"locate-anything", "grounding-dino"}:
        return format_name
    return format_name


def _base_inspection(path: Path, format_name: str) -> dict[str, Any]:
    if format_name == "ncnn":
        return _inspect_ncnn(path)
    if format_name in {"locate-anything", "grounding-dino"} and path.is_dir():
        return _inspect_huggingface(path, format_name)
    if format_name == "pt":
        return _inspect_pytorch(path)
    if format_name == "dfine":
        return _inspect_dfine(path)
    if format_name == "torchscript":
        return _inspect_torchscript(path)
    if format_name == "onnx":
        return _inspect_onnx(path)
    if format_name == "engine":
        return _inspect_tensorrt_metadata(path)
    metadata_path = path / "metadata.yaml" if path.is_dir() else path.parent / "metadata.yaml"
    metadata = _yaml_metadata(metadata_path)
    width, height = _image_size(metadata.get("imgsz"))
    name = path.name.lower()
    hinted = "int8" if "int8" in name else "fp16" if "fp16" in name or "half" in name else "unknown"
    quantization = {
        "int8": "INT8 (filename/metadata hint)",
        "fp16": "Not quantized (FP16 hint)",
        "unknown": "Unknown",
    }[hinted]
    return {
        "precision": hinted,
        "quantization": quantization,
        "quantized": hinted == "int8",
        "quantizationMethod": "export metadata" if metadata else "no embedded precision metadata found",
        "task": _task_name(metadata.get("task")),
        "labels": _labels(metadata.get("names", {})),
        "inputWidth": width,
        "inputHeight": height,
        "inputBlob": "images",
        "outputBlobs": ["auto"],
    }


def inspect_discovered_model(path: Path, models_root: Path, capabilities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    format_name = detect_format(path)
    relative = path.resolve().relative_to(models_root.resolve()).as_posix()
    family = relative.split("/", 1)[0]
    display_name = {
        "locate-anything": "NVIDIA LocateAnything 3B",
        "grounding-dino": "Grounding DINO Tiny",
        "dfine": f"D-FINE {path.stem}",
    }.get(format_name, path.stem if path.is_file() else re.sub(r"_(?:ncnn|openvino|saved|paddle)_model$", "", path.name))
    capability = capabilities.get(format_name, {})
    backend = "ncnn" if format_name == "ncnn" else format_name if format_name in {"locate-anything", "grounding-dino", "dfine"} else "ultralytics"
    inspection_error = ""
    try:
        details = _base_inspection(path, format_name)
    except Exception as error:
        details = {
            "precision": "unknown", "quantization": "Unknown", "quantized": False,
            "quantizationMethod": "inspection failed", "task": "raw", "labels": [],
            "inputWidth": 640, "inputHeight": 640, "inputBlob": "images", "outputBlobs": ["auto"],
        }
        inspection_error = str(error)
    fingerprint = _candidate_fingerprint(path, format_name)
    modified = max((candidate.stat().st_mtime for candidate in _model_files(path, format_name)), default=path.stat().st_mtime)
    identifier = f"{_slug(family)}-{_slug(display_name)}-{hashlib.sha256(relative.lower().encode('utf-8')).hexdigest()[:8]}"
    primary_path = str(path.resolve())
    return {
        "id": identifier,
        "name": display_name.replace("_", " "),
        "family": family.replace("_", " "),
        "relativePath": relative,
        "path": details.pop("primaryPath", primary_path),
        "secondaryPath": details.pop("secondaryPath", ""),
        "format": format_name,
        "framework": _framework(format_name),
        "backend": backend,
        "runtime": "NCNN" if format_name == "ncnn" else str(capability.get("runtime", format_name)),
        "accelerator": "Vulkan" if format_name == "ncnn" else str(capability.get("accelerator", "automatic")),
        "runtimeAvailable": bool(capability.get("available", False)) if format_name != "ncnn" else False,
        "compatible": not inspection_error,
        "statusReason": inspection_error,
        "sizeBytes": fingerprint[0],
        "modifiedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified)),
        **details,
    }


def discover_models(models_root: Path) -> list[dict[str, Any]]:
    """Inspect the model tree and return stable, UI-ready model records.

    The (total size, latest mtime) fingerprint avoids reparsing large model
    headers on every five-second React refresh.  Runtime availability is still
    refreshed on a cache hit because an execution provider can change without
    the artifact itself changing.
    """
    capabilities = {item["format"]: item for item in format_capabilities()}
    results: list[dict[str, Any]] = []
    live_cache_keys: set[str] = set()
    for path in discover_model_candidates(models_root):
        format_name = detect_format(path)
        cache_key = str(path.resolve()).lower()
        fingerprint = _candidate_fingerprint(path, format_name)
        live_cache_keys.add(cache_key)
        cached = _DISCOVERY_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            result = dict(cached[1])
            if format_name != "ncnn":
                capability = capabilities.get(format_name, {})
                result["runtimeAvailable"] = bool(capability.get("available", False))
                result["runtime"] = str(capability.get("runtime", result["runtime"]))
                result["accelerator"] = str(capability.get("accelerator", result["accelerator"]))
        else:
            result = inspect_discovered_model(path, models_root, capabilities)
            _DISCOVERY_CACHE[cache_key] = (fingerprint, dict(result))
        results.append(result)
    for key in set(_DISCOVERY_CACHE) - live_cache_keys:
        del _DISCOVERY_CACHE[key]
    results.sort(key=lambda item: (str(item["family"]).lower(), str(item["name"]).lower(), str(item["relativePath"]).lower()))
    return results


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def locate_anything_model_path() -> Path | None:
    raw_path = os.environ.get("RUNNER_LOCATE_ANYTHING_MODEL", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    return path if all((path / name).is_file() for name in LOCATE_ANYTHING_REQUIRED_FILES) else None


def grounding_dino_model_path() -> Path | None:
    raw_path = os.environ.get("RUNNER_GROUNDING_DINO_MODEL", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    return path if all((path / name).is_file() for name in GROUNDING_DINO_REQUIRED_FILES) else None


def extract_archive(archive: Path) -> Path:
    """Safely extract a directory-based export and return its model path."""
    state_root = os.environ.get("RUNNER_MODEL_STATE_ROOT", "").strip()
    if state_root:
        archive_key = hashlib.sha256(str(archive.resolve()).lower().encode("utf-8")).hexdigest()[:16]
        cache_root = Path(state_root).resolve() / "expanded" / archive_key
        cache_root.mkdir(parents=True, exist_ok=True)
    else:
        cache_root = archive.parent
    destination = cache_root / "expanded"
    ready_marker = destination / ".ready"
    if ready_marker.is_file():
        return discover_model(destination)

    temporary = cache_root / ".expanded-staging"
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
    """Prove that ONNX Runtime can create a CUDA session, not merely list it.

    Provider discovery can succeed even when DLL or driver dependencies fail
    at session creation.  A one-node in-memory graph makes the health result
    reflect the execution path that real inference will use.
    """
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
    locate_root = Path(os.environ.get("RUNNER_LOCATE_ANYTHING_ROOT", ""))
    locate_worker = locate_root / "locateanything_worker.py"
    locate_model = locate_anything_model_path()
    grounding_dino_model = grounding_dino_model_path()
    transformers_compatible = False
    dfine_available = False
    if module_available("transformers"):
        import transformers

        transformers_compatible = transformers.__version__ == "4.57.1"
        dfine_available = all(
            hasattr(transformers, name)
            for name in ("DFineConfig", "DFineForObjectDetection", "RTDetrImageProcessor")
        )
    locate_available = (
        cuda
        and locate_worker.is_file()
        and locate_model is not None
        and transformers_compatible
        and all(
            module_available(name)
            for name in (
                "accelerate",
                "cv2",
                "decord",
                "lmdb",
                "numpy",
                "packaging",
                "peft",
                "PIL",
                "requests",
                "sentencepiece",
                "timm",
                "torchvision",
            )
        )
    )
    specs = [
        ("pt", ".pt", True, "PyTorch", cuda, "CUDA" if cuda else "CPU"),
        ("dfine", ".pth", dfine_available, "Hugging Face Transformers", cuda, "CUDA" if cuda else "CPU"),
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
        (
            "locate-anything",
            ".locateanything.json",
            locate_available,
            "NVIDIA LocateAnything",
            cuda,
            "CUDA + CPU offload" if cuda else "unavailable",
        ),
        (
            "grounding-dino",
            ".groundingdino.json",
            cuda and grounding_dino_model is not None and transformers_compatible,
            "Hugging Face Transformers",
            cuda,
            "CUDA" if cuda else "unavailable",
        ),
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
    """Convert a probability mask to the browser's zero-run-first RLE format.

    The mask is restored to source-image dimensions, thresholded, flattened in
    row-major order, then encoded as alternating background/foreground run
    lengths.  The first count always represents background and may be zero.
    This exact convention is decoded by React's ``MaskOverlay`` component.
    """
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
    options: dict[str, Any] = field(default_factory=dict)
    references: int = 1


def _locate_anything_worker_type() -> Any:
    source_root = Path(os.environ.get("RUNNER_LOCATE_ANYTHING_ROOT", ""))
    worker_path = source_root / "locateanything_worker.py"
    if not worker_path.is_file():
        raise RuntimeError("LocateAnything source is missing; run npm run setup:locate-anything")
    spec = importlib.util.spec_from_file_location("runner_locateanything_worker", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("the pinned LocateAnything worker could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LocateAnythingWorker


def _load_locate_anything_worker(
    worker_type: Any,
    descriptor: dict[str, Any],
    artifact_path: Path,
    model_path: Path | None = None,
) -> tuple[Any, bool]:
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    model_name = str(descriptor.get("modelId", ""))
    revision = str(descriptor.get("modelRevision", ""))
    if model_name != LOCATE_ANYTHING_MODEL or revision != LOCATE_ANYTHING_MODEL_REVISION:
        raise ValueError("the LocateAnything descriptor does not reference the pinned NVIDIA checkpoint")
    model_path = model_path or locate_anything_model_path()
    if model_path is None:
        raise RuntimeError("the local LocateAnything checkpoint is incomplete; run npm run setup:locate-anything")

    worker = worker_type.__new__(worker_type)
    worker.device = "cuda"
    worker.dtype = torch.bfloat16
    worker.use_batch_runtime = False
    worker.scheduler = "pipeline"
    worker.group_size = 0
    common = {"trust_remote_code": True, "local_files_only": True}
    worker.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **common)
    worker.processor = AutoProcessor.from_pretrained(str(model_path), **common)

    # The released checkpoint is about 7.8 GB. Leave headroom for image tokens
    # on 8 GB laptop GPUs and let Accelerate offload overflow layers to RAM.
    total_vram = torch.cuda.get_device_properties(0).total_memory
    cpu_offload = total_vram < 12 * 1024**3
    model_options: dict[str, Any] = {**common, "torch_dtype": torch.bfloat16}
    if cpu_offload:
        state_root = os.environ.get("RUNNER_MODEL_STATE_ROOT", "").strip()
        offload_folder = (Path(state_root).resolve() / "offload" / "locate-anything") if state_root else artifact_path.parent / "offload"
        offload_folder.mkdir(parents=True, exist_ok=True)
        gpu_gib = max(4, int(total_vram / 1024**3) - 1)
        model_options.update(
            device_map="auto",
            max_memory={0: f"{gpu_gib}GiB", "cpu": "24GiB"},
            offload_folder=str(offload_folder),
        )
        worker.model = AutoModel.from_pretrained(str(model_path), **model_options).eval()
    else:
        worker.model = AutoModel.from_pretrained(str(model_path), **model_options).to("cuda").eval()
    return worker, cpu_offload


def parse_locate_anything_answer(answer: str, width: int, height: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert LocateAnything's normalized box/point tokens to runner results."""
    pattern = re.compile(
        r"(?:<ref>(?P<label>.*?)</ref>\s*)?<box><(?P<a>\d+)><(?P<b>\d+)>"
        r"(?:<(?P<c>\d+)><(?P<d>\d+)>)?</box>",
        re.DOTALL,
    )
    detections: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    label_ids: dict[str, int] = {}
    for match in pattern.finditer(answer):
        label = re.sub(r"\s+", " ", (match.group("label") or "grounded region")).strip()
        class_id = label_ids.setdefault(label, len(label_ids))
        values = [max(0, min(1000, int(match.group(name)))) for name in ("a", "b")]
        if match.group("c") is None:
            points.append(
                {
                    "classId": class_id,
                    "label": label,
                    "x": values[0] / 1000.0 * width,
                    "y": values[1] / 1000.0 * height,
                }
            )
            continue
        values.extend(max(0, min(1000, int(match.group(name)))) for name in ("c", "d"))
        x1, x2 = sorted((values[0] / 1000.0 * width, values[2] / 1000.0 * width))
        y1, y2 = sorted((values[1] / 1000.0 * height, values[3] / 1000.0 * height))
        detections.append(
            {
                "classId": class_id,
                "label": label,
                "confidence": 1.0,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "trackId": None,
                "mask": None,
            }
        )
    return detections, points


class ModelRegistry:
    """Own loaded model instances and serialize model/GPU lifecycle calls.

    The C++ process can create several logical engines for one model. Reference
    counts keep the Python instance alive until the last engine releases it;
    the re-entrant lock prevents concurrent activation/inference/unload calls
    from racing over GPU memory or non-thread-safe runtime objects.
    """

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
        """Validate, select a runtime, load once, and warm up the artifact."""
        model_id = str(request.get("modelId", ""))
        if not model_id or not all(ch.isalnum() or ch in "-_" for ch in model_id):
            raise ValueError("invalid model id")
        path, format_name = resolve_artifact(self._checked_path(str(request.get("path", ""))))
        if format_name == "ncnn":
            raise ValueError("NCNN exports must use the native manifest + .param + .bin import path")
        task_hint = str(request.get("task", "auto"))
        if task_hint == "raw":
            task_hint = "auto"
        input_width = max(32, min(4096, int(request.get("inputWidth", 640))))
        input_height = max(32, min(4096, int(request.get("inputHeight", 640))))
        available = {item["format"]: item for item in format_capabilities()}
        capability = available.get(format_name)
        if not capability or not capability["available"]:
            runtime = capability["runtime"] if capability else format_name
            raise RuntimeError(f"{runtime} is not installed for {format_name} models")

        if format_name == "locate-anything":
            return self._load_locate_anything(model_id, path, task_hint, request, capability)
        if format_name == "grounding-dino":
            return self._load_grounding_dino(model_id, path, task_hint, capability)
        if format_name == "dfine":
            return self._load_dfine(model_id, path, task_hint, capability)

        from ultralytics import YOLO

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

    def _load_dfine(
        self,
        model_id: str,
        path: Path,
        task_hint: str,
        capability: dict[str, Any],
    ) -> dict[str, Any]:
        import torch
        from PIL import Image

        from dfine_checkpoint import load_dfine_nano

        if task_hint not in {"auto", "yolo-detect"}:
            raise ValueError("D-FINE checkpoints require the yolo-detect task")
        device = "cuda" if capability["accelerated"] else "cpu"
        with self.lock:
            existing = self.models.get(model_id)
            if existing is not None and existing.path == path:
                existing.references += 1
                return self._metadata(existing)
            try:
                processor, model, labels = load_dfine_nano(path, device)
                inputs = processor(images=Image.new("RGB", (640, 640)), return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    model(**inputs)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    "D-FINE nano could not fit on the GPU. Close other GPU applications and retry."
                ) from error
            self.models[model_id] = LoadedModel(
                model_id=model_id,
                model=(processor, model),
                path=path,
                format_name="dfine",
                task="yolo-detect",
                device=device,
                input_width=640,
                input_height=640,
                labels=labels,
                runtime=capability["runtime"],
                accelerator=capability["accelerator"],
                accelerated=capability["accelerated"],
                options={"architecture": "D-FINE nano"},
            )
            return self._metadata(self.models[model_id])

    def _load_locate_anything(
        self,
        model_id: str,
        path: Path,
        task_hint: str,
        request: dict[str, Any],
        capability: dict[str, Any],
    ) -> dict[str, Any]:
        import torch

        if task_hint != "vision-grounding":
            raise ValueError("LocateAnything models require the vision-grounding task")
        if request.get("trustedCheckpoint") is not True:
            raise ValueError("LocateAnything requires explicit trusted remote-code confirmation")
        if path.is_dir():
            descriptor = {
                "modelId": LOCATE_ANYTHING_MODEL,
                "modelRevision": LOCATE_ANYTHING_MODEL_REVISION,
                "sourceRevision": LOCATE_ANYTHING_SOURCE_REVISION,
                "localFilesOnly": True,
                "maxNewTokens": 2048,
                "maxImageEdge": 1280,
            }
            model_path = path
        else:
            if path.stat().st_size > 64 * 1024:
                raise ValueError("LocateAnything descriptor is too large")
            descriptor = json.loads(path.read_text(encoding="utf-8"))
            model_path = locate_anything_model_path()
        if descriptor.get("sourceRevision") != LOCATE_ANYTHING_SOURCE_REVISION:
            raise ValueError("the LocateAnything descriptor does not match the pinned GitHub source")
        if descriptor.get("localFilesOnly") is not True:
            raise ValueError("LocateAnything descriptors must require the locally installed checkpoint")
        max_new_tokens = max(64, min(8192, int(descriptor.get("maxNewTokens", 2048))))
        max_image_edge = max(512, min(2560, int(descriptor.get("maxImageEdge", 1280))))

        with self.lock:
            existing = self.models.get(model_id)
            if existing is not None and existing.path == path:
                existing.references += 1
                return self._metadata(existing)
            try:
                worker, cpu_offload = _load_locate_anything_worker(
                    _locate_anything_worker_type(), descriptor, path, model_path
                )
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    "LocateAnything could not fit in GPU/RAM. Close other GPU applications and retry; "
                    "the 7.8 GB checkpoint uses CPU offload on this 8 GB profile."
                ) from error
            self.models[model_id] = LoadedModel(
                model_id=model_id,
                model=worker,
                path=path,
                format_name="locate-anything",
                task="vision-grounding",
                device="cuda",
                input_width=max_image_edge,
                input_height=max_image_edge,
                labels=[],
                runtime="NVIDIA LocateAnything",
                accelerator="CUDA + CPU offload" if cpu_offload else "CUDA",
                accelerated=True,
                options={
                    "maxNewTokens": max_new_tokens,
                    "maxImageEdge": max_image_edge,
                    "cpuOffload": cpu_offload,
                    "modelRevision": descriptor["modelRevision"],
                    "sourceRevision": descriptor["sourceRevision"],
                },
            )
            return self._metadata(self.models[model_id])

    def _load_grounding_dino(
        self,
        model_id: str,
        path: Path,
        task_hint: str,
        capability: dict[str, Any],
    ) -> dict[str, Any]:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if task_hint != "vision-grounding":
            raise ValueError("Grounding DINO models require the vision-grounding task")
        if path.is_dir():
            descriptor = {
                "modelId": GROUNDING_DINO_MODEL,
                "modelRevision": GROUNDING_DINO_MODEL_REVISION,
                "localFilesOnly": True,
                "textThreshold": 0.25,
            }
            model_path = path
        else:
            if path.stat().st_size > 64 * 1024:
                raise ValueError("Grounding DINO descriptor is too large")
            descriptor = json.loads(path.read_text(encoding="utf-8"))
            model_path = grounding_dino_model_path()
        if (
            descriptor.get("modelId") != GROUNDING_DINO_MODEL
            or descriptor.get("modelRevision") != GROUNDING_DINO_MODEL_REVISION
        ):
            raise ValueError("the Grounding DINO descriptor does not reference the pinned tiny checkpoint")
        if descriptor.get("localFilesOnly") is not True:
            raise ValueError("Grounding DINO descriptors must require the locally installed checkpoint")
        text_threshold = max(0.0, min(1.0, float(descriptor.get("textThreshold", 0.25))))
        if model_path is None:
            raise RuntimeError("the local Grounding DINO Tiny checkpoint is incomplete; run npm run setup:grounding-dino")

        with self.lock:
            existing = self.models.get(model_id)
            if existing is not None and existing.path == path:
                existing.references += 1
                return self._metadata(existing)
            try:
                processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
                model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                ).to("cuda").eval()
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    "Grounding DINO Tiny could not fit on the GPU. Close other GPU applications and retry."
                ) from error
            self.models[model_id] = LoadedModel(
                model_id=model_id,
                model=(processor, model),
                path=path,
                format_name="grounding-dino",
                task="vision-grounding",
                device="cuda",
                input_width=1333,
                input_height=800,
                labels=[],
                runtime=capability["runtime"],
                accelerator=capability["accelerator"],
                accelerated=True,
                options={
                    "textThreshold": text_threshold,
                    "modelRevision": descriptor["modelRevision"],
                },
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
            "options": loaded.options,
        }

    def unload(self, model_id: str) -> None:
        with self.lock:
            loaded = self.models.get(model_id)
            if loaded is None:
                return
            loaded.references -= 1
            if loaded.references <= 0:
                del self.models[model_id]
                loaded.model = None
                del loaded
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except RuntimeError:
                    pass

    def infer(
        self,
        model_id: str,
        encoded_image: bytes,
        confidence: float,
        iou: float,
        mask_threshold: float,
        prompt: str = "",
        grounding_mode: str = "detect",
        generation_mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Decode one image and normalize every backend into one JSON schema.

        Dispatch is model-specific, but all paths return source-space boxes,
        optional RLE masks/points/class scores, and the same timing fields.
        That normalization is what lets the C++ gateway and React components
        remain independent of PyTorch/Ultralytics/Transformers result classes.
        """
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
            if loaded.task == "vision-grounding":
                if loaded.format_name == "grounding-dino":
                    return self._infer_grounding_dino(
                        loaded, Image.fromarray(image), decoded, started, prompt, grounding_mode, confidence
                    )
                return self._infer_locate_anything(
                    loaded, Image.fromarray(image), decoded, started, prompt, grounding_mode, generation_mode
                )
            if loaded.format_name == "dfine":
                return self._infer_dfine(loaded, Image.fromarray(image), decoded, started, confidence)
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
            "points": [],
            "classifications": classifications,
            "tensors": [],
            "answer": "",
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

    @staticmethod
    def _infer_dfine(
        loaded: LoadedModel,
        image: Any,
        decoded: float,
        started: float,
        confidence: float,
    ) -> dict[str, Any]:
        import torch

        processor, model = loaded.model
        source_width, source_height = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(loaded.device) for key, value in inputs.items()}
        preprocessed = time.perf_counter()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        inference_finished = time.perf_counter()
        result = processor.post_process_object_detection(
            outputs,
            threshold=max(0.0, min(1.0, confidence)),
            target_sizes=[(source_height, source_width)],
        )[0]

        detections: list[dict[str, Any]] = []
        for box, score, raw_label in zip(result["boxes"], result["scores"], result["labels"]):
            class_id = int(raw_label.detach().cpu().item())
            label = loaded.labels[class_id] if 0 <= class_id < len(loaded.labels) else str(class_id)
            x1, y1, x2, y2 = (float(value) for value in box.detach().cpu().tolist())
            left = max(0.0, min(float(source_width), x1))
            top = max(0.0, min(float(source_height), y1))
            right = max(left, min(float(source_width), x2))
            bottom = max(top, min(float(source_height), y2))
            detections.append(
                {
                    "classId": class_id,
                    "label": label,
                    "confidence": float(score.detach().cpu().item()),
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top,
                    "trackId": None,
                    "mask": None,
                }
            )
        finished = time.perf_counter()
        return {
            "modelId": loaded.model_id,
            "task": loaded.task,
            "sourceWidth": source_width,
            "sourceHeight": source_height,
            "detections": detections,
            "points": [],
            "classifications": [],
            "tensors": [],
            "answer": "",
            "timing": {
                "decodeMs": (decoded - started) * 1000.0,
                "preprocessMs": (preprocessed - decoded) * 1000.0,
                "inferenceMs": (inference_finished - inference_started) * 1000.0,
                "detectionNmsMs": (finished - inference_finished) * 1000.0,
                "maskReconstructionMs": 0.0,
                "trackingMs": 0.0,
                "queueWaitMs": 0.0,
                "gradingMs": 0.0,
                "backendTotalMs": (finished - started) * 1000.0,
            },
        }

    @staticmethod
    def _infer_grounding_dino(
        loaded: LoadedModel,
        image: Any,
        decoded: float,
        started: float,
        prompt: str,
        grounding_mode: str,
        confidence: float,
    ) -> dict[str, Any]:
        import torch

        if grounding_mode != "detect":
            raise ValueError("Grounding DINO supports the named-category detection mode only")
        categories = [re.sub(r"\s+", " ", value).strip().lower() for value in re.split(r"[,.]", prompt)]
        categories = [value for value in categories if value]
        if not categories:
            raise ValueError("Grounding DINO requires at least one comma-separated category")
        if len(categories) > 64:
            raise ValueError("Grounding DINO accepts at most 64 categories per request")
        if len(prompt) > 1000:
            raise ValueError("Grounding DINO prompt must not exceed 1000 characters")

        processor, model = loaded.model
        source_width, source_height = image.size
        inputs = processor(images=image, text=[categories], return_tensors="pt").to(
            loaded.device, dtype=next(model.parameters()).dtype
        )
        preprocessed = time.perf_counter()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        inference_finished = time.perf_counter()
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=max(0.0, min(1.0, confidence)),
            text_threshold=float(loaded.options.get("textThreshold", 0.25)),
            target_sizes=[(source_height, source_width)],
            text_labels=[categories],
        )[0]

        label_ids = {label: index for index, label in enumerate(categories)}
        returned_labels = result["text_labels"] if "text_labels" in result else result["labels"]
        detections: list[dict[str, Any]] = []
        for box, score, raw_label in zip(result["boxes"], result["scores"], returned_labels):
            label = str(raw_label)
            if label not in label_ids:
                label_ids[label] = len(label_ids)
            x1, y1, x2, y2 = (float(value) for value in box.detach().cpu().tolist())
            detections.append(
                {
                    "classId": label_ids[label],
                    "label": label,
                    "confidence": float(score.detach().cpu().item()),
                    "x": max(0.0, x1),
                    "y": max(0.0, y1),
                    "width": max(0.0, min(float(source_width), x2) - max(0.0, x1)),
                    "height": max(0.0, min(float(source_height), y2) - max(0.0, y1)),
                    "trackId": None,
                    "mask": None,
                }
            )
        finished = time.perf_counter()
        return {
            "modelId": loaded.model_id,
            "task": loaded.task,
            "sourceWidth": source_width,
            "sourceHeight": source_height,
            "detections": detections,
            "points": [],
            "classifications": [],
            "tensors": [],
            "answer": "",
            "timing": {
                "decodeMs": (decoded - started) * 1000.0,
                "preprocessMs": (preprocessed - decoded) * 1000.0,
                "inferenceMs": (inference_finished - inference_started) * 1000.0,
                "detectionNmsMs": (finished - inference_finished) * 1000.0,
                "maskReconstructionMs": 0.0,
                "trackingMs": 0.0,
                "queueWaitMs": 0.0,
                "gradingMs": 0.0,
                "backendTotalMs": (finished - started) * 1000.0,
            },
        }

    @staticmethod
    def _infer_locate_anything(
        loaded: LoadedModel,
        image: Any,
        decoded: float,
        started: float,
        prompt: str,
        grounding_mode: str,
        generation_mode: str,
    ) -> dict[str, Any]:
        modes = {
            "detect",
            "ground-single",
            "ground-multi",
            "detect-text",
            "ground-text",
            "gui-box",
            "gui-point",
            "point",
        }
        if grounding_mode not in modes:
            raise ValueError("unsupported LocateAnything grounding mode")
        if generation_mode not in {"fast", "hybrid", "slow"}:
            raise ValueError("generationMode must be fast, hybrid, or slow")
        prompt = re.sub(r"\s+", " ", prompt).strip()
        prompt_required = grounding_mode not in {"detect-text"}
        if prompt_required and not prompt:
            raise ValueError("this LocateAnything mode requires a prompt")
        if len(prompt) > 1000:
            raise ValueError("LocateAnything prompt must not exceed 1000 characters")

        source_width, source_height = image.size
        maximum = int(loaded.options.get("maxImageEdge", 1280))
        if max(image.size) > maximum:
            image.thumbnail((maximum, maximum), resample=Image.Resampling.LANCZOS)
        preprocessed = time.perf_counter()
        generation = {
            "generation_mode": generation_mode,
            "max_new_tokens": int(loaded.options.get("maxNewTokens", 2048)),
            "verbose": True,
        }
        inference_started = time.perf_counter()
        if grounding_mode == "detect":
            categories = [value.strip() for value in re.split(r"</c>|,", prompt) if value.strip()]
            if not categories:
                raise ValueError("object detection requires at least one comma-separated category")
            response = loaded.model.detect(image, categories, **generation)
        elif grounding_mode == "ground-single":
            response = loaded.model.ground_single(image, prompt, **generation)
        elif grounding_mode == "ground-multi":
            response = loaded.model.ground_multi(image, prompt, **generation)
        elif grounding_mode == "detect-text":
            response = loaded.model.detect_text(image, **generation)
        elif grounding_mode == "ground-text":
            response = loaded.model.ground_text(image, prompt, **generation)
        elif grounding_mode == "gui-box":
            response = loaded.model.ground_gui(image, prompt, output_type="box", **generation)
        elif grounding_mode == "gui-point":
            response = loaded.model.ground_gui(image, prompt, output_type="point", **generation)
        else:
            response = loaded.model.point(image, prompt, **generation)
        inference_finished = time.perf_counter()
        answer = str(response.get("answer", ""))
        detections, points = parse_locate_anything_answer(answer, source_width, source_height)
        finished = time.perf_counter()
        return {
            "modelId": loaded.model_id,
            "task": loaded.task,
            "sourceWidth": source_width,
            "sourceHeight": source_height,
            "detections": detections,
            "points": points,
            "classifications": [],
            "tensors": [],
            "answer": answer,
            "timing": {
                "decodeMs": (decoded - started) * 1000.0,
                "preprocessMs": (preprocessed - decoded) * 1000.0,
                "inferenceMs": (inference_finished - inference_started) * 1000.0,
                "detectionNmsMs": (finished - inference_finished) * 1000.0,
                "maskReconstructionMs": 0.0,
                "trackingMs": 0.0,
                "queueWaitMs": 0.0,
                "gradingMs": 0.0,
                "backendTotalMs": (finished - started) * 1000.0,
            },
        }


class ServiceHandler(BaseHTTPRequestHandler):
    """Private loopback HTTP adapter used only by ``PythonEngine.cpp``.

    It deliberately accepts raw encoded image bytes for inference: the public
    C++ API has already parsed and validated the browser's multipart request.
    Keep this service bound to loopback; ``http.server`` is not a hardened
    internet-facing application server.
    """

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
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(HTTPStatus.OK, self.health_payload)
                return
            if path == "/models/discover":
                self.send_json(HTTPStatus.OK, discover_models(self.registry.models_root))
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
        except Exception as error:  # pragma: no cover - defensive service boundary
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/surface-map":
                # Imported lazily so model discovery and ordinary inference do
                # not pay the OpenCV surface-mapping initialization cost.
                from surface_mapping import map_surface_request

                request = json.loads(self.read_body(MAX_SURFACE_REQUEST_BYTES))
                if not isinstance(request, dict):
                    raise ValueError("surface-map request must be a JSON object")
                self.send_json(HTTPStatus.OK, map_surface_request(request))
                return
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
                prompt = query.get("prompt", [""])[0]
                grounding_mode = query.get("groundingMode", ["detect"])[0]
                generation_mode = query.get("generationMode", ["hybrid"])[0]
                image = self.read_body(MAX_IMAGE_BYTES)
                self.send_json(
                    HTTPStatus.OK,
                    self.registry.infer(
                        model_id,
                        image,
                        confidence,
                        iou,
                        mask_threshold,
                        prompt,
                        grounding_mode,
                        generation_mode,
                    ),
                )
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
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--locate-anything-root", type=Path)
    parser.add_argument("--locate-anything-model", type=Path)
    parser.add_argument("--grounding-dino-model", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the model service may only bind to loopback")
    args.models_root.mkdir(parents=True, exist_ok=True)
    if args.state_root is not None:
        args.state_root.mkdir(parents=True, exist_ok=True)
        os.environ["RUNNER_MODEL_STATE_ROOT"] = str(args.state_root.resolve())
    if args.locate_anything_root is not None:
        os.environ["RUNNER_LOCATE_ANYTHING_ROOT"] = str(args.locate_anything_root.resolve())
    if args.locate_anything_model is not None:
        os.environ["RUNNER_LOCATE_ANYTHING_MODEL"] = str(args.locate_anything_model.resolve())
    if args.grounding_dino_model is not None:
        os.environ["RUNNER_GROUNDING_DINO_MODEL"] = str(args.grounding_dino_model.resolve())
    print("[model-service] initializing PyTorch", flush=True)
    import torch
    print("[model-service] initializing Ultralytics", flush=True)
    import ultralytics
    print("[model-service] probing installed runtimes", flush=True)

    ServiceHandler.registry = ModelRegistry(args.models_root)
    ServiceHandler.health_payload = {
        "status": "ok",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "cudaVersion": torch.version.cuda,
        "gpuDevice": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "locateAnythingSourceRevision": LOCATE_ANYTHING_SOURCE_REVISION,
        "locateAnythingModelRevision": LOCATE_ANYTHING_MODEL_REVISION,
        "locateAnythingModelPath": str(locate_anything_model_path() or ""),
        "groundingDinoModelRevision": GROUNDING_DINO_MODEL_REVISION,
        "groundingDinoModelPath": str(grounding_dino_model_path() or ""),
        "formats": format_capabilities(),
    }
    print("[model-service] opening loopback server", flush=True)
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
