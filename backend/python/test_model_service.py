import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from model_service import (
    detect_format,
    discover_model_candidates,
    extract_archive,
    inspect_discovered_model,
    parse_locate_anything_answer,
)


class ModelServiceTests(unittest.TestCase):
    def test_detects_single_file_formats(self) -> None:
        expected = {
            "model.pt": "pt",
            "model.pth": "dfine",
            "model.torchscript": "torchscript",
            "model.onnx": "onnx",
            "model.engine": "engine",
            "model.tflite": "litert",
            "model_edgetpu.tflite": "edgetpu",
            "model.mnn": "mnn",
            "model.pte": "executorch",
            "model.locateanything.json": "locate-anything",
            "model.groundingdino.json": "grounding-dino",
        }
        for filename, format_name in expected.items():
            self.assertEqual(detect_format(Path(filename)), format_name)

    def test_archive_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escaped/model.onnx", b"not-a-model")
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                extract_archive(archive)

    def test_discovers_packages_once_and_detects_ncnn_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "fruit-seg" / "fruit_int8_hybrid_ncnn_model"
            package.mkdir(parents=True)
            (package / "model.ncnn.param").write_text(
                "7767517\n2 3\nInput in0 0 1 in0\nConvolution head 1 2 in0 out0 out1\n",
                encoding="utf-8",
            )
            (package / "model.ncnn.bin").write_bytes(b"weights")
            (package / "metadata.yaml").write_text(
                "task: segment\nimgsz:\n- 640\n- 640\nnames:\n  0: fruit\n"
                "quantization:\n  precision: INT8 and FP32 mixed precision\n  mode: hybrid\n"
                "  method: ACIQ post-training quantization\n",
                encoding="utf-8",
            )
            (root / "fruit-seg" / "model.pt").write_bytes(b"checkpoint")

            candidates = discover_model_candidates(root)
            self.assertEqual(set(candidates), {package, root / "fruit-seg" / "model.pt"})
            inspected = inspect_discovered_model(package, root, {})
            self.assertEqual(inspected["precision"], "int8")
            self.assertEqual(inspected["quantization"], "INT8 + FP32 hybrid")
            self.assertEqual(inspected["labels"], ["fruit"])
            self.assertEqual(inspected["outputBlobs"], ["out0", "out1"])

    def test_detects_huggingface_safetensors_precision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "GroundingDINO-Tiny"
            package.mkdir()
            (package / "config.json").write_text(
                json.dumps({"model_type": "grounding-dino", "torch_dtype": "float32"}),
                encoding="utf-8",
            )
            header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            (package / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)

            inspected = inspect_discovered_model(
                package,
                root,
                {"grounding-dino": {"runtime": "Transformers", "accelerator": "CUDA", "available": True}},
            )
            self.assertEqual(inspected["precision"], "fp32")
            self.assertEqual(inspected["quantization"], "Not quantized (FP32 weights)")

    def test_discovers_and_inspects_dfine_nano_checkpoint(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "dfine"
            model_dir.mkdir()
            checkpoint = model_dir / "best.pth"
            state = {
                "backbone.stem.stem1.conv.weight": torch.zeros(1, dtype=torch.float32),
                "decoder.enc_score_head.weight": torch.zeros((1, 128), dtype=torch.float32),
                "decoder.denoising_class_embed.weight": torch.zeros((2, 128), dtype=torch.float32),
            }
            for index in range(3):
                state[f"decoder.decoder.layers.{index}.self_attn.in_proj_weight"] = torch.zeros((384, 128))
            torch.save({"ema": {"module": state}}, checkpoint)

            self.assertIn(checkpoint, discover_model_candidates(root))
            inspected = inspect_discovered_model(
                checkpoint,
                root,
                {"dfine": {"runtime": "Transformers", "accelerator": "CUDA", "available": True}},
            )
            self.assertTrue(inspected["compatible"])
            self.assertEqual(inspected["format"], "dfine")
            self.assertEqual(inspected["backend"], "dfine")
            self.assertEqual(inspected["task"], "yolo-detect")
            self.assertEqual(inspected["labels"], ["object"])
            self.assertEqual(inspected["precision"], "fp32")

    def test_parses_locate_anything_boxes_and_points(self) -> None:
        answer = (
            "<ref>red fruit</ref><box><100><200><500><800></box> "
            "<ref>button</ref><box><750><250></box>"
        )
        boxes, points = parse_locate_anything_answer(answer, 200, 100)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "red fruit")
        self.assertEqual((boxes[0]["x"], boxes[0]["y"], boxes[0]["width"], boxes[0]["height"]), (20, 20, 80, 60))
        self.assertEqual(len(points), 1)
        self.assertEqual((points[0]["x"], points[0]["y"]), (150, 25))

    def test_clamps_and_orders_locate_anything_coordinates(self) -> None:
        boxes, points = parse_locate_anything_answer("<box><900><800><100><200></box>", 100, 100)
        self.assertFalse(points)
        self.assertEqual((boxes[0]["x"], boxes[0]["y"], boxes[0]["width"], boxes[0]["height"]), (10, 20, 80, 60))


if __name__ == "__main__":
    unittest.main()
