import tempfile
import unittest
import zipfile
from pathlib import Path

from model_service import detect_format, extract_archive


class ModelServiceTests(unittest.TestCase):
    def test_detects_single_file_formats(self) -> None:
        expected = {
            "model.pt": "pt",
            "model.torchscript": "torchscript",
            "model.onnx": "onnx",
            "model.engine": "engine",
            "model.tflite": "litert",
            "model_edgetpu.tflite": "edgetpu",
            "model.mnn": "mnn",
            "model.pte": "executorch",
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


if __name__ == "__main__":
    unittest.main()
