from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import sys

import torch

EXPECTED = {
    "torch": "2.9.0",
    "torchaudio": "2.9.0",
    "transformers": "5.0.0",
    "gradio": "6.11.0",
    "huggingface-hub": "1.3.0",
    "hf-xet": "1.6.0",
    "peft": "0.18.1",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "onnxruntime-gpu": "1.28.0",
    "triton-windows": "3.5.1.post24",
    "flash-attn": "2.8.3",
}


def normalized(version: str) -> str:
    return version.split("+")[0]


print(f"[python] {sys.version.split()[0]}")
for package, expected in EXPECTED.items():
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(f"Missing required package: {package}") from exc
    print(f"[{package}] {actual}")
    if normalized(actual) != expected:
        raise SystemExit(f"Unsupported {package} version: expected {expected}, found {actual}")

print(f"[cuda runtime] {torch.version.cuda}")
print(f"[cuda available] {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required by this Easy GUI configuration.")
print(f"[gpu] {torch.cuda.get_device_name(0)}")
major, minor = torch.cuda.get_device_capability(0)
print(f"[compute capability] {major}.{minor}")
if major < 8:
    raise SystemExit("FlashAttention 2 in this build requires an Ampere-class or newer NVIDIA GPU.")
if importlib.util.find_spec("flash_attn") is None:
    raise SystemExit("flash_attn import is unavailable after installation.")
if importlib.util.find_spec("triton") is None:
    raise SystemExit("Triton import is unavailable after installation.")
print("[accelerators] FlashAttention 2 and Triton runtime are available.")
print("[cpp audio] ONNX Runtime GPU is available.")
