"""
llama.cpp backbone wrapper for MOSS-TTS-Delay.

Uses a thin C bridge (libbackbone_bridge.so) to interface with llama.cpp.
Feeds pre-computed embedding vectors and extracts hidden states,
bypassing the built-in token embedding and LM head.
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Keep os.add_dll_directory() handles alive for the lifetime of the process.
# On Python 3.8+ Windows, dependent DLL lookup for ctypes-loaded libraries no
# longer reliably follows PATH alone.
_DLL_DIRECTORY_HANDLES = []

_LIB_NAME = "backbone_bridge.dll" if __import__("os").name == "nt" else "libbackbone_bridge.so"


def _find_bridge_lib() -> Path:
    """Locate the compiled bridge shared library."""
    import os
    candidates = []
    env_bridge = os.environ.get("MOSS_LLAMA_CPP_BRIDGE")
    if env_bridge:
        candidates.append(Path(env_bridge))
    candidates += [
        Path(__file__).parent / _LIB_NAME,
        Path(__file__).parent / "build" / _LIB_NAME,
        Path(__file__).parent.parent.parent / "build" / _LIB_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find {_LIB_NAME}. Compile with:\n"
        f"  cd {Path(__file__).parent} && bash build_bridge.sh /path/to/llama.cpp"
    )


def _load_bridge(lib_path: Path):
    """Load the C bridge and set up function signatures."""
    import os

    if os.name == "nt":
        candidates = [lib_path.parent]

        # Packaged runtime layout:
        #   .runtime/llama-cpp/bridge/backbone_bridge.dll
        #   .runtime/llama-cpp/bin/{llama,ggml,cuda...}.dll
        if lib_path.parent.name.lower() == "bridge":
            candidates.append(lib_path.parent.parent / "bin")

        env_bin = os.environ.get("MOSS_LLAMA_CPP_BIN")
        if env_bin:
            candidates.append(Path(env_bin))

        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            candidates.append(Path(cuda_path) / "bin")

        seen = set()
        for directory in candidates:
            try:
                directory = directory.resolve()
            except Exception:
                continue
            key = str(directory).lower()
            if key in seen or not directory.is_dir():
                continue
            seen.add(key)
            try:
                handle = os.add_dll_directory(str(directory))
                _DLL_DIRECTORY_HANDLES.append(handle)
                log.debug("Registered DLL directory: %s", directory)
            except (AttributeError, OSError) as exc:
                log.debug("Could not register DLL directory %s: %s", directory, exc)

    lib = ctypes.CDLL(str(lib_path))

    lib.bridge_create.argtypes = [
        ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ]
    lib.bridge_create.restype = ctypes.c_void_p

    if hasattr(lib, "bridge_api_version"):
        lib.bridge_api_version.argtypes = []
        lib.bridge_api_version.restype = ctypes.c_int32

    if hasattr(lib, "bridge_set_lora"):
        lib.bridge_set_lora.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float,
        ]
        lib.bridge_set_lora.restype = ctypes.c_int32

    if hasattr(lib, "bridge_clear_lora"):
        lib.bridge_clear_lora.argtypes = [ctypes.c_void_p]
        lib.bridge_clear_lora.restype = ctypes.c_int32

    lib.bridge_decode_embd.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int8,
    ]
    lib.bridge_decode_embd.restype = ctypes.c_int32

    lib.bridge_decode_embd_batch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int8,
    ]
    lib.bridge_decode_embd_batch.restype = ctypes.c_int32

    lib.bridge_get_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.bridge_get_embeddings.restype = ctypes.POINTER(ctypes.c_float)

    lib.bridge_get_logits.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.bridge_get_logits.restype = ctypes.POINTER(ctypes.c_float)

    lib.bridge_n_embd.argtypes = [ctypes.c_void_p]
    lib.bridge_n_embd.restype = ctypes.c_int32

    lib.bridge_n_vocab.argtypes = [ctypes.c_void_p]
    lib.bridge_n_vocab.restype = ctypes.c_int32

    lib.bridge_clear_kv.argtypes = [ctypes.c_void_p]
    lib.bridge_clear_kv.restype = None

    lib.bridge_free.argtypes = [ctypes.c_void_p]
    lib.bridge_free.restype = None

    return lib


GGML_TYPE_MAP: dict[str, int] = {
    "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7,
    "q8_0": 8, "q8_1": 9, "q4_k": 12, "q5_k": 13, "q8_k": 15, "bf16": 30,
}

FLASH_ATTN_MAP: dict[str | bool, int] = {
    "auto": -1, "disabled": 0, False: 0, "enabled": 1, True: 1,
}


def _resolve_ggml_type(name: str) -> int:
    """Map a human-readable type name (e.g. 'q8_0') to its ggml_type int."""
    key = name.strip().lower()
    if key in GGML_TYPE_MAP:
        return GGML_TYPE_MAP[key]
    raise ValueError(
        f"Unknown ggml type {name!r}. "
        f"Valid options: {', '.join(sorted(GGML_TYPE_MAP))}"
    )


def _resolve_flash_attn(value: str | bool) -> int:
    """Map flash_attn config value to llama_flash_attn_type int."""
    if isinstance(value, bool):
        return FLASH_ATTN_MAP[value]
    key = value.strip().lower()
    if key in FLASH_ATTN_MAP:
        return FLASH_ATTN_MAP[key]
    raise ValueError(
        f"Unknown flash_attn value {value!r}. "
        f"Valid options: auto, disabled, enabled, true, false"
    )


class LlamaCppBackbone:
    """Wrapper around the Qwen3 backbone running in llama.cpp.

    Accepts embedding vectors as input (bypassing tok_embd) and returns
    hidden states (after final RMSNorm, before the built-in LM head).
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        type_k: str = "f16",
        type_v: str = "f16",
        flash_attn: str | bool = "auto",
        lora_path: str | Path | None = None,
        lora_scale: float = 1.0,
    ):
        lib_path = _find_bridge_lib()
        log.info("Loading bridge from %s", lib_path)
        self._lib = _load_bridge(lib_path)

        ggml_type_k = _resolve_ggml_type(type_k)
        ggml_type_v = _resolve_ggml_type(type_v)
        fa_type = _resolve_flash_attn(flash_attn)

        model_path = str(Path(model_path).resolve())
        log.info(
            "Loading GGUF model: %s (type_k=%s, type_v=%s, flash_attn=%s)",
            model_path, type_k, type_v, flash_attn,
        )
        self._handle = self._lib.bridge_create(
            model_path.encode("utf-8"), n_ctx, n_batch, n_threads, n_gpu_layers,
            ggml_type_k, ggml_type_v, fa_type,
        )
        if not self._handle:
            raise RuntimeError(f"Failed to load model from {model_path}")

        self._lora_path = None
        if lora_path:
            self.set_lora(lora_path, lora_scale)

        self.n_embd = self._lib.bridge_n_embd(self._handle)
        self.n_vocab = self._lib.bridge_n_vocab(self._handle)
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        log.info(
            "LlamaCppBackbone ready: n_embd=%d, n_vocab=%d, n_ctx=%d, n_batch=%d",
            self.n_embd, self.n_vocab, n_ctx, n_batch,
        )

    def decode_single(self, embd: np.ndarray, pos: int, output: bool = True) -> None:
        """Feed a single embedding vector at the given position."""
        assert embd.shape == (self.n_embd,), f"Expected ({self.n_embd},), got {embd.shape}"
        embd = np.ascontiguousarray(embd, dtype=np.float32)
        ptr = embd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self._lib.bridge_decode_embd(self._handle, ptr, pos, int(output))
        if ret != 0:
            raise RuntimeError(f"llama_decode failed with code {ret}")

    def decode_batch(
        self,
        embds: np.ndarray,
        pos_start: int = 0,
        output_last: bool = True,
    ) -> None:
        """Feed multiple embedding vectors (prefill).

        Automatically chunks into sub-batches of ``n_batch`` tokens.
        """
        n_tokens = embds.shape[0]
        assert embds.shape[1] == self.n_embd
        embds = np.ascontiguousarray(embds, dtype=np.float32)

        n_batch = self.n_batch
        for chunk_start in range(0, n_tokens, n_batch):
            chunk_end = min(chunk_start + n_batch, n_tokens)
            chunk = np.ascontiguousarray(embds[chunk_start:chunk_end], dtype=np.float32)
            is_last_chunk = chunk_end == n_tokens
            ptr = chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ret = self._lib.bridge_decode_embd_batch(
                self._handle, ptr,
                chunk_end - chunk_start,
                pos_start + chunk_start,
                int(output_last and is_last_chunk),
            )
            if ret != 0:
                raise RuntimeError(
                    f"llama_decode (batch) failed with code {ret} "
                    f"at chunk [{chunk_start}:{chunk_end}] of {n_tokens} tokens"
                )

    def get_hidden_state(self, idx: int = -1) -> np.ndarray:
        """Get the hidden state for the i-th output token.

        Returns a copy as float32 array of shape (n_embd,).
        """
        ptr = self._lib.bridge_get_embeddings(self._handle, idx)
        if not ptr:
            raise RuntimeError("llama_get_embeddings_ith returned NULL")
        arr = np.ctypeslib.as_array(ptr, shape=(self.n_embd,))
        return arr.copy()

    def get_logits(self, idx: int = -1) -> np.ndarray:
        """Get the text logits for the i-th output token.

        Returns a copy as float32 array of shape (n_vocab,).
        """
        ptr = self._lib.bridge_get_logits(self._handle, idx)
        if not ptr:
            raise RuntimeError("llama_get_logits_ith returned NULL")
        arr = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
        return arr.copy()

    def set_lora(self, lora_path: str | Path, scale: float = 1.0) -> None:
        """Load a GGUF LoRA adapter and apply it to this llama.cpp context."""
        if not hasattr(self._lib, "bridge_set_lora"):
            raise RuntimeError(
                "The packaged llama.cpp bridge predates GGUF LoRA support. "
                "Rebuild it with tools\\build_llamacpp_cuda_runtime.bat."
            )
        path = str(Path(lora_path).resolve())
        ret = self._lib.bridge_set_lora(
            self._handle, path.encode("utf-8"), ctypes.c_float(float(scale))
        )
        if ret != 0:
            raise RuntimeError(f"Failed to apply GGUF LoRA adapter (bridge code {ret}): {path}")
        self._lora_path = path
        log.info("Applied GGUF LoRA adapter: %s (scale=%.4f)", path, scale)

    def clear_lora(self) -> None:
        if self._handle and hasattr(self._lib, "bridge_clear_lora"):
            ret = self._lib.bridge_clear_lora(self._handle)
            if ret != 0:
                raise RuntimeError(f"Failed to clear GGUF LoRA adapter (bridge code {ret})")
        self._lora_path = None

    def clear_kv(self) -> None:
        """Clear the KV cache (for starting a new sequence)."""
        self._lib.bridge_clear_kv(self._handle)

    def close(self) -> None:
        if self._handle:
            self._lib.bridge_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
