from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download


WHISPER_LANGS = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl",
}

WHISPER_MODELS = {
    "Faster-Whisper tiny  •  ~0.5 GB VRAM": "Systran/faster-whisper-tiny",
    "Faster-Whisper base  •  ~0.7 GB VRAM": "Systran/faster-whisper-base",
    "Faster-Whisper small  •  ~1.2 GB VRAM": "Systran/faster-whisper-small",
    "Faster-Whisper medium  •  ~2.5 GB VRAM": "Systran/faster-whisper-medium",
    "Faster-Whisper large-v2  •  ~4.5 GB VRAM": "Systran/faster-whisper-large-v2",
    "Faster-Whisper large-v3  •  ~4.5 GB VRAM": "Systran/faster-whisper-large-v3",
    "Faster-Whisper distil-large-v3  •  ~3 GB VRAM": "Systran/faster-distil-whisper-large-v3",
}


class ASRManager:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.model = None
        self.label = None

    def unload(self) -> None:
        self.model = None
        self.label = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    def ensure_model(self, label: str) -> Path:
        if label not in WHISPER_MODELS:
            raise ValueError(f"Unknown Faster-Whisper model: {label}")
        repo_id = WHISPER_MODELS[label]
        local_dir = self.models_dir / repo_id.rsplit("/", 1)[-1]
        if local_dir.exists() and any(local_dir.iterdir()):
            return local_dir
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=WHISPER_MODELS[label],
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return local_dir

    def _load(self, label: str):
        if self.model is not None and self.label == label:
            return self.model
        from faster_whisper import WhisperModel
        import torch

        self.unload()
        model_dir = self.ensure_model(label)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)
        self.label = label
        return self.model

    def transcribe(
        self,
        audio_path: str,
        label: str = "Faster-Whisper large-v3  •  ~4.5 GB VRAM",
        language: str = "Auto-detect",
        batch_size: int = 8,
    ) -> tuple[str, str]:
        if not audio_path:
            return "", "No audio selected."
        model = self._load(label)
        lang_code = WHISPER_LANGS.get(language)
        batch_size = max(1, int(batch_size or 1))
        if batch_size > 1:
            from faster_whisper import BatchedInferencePipeline

            pipeline = BatchedInferencePipeline(model=model)
            segments, info = pipeline.transcribe(
                audio_path,
                beam_size=5,
                language=lang_code,
                batch_size=batch_size,
            )
        else:
            segments, info = model.transcribe(audio_path, beam_size=5, language=lang_code)
        text = "".join(segment.text for segment in segments).strip()
        detected = getattr(info, "language", None) or "unknown"
        return text, f"{label} transcription complete. Detected language: {detected}."

    def transcribe_many(
        self,
        audio_paths: list[str],
        label: str = "Faster-Whisper large-v3  •  ~4.5 GB VRAM",
        language: str = "Auto-detect",
        batch_size: int = 8,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, str]:
        results: dict[str, str] = {}
        total = len(audio_paths)
        for index, audio_path in enumerate(audio_paths, 1):
            if progress_cb:
                progress_cb(index - 1, total, f"Transcribing {Path(audio_path).name}")
            text, _ = self.transcribe(audio_path, label, language, batch_size)
            results[audio_path] = text
            if progress_cb:
                progress_cb(index, total, f"Transcribed {Path(audio_path).name}")
        return results
