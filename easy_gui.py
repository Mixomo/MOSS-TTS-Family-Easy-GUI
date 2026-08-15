from __future__ import annotations

import gc
import hashlib
import html
import re
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "moss_tts_upstream"
OUTPUTS_DIR = ROOT / "outputs"
VOICES_DIR = ROOT / "voices"
TRAINING_DIR = ROOT / "training"
DATASETS_DIR = TRAINING_DIR / "datasets"
TRAINING_OUTPUTS_DIR = TRAINING_DIR / "outputs"
PROJECTS_DIR = TRAINING_DIR / "projects"
RUNTIME_DIR = ROOT / ".runtime"
CACHE_DIR = RUNTIME_DIR / "hf-cache"
TEMP_DIR = RUNTIME_DIR / "temp"
ASSETS_DIR = ROOT / "assets"
for p in (OUTPUTS_DIR, VOICES_DIR, DATASETS_DIR, TRAINING_OUTPUTS_DIR, PROJECTS_DIR, CACHE_DIR, TEMP_DIR, ASSETS_DIR):
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_XET_CACHE"] = str(CACHE_DIR / "xet")
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TRITON_CACHE_DIR"] = str(RUNTIME_DIR / "triton-cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(RUNTIME_DIR / "torchinductor-cache")
os.environ["TORCH_EXTENSIONS_DIR"] = str(RUNTIME_DIR / "torch-extensions")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Gradio 6 currently triggers this Starlette deprecation warning on virtually every
# queued request. It is harmless and extremely noisy, so suppress only this exact
# warning while leaving all other runtime/dependency warnings visible.
warnings.filterwarnings(
    "ignore",
    message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*deprecated.*HTTP_422_UNPROCESSABLE_CONTENT.*",
    category=Warning,
)

if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

import gradio as gr

from moss_easy.asr import ASRManager, WHISPER_LANGS, WHISPER_MODELS
from moss_easy.voice_library import delete_voice, list_voice_names, resolve_voice, save_voice_sample

APP_TITLE = "MOSS-TTS Family Easy GUI"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
LANGUAGES = ["Auto", "Chinese", "Cantonese", "English", "Arabic", "Czech", "Danish", "Dutch", "Finnish", "French", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Italian", "Japanese", "Korean", "Macedonian", "Malay", "Persian (Farsi)", "Polish", "Portuguese", "Romanian", "Russian", "Spanish", "Swahili", "Swedish", "Tagalog", "Thai", "Turkish", "Vietnamese"]

REALTIME_LANGUAGES = [
    "Auto", "Chinese (zh)", "English (en)", "German (de)", "Spanish (es)", "French (fr)",
    "Japanese (ja)", "Italian (it)", "Hungarian (hu)", "Korean (ko)", "Russian (ru)",
    "Persian / Farsi (fa)", "Arabic (ar)", "Polish (pl)", "Portuguese (pt)", "Czech (cs)",
    "Danish (da)", "Swedish (sv)", "Greek (el)", "Turkish (tr)",
]
VOICE_GENERATOR_LANGUAGES = ["Auto", "Chinese (zh)", "English (en)"]

# Keep Windows DLL search-directory handles alive for the entire process.
_WINDOWS_DLL_DIR_HANDLES = []
def _cpp_runtime_available() -> bool:
    bridge = RUNTIME_DIR / "llama-cpp" / "bridge" / "backbone_bridge.dll"
    bin_dir = RUNTIME_DIR / "llama-cpp" / "bin"
    if not bridge.is_file() or not bin_dir.is_dir():
        return False
    has_llama = any(bin_dir.glob("llama*.dll"))
    has_ggml = any(bin_dir.glob("ggml*.dll"))
    return has_llama and has_ggml


TTS_BACKENDS = ["Transformers / PyTorch"]
if _cpp_runtime_available():
    TTS_BACKENDS.append("llama.cpp CUDA + ONNX")

# Official OpenMOSS-Team/MOSS-TTS-GGUF backbone variants. The actual filename is
# resolved from the repository file list at first inference so the GUI is not
# coupled to cosmetic filename changes upstream.
CPP_MODEL_VARIANTS = {
    "MOSS-TTS v1.5 — 8B Delay — F16": ("F16", ("f16",)),
    "MOSS-TTS v1.5 — 8B Delay — Q8_0": ("Q8_0", ("q8_0", "q8")),
    "MOSS-TTS v1.5 — 8B Delay — Q6_K": ("Q6_K", ("q6_k", "q6k")),
    "MOSS-TTS v1.5 — 8B Delay — Q5_K_M": ("Q5_K_M", ("q5_k_m", "q5km")),
    "MOSS-TTS v1.5 — 8B Delay — Q4_K_M": ("Q4_K_M", ("q4_k_m", "q4km")),
}
CPP_MODEL_CHOICES = list(CPP_MODEL_VARIANTS)
CPP_GGUF_REPO = "OpenMOSS-Team/MOSS-TTS-GGUF"  # legacy/pre-v1.5 preconverted repo; v1.5 is converted locally
CPP_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX"

CPP_VRAM_PRESETS = {
    "8 GB": {
        "model": "MOSS-TTS v1.5 — 8B Delay — Q4_K_M",
        "kv": "q4_0",
        "low_memory": True,
        "heads": "numpy",
        "audio_gpu": True,
        "description": "Minimum-VRAM staged pipeline. Official low-memory strategy: CPU/NumPy LM heads and GPU audio stages loaded only when needed.",
    },
    "12 GB": {
        "model": "MOSS-TTS v1.5 — 8B Delay — Q6_K",
        "kv": "q8_0",
        "low_memory": True,
        "heads": "numpy",
        "audio_gpu": True,
        "description": "Higher-quality GGUF while keeping the official staged-loading strategy and CPU LM heads.",
    },
    "16 GB": {
        "model": "MOSS-TTS v1.5 — 8B Delay — Q8_0",
        "kv": "q8_0",
        "low_memory": True,
        "heads": "numpy",
        "audio_gpu": True,
        "description": "Q8 backbone with staged encoder/backbone/decoder loading. LM heads stay in RAM to preserve GPU headroom.",
    },
    "24 GB": {
        "model": "MOSS-TTS v1.5 — 8B Delay — Q8_0",
        "kv": "q8_0",
        "low_memory": False,
        "heads": "torch",
        "audio_gpu": True,
        "description": "Maximum-throughput resident pipeline. Backbone, CUDA LM heads and ONNX audio tokenizer remain GPU-resident.",
    },
}
CPP_VRAM_PRESET_CHOICES = list(CPP_VRAM_PRESETS)


TTS_MODELS = {
    "MOSS-TTS v1.5 — 8B Delay • Latest flagship • 31 languages": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "MOSS-TTS Local v1.5 — 4B Local • Latest • 48 kHz stereo • 31 languages": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "MOSS-TTS 1.0 — 8B Delay • Legacy flagship • 20 languages": "OpenMOSS-Team/MOSS-TTS",
    "MOSS-TTS Local 1.0 — 1.7B Local • Legacy • 24 kHz mono • 20 languages": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}
SPECIAL_MODELS = {
    "MOSS-TTSD v1.0  •  8B  •  ~21–24 GB VRAM": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "MOSS-VoiceGenerator  •  1.7B  •  ~8–10 GB VRAM": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "MOSS-SoundEffect  •  8B  •  ~21–24 GB VRAM": "OpenMOSS-Team/MOSS-SoundEffect",
    "MOSS-SoundEffect v2.0  •  DiT 1.3B + Qwen3 1.7B  •  ~10–14 GB VRAM": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
    "MOSS-TTS Realtime  •  1.7B  •  ~8–10 GB VRAM": "OpenMOSS-Team/MOSS-TTS-Realtime",
}
LORA_PROFILES = {
    "MOSS-TTS v1.5 — 8B Delay • Latest • LoRA": (
        "delay-v1.5-8b", "OpenMOSS-Team/MOSS-TTS-v1.5", "moss_tts_delay", "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    ),
    "MOSS-TTS Local v1.5 — 4B Local • Latest • 48 kHz stereo • LoRA": (
        "local-v1.5-4b", "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", "moss_tts_local_v1.5", "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
    ),
    "MOSS-TTS 1.0 — 8B Delay • Legacy • LoRA": (
        "delay-v1.0-8b", "OpenMOSS-Team/MOSS-TTS", "moss_tts_delay", "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    ),
    "MOSS-TTS Local 1.0 — 1.7B Local • Legacy • 24 kHz mono • LoRA": (
        "local-v1.0-1.7b", "OpenMOSS-Team/MOSS-TTS-Local-Transformer", "moss_tts_local", "OpenMOSS-Team/MOSS-Audio-Tokenizer"
    ),
}
DIALOGUE_MAX_TURNS = 12
CHUNK_CHOICES = ["None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]
DIALOGUE_NATIVE_MAX_SPEAKERS = 5


VRAM_PRESETS = {
    "24 GB": {"budget_gb": 24.0, "headroom_gb": 1.5},
    "32 GB+": {"budget_gb": 32.0, "headroom_gb": 1.5},
}


CSS = """
.title-section { border-bottom: 1px solid var(--border-color-primary); margin-bottom: 6px; padding-bottom: 4px; align-items:center !important; }
.tabs { margin-top: 2px; }
.form-section { padding: 14px; border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--block-background-fill); }
.button-primary { background: #2563eb !important; color: white !important; }
.button-stop, .red-btn { background: #dc3545 !important; color: white !important; }
.green-btn { background: #28a745 !important; color: white !important; }
.global-toolbar { padding: 10px 12px; border: 1px solid var(--border-color-primary); border-radius: 10px; background: var(--block-background-fill); margin-bottom: 12px; }
.global-toolbar button { min-height: 38px !important; }
.audio-safe-space { overflow: visible !important; padding-bottom: 20px !important; border: 0 !important; box-shadow: none !important; }
.audio-safe-space .wave, .audio-safe-space [data-testid="waveform"] { margin-bottom: 26px !important; }
.output-clean, .output-clean > div, .output-clean .wrap { border: 0 !important; box-shadow: none !important; }
.output-path textarea { border: 0 !important; background: var(--input-background-fill) !important; min-height: 40px !important; }
.project-strip { padding: 12px; border-radius: 10px; border: 1px solid var(--border-color-primary); margin-bottom: 12px; }
.console-accordion, .console-accordion > div { border-radius: 8px !important; }
.cmd-mirror { display:block; width:100%; height:333px; border:0; border-radius:8px; overflow:hidden; }
.dialogue-toolbar { margin-bottom: 8px; }
.dialogue-turn-card { padding: 10px 12px !important; border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important; margin-bottom: 8px !important; }
.dialogue-turn-card .gr-row { align-items: end !important; }
.dialogue-actions button { min-width: 40px !important; padding-left: 8px !important; padding-right: 8px !important; }
.progress-card { padding:8px 10px; border-radius:8px; border:1px solid var(--border-color-primary); margin-top:8px; }
.train-bar { height:10px; width:100%; border-radius:6px; background:#202638; overflow:hidden; margin-top:6px; } .train-fill { height:100%; background:#3b82f6; }
.small-note { opacity:.78; font-size:.9em; }
.tab-subtitle { opacity:.82; margin:0 0 4px 0 !important; padding:0 !important; }
.compact-status { margin:0 !important; padding:0 !important; min-height:0 !important; }
.title-section .prose, .title-section h1 { margin:0 !important; padding:0 !important; }
.title-section button { min-height:36px !important; white-space:nowrap; }
"""

MODEL_LOCK = threading.RLock()
MODEL_STATE = {"repo": None, "model": None, "processor": None, "extra": None}
TRAIN_PROCESS = None
TENSORBOARD_PROCESS = None
TRAIN_STATE = {"pct": 0.0, "text": "Idle", "running": False}
TRAIN_LOCK = threading.RLock()
ASR = ASRManager(ROOT / "models" / "asr")
ASR_CHOICES = list(WHISPER_MODELS)
ASR_LANGUAGES = list(WHISPER_LANGS)
HF_SNAPSHOT_LOCK = threading.RLock()
HF_SNAPSHOTS: dict[str, str] = {}
CMD_MIRROR_LINES = deque(maxlen=1200)
CMD_MIRROR_LOCK = threading.Lock()
CMD_MIRROR_CURRENT = ""
CMD_MIRROR_OVERWRITE = False


class _CmdMirror:
    def __init__(self, stream):
        self.stream = stream
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        if data:
            with CMD_MIRROR_LOCK:
                _mirror_write(str(data))
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def _clean_cmd_line(line: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).rstrip()


def _mirror_commit(line: str) -> None:
    line = _clean_cmd_line(line)
    if line.strip():
        CMD_MIRROR_LINES.append(line)


def _mirror_write(data: str) -> None:
    global CMD_MIRROR_CURRENT, CMD_MIRROR_OVERWRITE
    for ch in data:
        if ch == "\r":
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = True
            continue
        if ch == "\n":
            _mirror_commit(CMD_MIRROR_CURRENT)
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = False
            continue
        if CMD_MIRROR_OVERWRITE:
            CMD_MIRROR_CURRENT = ""
            CMD_MIRROR_OVERWRITE = False
        CMD_MIRROR_CURRENT += ch


def _install_cmd_mirror():
    if getattr(sys.stdout, "_moss_cmd_mirror", False):
        return

    class LockedMirror(_CmdMirror):
        _moss_cmd_mirror = True

    sys.stdout = LockedMirror(sys.stdout)
    sys.stderr = LockedMirror(sys.stderr)


_HTML_ESC = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})
_ATTR_ESC = str.maketrans({"&": "&amp;", '"': "&quot;", "<": "&lt;", ">": "&gt;"})


def _line_color(line: str) -> str:
    low = line.lower()
    if "error" in low or "traceback" in low or "exception" in low or "failed" in low:
        return "#f87171"
    if "warn" in low or "deprecated" in low:
        return "#fbbf24"
    if "download" in low or "snapshot" in low or "%|" in line:
        return "#60a5fa"
    if "train" in low or "epoch" in low or "lora" in low:
        return "#a78bfa"
    if "saved" in low or "ready" in low or "done" in low or "complete" in low:
        return "#4ade80"
    return "#cccccc"


def console_html():
    with CMD_MIRROR_LOCK:
        lines = list(CMD_MIRROR_LINES)
        current = _clean_cmd_line(CMD_MIRROR_CURRENT)
    if current.strip():
        lines.append(current)
    display = lines[-160:] if lines else ["Idle."]
    rows = []
    for line in display:
        safe = line.translate(_HTML_ESC)
        rows.append(f'<div style="color:{_line_color(line)};white-space:pre;line-height:1.55">{safe}</div>')
    content = "\n".join(rows)
    srcdoc = f"""<!doctype html><html><head><style>
html,body{{margin:0;background:#111;color:#ccc;font-family:Consolas,ui-monospace,monospace;font-size:12px;}}
#wrap{{height:333px;border-radius:8px;border:1px solid #333;overflow:hidden;box-sizing:border-box;}}
#body{{height:333px;overflow:auto;padding:8px 20px 8px 12px;box-sizing:border-box;scrollbar-width:thin;scrollbar-color:#555 transparent;}}
#body::-webkit-scrollbar{{width:5px;height:5px}} #body::-webkit-scrollbar-thumb{{background:#555;border-radius:3px}}
</style></head><body><div id="wrap"><div id="body">{content}<div id="anchor"></div></div></div>
<script>const b=document.getElementById('body'); b.onscroll=()=>{{window._paused=!(b.scrollTop+b.clientHeight>=b.scrollHeight-40);}}; if(!window._paused)b.scrollTop=b.scrollHeight; setTimeout(()=>{{if(!window._paused)b.scrollTop=b.scrollHeight;}},50);</script>
</body></html>"""
    return f'<iframe class="cmd-mirror" scrolling="no" srcdoc="{srcdoc.translate(_ATTR_ESC)}"></iframe>'


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _install_torchaudio_soundfile_io():
    """Use libsndfile for path-based audio reads used by MOSS processors.

    TorchAudio 2.9 routes load() through TorchCodec. On Windows that makes
    simple voice-reference WAV loading depend on external FFmpeg DLL discovery.
    MOSS only needs a waveform tensor + sample rate here, so use the already
    pinned soundfile runtime for filesystem audio and retain TorchAudio for DSP.
    """
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
    except Exception:
        return
    if getattr(torchaudio.load, "_moss_soundfile_io", False):
        return
    original_load = torchaudio.load
    original_save = torchaudio.save

    def load_audio(uri, *args, **kwargs):
        if isinstance(uri, (str, os.PathLike, Path)):
            data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
            tensor = torch.from_numpy(np.asarray(data).T.copy())
            return tensor, int(sr)
        return original_load(uri, *args, **kwargs)

    def save_audio(uri, src, sample_rate, *args, **kwargs):
        if isinstance(uri, (str, os.PathLike, Path)):
            tensor = src.detach().cpu() if hasattr(src, "detach") else torch.as_tensor(src)
            arr = tensor.numpy()
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim == 2:
                arr = arr.T
            sf.write(str(uri), arr, int(sample_rate))
            return None
        return original_save(uri, src, sample_rate, *args, **kwargs)

    load_audio._moss_soundfile_io = True
    save_audio._moss_soundfile_io = True
    torchaudio.load = load_audio
    torchaudio.save = save_audio


def _save_audio_file(path, audio, sample_rate):
    import numpy as np
    import soundfile as sf
    import torch
    tensor = audio.detach().cpu() if hasattr(audio, "detach") else torch.as_tensor(audio)
    arr = tensor.numpy()
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2:
        arr = arr.T
    sf.write(str(path), np.asarray(arr), int(sample_rate))


_install_cmd_mirror()
_install_torchaudio_soundfile_io()

def training_progress_html():
    with TRAIN_LOCK:
        pct = max(0.0, min(100.0, float(TRAIN_STATE.get("pct", 0.0))))
        text = str(TRAIN_STATE.get("text", "Idle"))
        running = bool(TRAIN_STATE.get("running", False))
    state = "Running" if running else "Idle / Finished"
    return f'<div class="progress-card"><b>{html.escape(state)}</b> — {html.escape(text)}<div class="train-bar"><div class="train-fill" style="width:{pct:.1f}%"></div></div><div>{pct:.1f}%</div></div>'


def resolve_hf_snapshot(repo_or_path: str) -> str:
    """Resolve a Hugging Face repo ID to a real local snapshot path.

    MOSS remote processors convert their input to pathlib.Path internally. On
    Windows that turns repo IDs such as ``org/model`` into ``org\\model``
    and breaks nested AutoConfig/AutoTokenizer calls. Materializing the repo
    first keeps the official processor code unchanged while ensuring it only
    receives a valid filesystem path.
    """
    value = str(repo_or_path).strip()
    if not value:
        raise ValueError("Empty model repository/path.")

    local = Path(value).expanduser()
    if local.exists():
        return str(local.resolve())

    # Repo IDs always use forward slashes, regardless of host OS. This also
    # repairs an accidentally normalized repo ID before it reaches HF Hub.
    repo_id = value.replace("\\", "/")
    with HF_SNAPSHOT_LOCK:
        cached = HF_SNAPSHOTS.get(repo_id)
        if cached and Path(cached).exists():
            return cached

        from huggingface_hub import snapshot_download
        log(f"Downloading/checking model snapshot: {repo_id}")
        snapshot = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(CACHE_DIR / "hub"),
        )
        snapshot = str(Path(snapshot).resolve())
        HF_SNAPSHOTS[repo_id] = snapshot
        log(f"Model snapshot ready: {repo_id}")
        return snapshot


def resolve_attn(use_flash_attention: bool = True):
    import torch
    if use_flash_attention and torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def _effective_attention_backend(model) -> str:
    """Best-effort verification of the attention implementation actually selected by Transformers."""
    candidates = [model, getattr(model, "language_model", None), getattr(model, "model", None)]
    for obj in candidates:
        cfg = getattr(obj, "config", None) if obj is not None else None
        if cfg is None:
            continue
        for name in ("_attn_implementation", "attn_implementation", "_attn_implementation_internal"):
            value = getattr(cfg, name, None)
            if value:
                return str(value)
    names = {m.__class__.__name__.lower() for m in model.modules()} if hasattr(model, "modules") else set()
    if any("flashattention2" in n or "flashattention" in n for n in names):
        return "flash_attention_2"
    return "unknown"


def log_attention_verification(model, requested: str) -> None:
    effective = _effective_attention_backend(model)
    if effective == "unknown":
        log(f"Attention requested: {requested}; effective backend could not be introspected.")
    elif requested == effective or (requested == "sdpa" and "sdpa" in effective.lower()):
        log(f"Attention requested: {requested}; effective: {effective}.")
    else:
        log(f"WARNING: Attention requested: {requested}; effective: {effective} (fallback/change detected).")


def _iso_from_label(label: str) -> str | None:
    if not label or label == "Auto":
        return None
    m = re.search(r"\(([a-z]{2})\)$", label)
    return m.group(1) if m else label


def _cpp_hf_token(explicit_token: str | None = None) -> str | None:
    """Return the token used for gated GGUF access.

    A token pasted in the GUI takes priority. If it is empty, fall back to the
    environment / Hugging Face login token / common global Windows token paths.
    """
    if explicit_token and str(explicit_token).strip():
        return str(explicit_token).strip()

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token
        value = (get_token() or "").strip()
        if value:
            return value
    except Exception:
        pass

    home = Path.home()
    candidates = [
        home / ".cache" / "huggingface" / "token",
        home / ".huggingface" / "token",
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8").strip()
                if value:
                    return value
        except OSError:
            pass
    return None


def _quant_from_gguf_filename(filename: str) -> str | None:
    """Map both first-class and legacy OpenMOSS GGUF filenames to a GUI quant."""
    name = Path(filename).name.lower()
    if not name.endswith(".gguf"):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", name)
    checks = (
        ("Q5_K_M", ("q5_k_m", "q5km")),
        ("Q4_K_M", ("q4_k_m", "q4km")),
        ("Q8_0", ("q8_0", "q8")),
        ("Q6_K", ("q6_k", "q6k")),
        ("F16", ("f16",)),
    )
    for quant, aliases in checks:
        if any(alias in normalized or alias in name for alias in aliases):
            return quant
    return None


def _cpp_variant_label(quant: str) -> str:
    return f"MOSS-TTS 8B — {quant}"


def _discover_cpp_repo_files(token: str | None = None) -> list[str]:
    from huggingface_hub import list_repo_files
    return list_repo_files(CPP_GGUF_REPO, token=token)


def check_hf_token_access(explicit_token: str):
    """Validate the token against the v1.5 model + ONNX codec used by the C++ path."""
    token=(explicit_token or "").strip()
    if not token:
        return "❌ No token entered. Paste a Hugging Face **Read** token first."
    try:
        from huggingface_hub import HfApi, list_repo_files
        api=HfApi(token=token); user=api.whoami(token=token)
        name=user.get("name") or user.get("fullname") or "authenticated user"
    except Exception as exc:
        return f"❌ Token authentication failed: `{exc}`"
    results=[]
    for repo in (CPP_V15_REPO, CPP_CODEC_REPO):
        try:
            files=list_repo_files(repo,token=token)
            results.append(f"✅ `{repo}` ({len(files)} files visible)")
        except Exception as exc:
            results.append(f"❌ `{repo}`: {exc}")
    return f"Authenticated as **{name}**.\n\n" + "\n\n".join(results)


def _discover_cpp_model_choices(explicit_token: str | None = None) -> list[str]:
    """v1.5 C++ variants are produced from the official v1.5 checkpoint locally."""
    return list(CPP_MODEL_CHOICES)


def _resolve_cpp_gguf_filename(repo_files: list[str], model_label: str) -> str:
    if model_label not in CPP_MODEL_VARIANTS:
        raise gr.Error(f"Unknown official MOSS GGUF variant: {model_label}")

    quant_name = CPP_MODEL_VARIANTS[model_label][0]
    ggufs = [f for f in repo_files if f.lower().endswith(".gguf")]

    # First-class files (e.g. MOSS_TTS_F16.gguf / MOSS_TTS_Q4_K_M.gguf)
    # and the older backend layout (MOSS_TTS_backbone_q4km.gguf) are both valid.
    matches = [f for f in ggufs if _quant_from_gguf_filename(f) == quant_name]
    if matches:
        # Prefer current first-class names over a legacy "backbone" filename
        # when both are present.
        matches.sort(key=lambda f: ("backbone" in Path(f).name.lower(), len(f), f.lower()))
        return matches[0]

    available = []
    for f in ggufs:
        q = _quant_from_gguf_filename(f)
        if q:
            available.append(f"{q}: {f}")

    raise gr.Error(
        f"The official repository does not currently expose a GGUF matching {quant_name}. "
        f"Visible GGUF files: {', '.join(available) if available else 'none'}. "
        "If repository access is gated, accepting the terms is not enough by itself: "
        "this application must also authenticate with an HF token."
    )


CPP_V15_REPO = "OpenMOSS-Team/MOSS-TTS-v1.5"
CPP_V15_ROOT = ROOT / "models" / "moss-tts-cpp-v1.5"
CPP_V15_LORA_DIR = CPP_V15_ROOT / "lora_adapters"
CPP_V15_LORA_DIR.mkdir(parents=True, exist_ok=True)


def _llama_cpp_source_dir() -> Path:
    source = RUNTIME_DIR / "llama-cpp" / "source"
    if not (source / "convert_hf_to_gguf.py").is_file() or not (source / "convert_lora_to_gguf.py").is_file():
        raise gr.Error(
            "The llama.cpp source tools required for first-time MOSS-TTS v1.5 / LoRA conversion are missing. "
            "Run tools\\build_llamacpp_cuda_runtime.bat once; it keeps the pinned OpenMOSS/llama.cpp source under .runtime\\llama-cpp\\source."
        )
    return source


def _llama_quantize_exe() -> Path:
    candidates = [
        RUNTIME_DIR / "llama-cpp" / "bin" / "llama-quantize.exe",
        RUNTIME_DIR / "llama-cpp" / "build" / "bin" / "Release" / "llama-quantize.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise gr.Error(
        "llama-quantize.exe is missing from the packaged C++ runtime. "
        "Re-run tools\\build_llamacpp_cuda_runtime.bat from this patch to add v1.5 quantization support."
    )


def _run_conversion_command(cmd: list[str], label: str) -> None:
    log(label + ": " + subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log("[convert] " + line)
    rc = proc.wait()
    if rc != 0:
        raise gr.Error(f"{label} failed with exit code {rc}. Review the live console.")


def _ensure_cpp_v15_components() -> tuple[Path, Path, Path]:
    """Extract the official v1.5 Delay checkpoint into llama.cpp component groups."""
    extracted = CPP_V15_ROOT / "extracted"
    backbone_dir = extracted / "qwen3_backbone"
    embeddings = extracted / "embeddings"
    lm_heads = extracted / "lm_heads"
    tokenizer = backbone_dir
    complete = (
        (backbone_dir / "config.json").is_file()
        and embeddings.is_dir() and any(embeddings.glob("*.npy"))
        and lm_heads.is_dir() and any(lm_heads.glob("*.npy"))
        and (tokenizer / "tokenizer.json").is_file()
    )
    if complete:
        return backbone_dir, embeddings, lm_heads

    CPP_V15_ROOT.mkdir(parents=True, exist_ok=True)
    local_repo = Path(resolve_hf_snapshot(CPP_V15_REPO))
    extractor = ROOT / "moss_tts_upstream" / "moss_tts_delay" / "llama_cpp" / "conversion" / "extract_weights.py"
    if not extractor.is_file():
        raise gr.Error(f"Official MOSS llama.cpp extraction script is missing: {extractor}")
    log("Preparing official MOSS-TTS v1.5 components for llama.cpp (one-time)...")
    _run_conversion_command(
        [sys.executable, str(extractor), "--model", str(local_repo), "--output", str(extracted)],
        "MOSS-TTS v1.5 weight extraction",
    )
    return backbone_dir, embeddings, lm_heads


def _ensure_cpp_v15_backbone(model_label: str) -> tuple[Path, Path, Path, Path]:
    if model_label not in CPP_MODEL_VARIANTS:
        raise gr.Error("Select a MOSS-TTS v1.5 C++ quantization.")
    quant = CPP_MODEL_VARIANTS[model_label][0]
    backbone_dir, embeddings, lm_heads = _ensure_cpp_v15_components()
    source = _llama_cpp_source_dir()
    model_dir = CPP_V15_ROOT / "gguf"
    model_dir.mkdir(parents=True, exist_ok=True)
    f16 = model_dir / "MOSS_TTS_v1.5_F16.gguf"
    if not f16.is_file():
        _run_conversion_command(
            [sys.executable, str(source / "convert_hf_to_gguf.py"), str(backbone_dir), "--outfile", str(f16), "--outtype", "f16"],
            "MOSS-TTS v1.5 backbone GGUF conversion",
        )
    if quant == "F16":
        return f16, embeddings, lm_heads, backbone_dir
    out = model_dir / f"MOSS_TTS_v1.5_{quant}.gguf"
    if not out.is_file():
        quantizer = _llama_quantize_exe()
        _run_conversion_command([str(quantizer), str(f16), str(out), quant], f"MOSS-TTS v1.5 {quant} quantization")
    return out, embeddings, lm_heads, backbone_dir


def _normalize_moss_peft_for_qwen(adapter_dir: Path, cache_dir: Path, backbone_dir: Path) -> Path:
    """Rewrite MOSS wrapper tensor names into standard Qwen3 PEFT names for llama.cpp's converter."""
    from safetensors.torch import load_file, save_file
    src_weights = adapter_dir / "adapter_model.safetensors"
    src_cfg = adapter_dir / "adapter_config.json"
    if not src_weights.is_file() or not src_cfg.is_file():
        raise gr.Error(f"LoRA adapter is incomplete: {adapter_dir}")
    normalized = cache_dir / "normalized_peft"
    normalized.mkdir(parents=True, exist_ok=True)
    dst_weights = normalized / "adapter_model.safetensors"
    dst_cfg = normalized / "adapter_config.json"
    stamp = normalized / ".source_signature"
    signature = f"{src_weights.stat().st_size}:{src_weights.stat().st_mtime_ns}:{src_cfg.stat().st_mtime_ns}"
    if dst_weights.is_file() and dst_cfg.is_file() and stamp.is_file() and stamp.read_text().strip() == signature:
        return normalized

    tensors = load_file(str(src_weights), device="cpu")
    remapped = {}
    changed = 0
    for name, tensor in tensors.items():
        new_name = name
        replacements = (
            ("base_model.model.language_model.layers.", "base_model.model.model.layers."),
            ("base_model.model.model.language_model.layers.", "base_model.model.model.layers."),
            ("language_model.layers.", "model.layers."),
        )
        for old, new in replacements:
            if old in new_name:
                new_name = new_name.replace(old, new)
        if new_name != name:
            changed += 1
        remapped[new_name] = tensor
    if changed == 0:
        log("GGUF LoRA normalization: tensor names already look Qwen3-compatible.")
    else:
        log(f"GGUF LoRA normalization: remapped {changed}/{len(remapped)} tensors from MOSS wrapper names to Qwen3 names.")
    save_file(remapped, str(dst_weights))
    cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = str(backbone_dir)
    dst_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    stamp.write_text(signature, encoding="utf-8")
    return normalized


def _sanitize_gguf_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    if not cleaned:
        raise gr.Error("Enter a name for the converted GGUF LoRA.")
    return cleaned[:120]


def _cpp_lora_metadata_path(gguf_path: Path) -> Path:
    return gguf_path.with_suffix(".json")


def _converted_cpp_lora_choices():
    choices = [("None", "")]
    if not CPP_V15_LORA_DIR.is_dir():
        return choices
    items = []
    for gguf in CPP_V15_LORA_DIR.glob("*.gguf"):
        meta_path = _cpp_lora_metadata_path(gguf)
        meta = {}
        try:
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if meta.get("base_repo") not in ("", None, CPP_V15_REPO):
            continue
        label = str(meta.get("display_name") or gguf.stem)
        source = str(meta.get("source_checkpoint") or "")
        if source:
            source_name = Path(source).name
            label = f"{label}  •  {source_name}"
        items.append((gguf.stat().st_mtime_ns, label, str(gguf.resolve())))
    items.sort(key=lambda x: (-x[0], x[1].lower()))
    choices.extend((label, path) for _, label, path in items)
    return choices


def refresh_cpp_lora_choices():
    return gr.update(choices=_converted_cpp_lora_choices(), value="")


def refresh_cpp_checkpoint_choices():
    return gr.update(choices=lora_adapter_choices_for_repo(CPP_V15_REPO), value="")


def _require_gguf_conversion_dependencies():
    missing = []
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        missing.append("sentencepiece")
    if missing:
        raise gr.Error(
            "GGUF conversion dependencies are missing: "
            + ", ".join(missing)
            + ". Run 1- install.bat once after applying this patch."
        )
    _llama_cpp_source_dir()
    _llama_quantize_exe()


def prepare_cpp_v15_backbone_ui(model_label: str, hf_token: str = ""):
    """Explicitly prepare one v1.5 backbone quantization + ONNX codec."""
    from huggingface_hub import snapshot_download

    _require_gguf_conversion_dependencies()
    if model_label not in CPP_MODEL_VARIANTS:
        raise gr.Error("Select a MOSS-TTS v1.5 backbone quantization.")

    quant = CPP_MODEL_VARIANTS[model_label][0]
    log(f"GGUF Conversion: preparing MOSS-TTS v1.5 Delay 8B backbone ({quant})...")
    gguf, embeddings, lm_heads, backbone_dir = _ensure_cpp_v15_backbone(model_label)

    codec_dir = CPP_V15_ROOT / "onnx"
    codec_dir.mkdir(parents=True, exist_ok=True)
    token = _cpp_hf_token(hf_token)
    snapshot_download(repo_id=CPP_CODEC_REPO, local_dir=str(codec_dir), token=token)

    enc = codec_dir / "encoder.onnx"
    dec = codec_dir / "decoder.onnx"
    missing = [
        str(p) for p in
        (gguf, embeddings, lm_heads, backbone_dir / "config.json", enc, dec)
        if not p.exists()
    ]
    if missing:
        raise gr.Error("C++ preparation completed incompletely. Missing: " + ", ".join(missing))

    play_completion_chime()
    return (
        f"✅ **MOSS-TTS v1.5 Delay 8B — {quant}** is ready for llama.cpp inference.\n\n"
        f"`{gguf}`"
    )


def _prepared_cpp_backbone(model_label: str) -> tuple[Path, Path, Path, Path]:
    """Return already-prepared assets. Never converts or downloads."""
    if model_label not in CPP_MODEL_VARIANTS:
        raise gr.Error("Select a MOSS-TTS v1.5 C++ quantization.")
    quant = CPP_MODEL_VARIANTS[model_label][0]

    extracted = CPP_V15_ROOT / "extracted"
    backbone_dir = extracted / "qwen3_backbone"
    embeddings = extracted / "embeddings"
    lm_heads = extracted / "lm_heads"
    model_dir = CPP_V15_ROOT / "gguf"
    gguf = model_dir / f"MOSS_TTS_v1.5_{quant}.gguf"

    # F16 uses the same naming convention.
    required = [
        gguf,
        backbone_dir / "config.json",
        embeddings,
        lm_heads,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise gr.Error(
            "This C++ model has not been prepared yet. Open **GGUF Conversion**, "
            "choose the desired quantization and press **Prepare v1.5 C++ Backbone**."
        )
    return gguf, embeddings, lm_heads, backbone_dir


def convert_cpp_lora_ui(adapter: str, custom_name: str):
    """Explicit PEFT -> GGUF LoRA conversion. Never called by inference."""
    _require_gguf_conversion_dependencies()

    if not adapter or not str(adapter).strip():
        raise gr.Error("Select a v1.5 Delay 8B LoRA checkpoint or final adapter.")
    adapter_dir = Path(str(adapter)).expanduser().resolve()
    if not adapter_dir.is_dir():
        raise gr.Error(f"LoRA checkpoint does not exist: {adapter_dir}")

    cfg_path = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapter_model.safetensors"
    if not cfg_path.is_file() or not weights.is_file():
        raise gr.Error("The selected checkpoint must contain adapter_config.json and adapter_model.safetensors.")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = str(cfg.get("base_model_name_or_path", "")).lower().replace("\\", "/")
    if (
        "moss-tts-local" in base
        or (
            "moss-tts-v1.5" not in base
            and "models--openmoss-team--moss-tts-v1.5" not in base
            and "moss-tts" in base
        )
    ):
        raise gr.Error("Only MOSS-TTS v1.5 Delay 8B LoRA adapters can be converted for this C++ backend.")

    # LoRA conversion needs the extracted Qwen3 v1.5 base metadata.
    # If inference has not prepared it yet, obtain/extract that metadata automatically.
    extracted = CPP_V15_ROOT / "extracted"
    backbone_dir = extracted / "qwen3_backbone"
    if not (backbone_dir / "config.json").is_file():
        log("GGUF LoRA conversion: preparing MOSS-TTS v1.5 base metadata...")
        # Use Q4_K_M only as the public label required by the preparation helper;
        # the LoRA conversion itself remains independent from the backbone quantization.
        _ensure_cpp_v15_backbone("MOSS-TTS v1.5 — 8B Delay — Q4_K_M")

    display_name = (custom_name or "").strip()
    filename = _sanitize_gguf_name(display_name)
    out = CPP_V15_LORA_DIR / f"{filename}.gguf"
    meta_path = _cpp_lora_metadata_path(out)

    if out.exists():
        raise gr.Error(
            f"A converted adapter named '{filename}' already exists. Choose another name or remove the old file first."
        )

    work_dir = CPP_V15_LORA_DIR / f".work_{filename}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_moss_peft_for_qwen(adapter_dir, work_dir, backbone_dir)
    source = _llama_cpp_source_dir()

    log(f"GGUF Conversion: converting LoRA '{adapter_dir.name}' -> '{filename}.gguf'...")
    try:
        _run_conversion_command(
            [
                sys.executable,
                str(source / "convert_lora_to_gguf.py"),
                str(normalized),
                "--base",
                str(backbone_dir),
                "--outfile",
                str(out),
                "--outtype",
                "f16",
            ],
            f"MOSS-TTS v1.5 LoRA GGUF conversion ({filename})",
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not out.is_file():
        raise gr.Error("LoRA conversion completed without producing the requested GGUF file.")

    signature = hashlib.sha256(
        f"{weights.stat().st_size}:{weights.stat().st_mtime_ns}:{cfg_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    meta = {
        "format_version": 1,
        "display_name": display_name,
        "filename": out.name,
        "base_repo": CPP_V15_REPO,
        "source_checkpoint": str(adapter_dir),
        "source_signature": signature,
        "created": datetime.now().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    play_completion_chime()
    return (
        f"✅ Converted **{display_name}** to GGUF LoRA.\n\n`{out}`",
        gr.update(choices=_converted_cpp_lora_choices(), value=str(out.resolve())),
    )


def _resolve_preconverted_cpp_lora(adapter: str) -> Path | None:
    if not adapter or not str(adapter).strip():
        return None
    path = Path(str(adapter)).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise gr.Error(
            "The llama.cpp backend accepts only LoRA adapters converted in the **GGUF Conversion** tab."
        )
    try:
        path.relative_to(CPP_V15_LORA_DIR.resolve())
    except ValueError:
        raise gr.Error("Select a GGUF LoRA registered by this Easy GUI.")
    return path

def _download_cpp_assets(model_label: str, explicit_token: str | None = None):
    """Automatically prepare the official v1.5 Delay C++ assets on first inference."""
    from huggingface_hub import snapshot_download

    _require_gguf_conversion_dependencies()
    gguf, embeddings, lm_heads, backbone_dir = _ensure_cpp_v15_backbone(model_label)

    codec_dir = CPP_V15_ROOT / "onnx"
    codec_dir.mkdir(parents=True, exist_ok=True)
    enc = codec_dir / "encoder.onnx"
    dec = codec_dir / "decoder.onnx"
    if not enc.is_file() or not dec.is_file():
        token = _cpp_hf_token(explicit_token)
        log("Downloading/checking official MOSS-TTS v1.5 ONNX audio codec...")
        snapshot_download(
            repo_id=CPP_CODEC_REPO,
            local_dir=str(codec_dir),
            token=token,
            allow_patterns=["encoder.onnx", "decoder.onnx"],
        )

    if not enc.is_file() or not dec.is_file():
        raise gr.Error("The official ONNX audio codec could not be prepared.")

    return gguf, embeddings, lm_heads, backbone_dir, codec_dir


def load_cpp_tts(model_label: str, kv_type: str, gpu_layers: int, use_flash_attention: bool, hf_token: str = "", low_memory: bool = False, heads_backend: str = "torch", use_gpu_audio: bool = True, adapter: str = ""):
    if model_label not in CPP_MODEL_VARIANTS:
        raise gr.Error("Select one of the MOSS-TTS v1.5 Delay GGUF variants exposed by the llama.cpp backend.")
    quant_name = CPP_MODEL_VARIANTS[model_label][0]
    adapter_key = str(Path(adapter).resolve()) if adapter and Path(adapter).exists() else ""
    runtime_key = f"cpp-v1.5|{quant_name}|lora={adapter_key}|kv={kv_type}|ngl={int(gpu_layers)}|fa={int(bool(use_flash_attention))}|lowmem={int(bool(low_memory))}|heads={heads_backend}|audio_gpu={int(bool(use_gpu_audio))}"
    with MODEL_LOCK:
        if MODEL_STATE["repo"] == runtime_key and MODEL_STATE["model"] is not None:
            return MODEL_STATE["model"]
        unload_model()
        gguf, embeddings, lm_heads, backbone_dir, codec_dir = _download_cpp_assets(model_label, hf_token)
        lora_gguf = _resolve_preconverted_cpp_lora(adapter)
        bridge = RUNTIME_DIR / "llama-cpp" / "bridge" / "backbone_bridge.dll"
        dll_dir = RUNTIME_DIR / "llama-cpp" / "bin"
        if not bridge.exists():
            raise gr.Error("A compatible OpenMOSS llama.cpp Windows CUDA runtime is not installed. Run tools\\build_llamacpp_cuda_runtime.bat.")
        os.environ["MOSS_LLAMA_CPP_BRIDGE"] = str(bridge)
        os.environ["MOSS_LLAMA_CPP_DLL_DIR"] = str(dll_dir)
        if os.name == "nt" and dll_dir.exists():
            try:
                dll_handle = os.add_dll_directory(str(dll_dir))
                _WINDOWS_DLL_DIR_HANDLES.append(dll_handle)
                import ctypes
                for dll_name in ("ggml-base.dll", "ggml.dll", "ggml-cpu.dll", "ggml-cuda.dll", "llama.dll"):
                    dll_path = dll_dir / dll_name
                    if dll_path.exists(): ctypes.WinDLL(str(dll_path))
                bridge_lib = ctypes.WinDLL(str(bridge))
                if lora_gguf and not hasattr(bridge_lib, "bridge_set_lora"):
                    raise gr.Error("Your backbone_bridge.dll is from an older patch. Rebuild the llama.cpp runtime to enable dynamic GGUF LoRA.")
            except (AttributeError, OSError) as exc:
                raise gr.Error(f"Unable to load the packaged llama.cpp CUDA runtime: {exc}") from exc
        from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig
        import onnxruntime as ort
        import moss_audio_tokenizer.onnx.inference as moss_onnx_inference
        def _easy_gui_load_ort_session(onnx_path, use_gpu):
            opts=ort.SessionOptions(); available=set(ort.get_available_providers()); providers=[]
            if use_gpu and "CUDAExecutionProvider" in available: providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return ort.InferenceSession(str(onnx_path),sess_options=opts,providers=providers)
        moss_onnx_inference._load_ort_session=_easy_gui_load_ort_session
        enc=codec_dir/"encoder.onnx"; dec=codec_dir/"decoder.onnx"
        for required in (gguf,enc,dec,embeddings,lm_heads,backbone_dir/"tokenizer.json"):
            if not required.exists(): raise gr.Error(f"Required llama.cpp v1.5 asset is missing: {required}")
        cfg=PipelineConfig(
            backbone_gguf=str(gguf), lora_gguf=str(lora_gguf or ""), lora_scale=1.0,
            embedding_dir=str(embeddings), lm_head_dir=str(lm_heads), tokenizer_dir=str(backbone_dir),
            audio_backend="onnx", audio_encoder_onnx=str(enc), audio_decoder_onnx=str(dec),
            heads_backend=heads_backend,n_ctx=4096,n_batch=512,n_threads=max(1,min(8,(os.cpu_count() or 4)//2)),
            n_gpu_layers=int(gpu_layers),max_new_tokens=2048,use_gpu_audio=bool(use_gpu_audio),low_memory=bool(low_memory),
            kv_cache_type_k=kv_type,kv_cache_type_v=kv_type,flash_attn="enabled" if use_flash_attention else "disabled",
        )
        lora_msg=f", LoRA={Path(adapter).name}" if lora_gguf else ""
        log(f"Loading MOSS-TTS v1.5 llama.cpp CUDA: {quant_name}{lora_msg}, KV={kv_type}, GPU layers={gpu_layers}, FlashAttention={use_flash_attention}...")
        pipeline=LlamaCppPipeline(cfg)
        MODEL_STATE.update(repo=runtime_key,model=pipeline,processor=None,extra={"lora_gguf":str(lora_gguf or "")})
        log(f"MOSS-TTS v1.5 llama.cpp backend ready: {quant_name}{lora_msg}.")
        return pipeline


def _cpp_vram_preset_updates(preset: str):
    cfg = CPP_VRAM_PRESETS.get(preset, CPP_VRAM_PRESETS["24 GB"])
    return (
        gr.update(value=cfg["model"]),
        gr.update(value=cfg["kv"]),
        gr.update(value=bool(cfg["low_memory"])),
        gr.update(value="CPU / NumPy" if cfg["heads"] == "numpy" else "GPU / Torch"),
        gr.update(value=bool(cfg["audio_gpu"])),
        cfg["description"],
    )


def _backend_model_adapter_updates(backend: str, hf_token: str = ""):
    """Backend -> model + LoRA choices + C++ controls visibility."""
    is_cpp = str(backend).startswith("llama.cpp")
    if is_cpp:
        choices = _discover_cpp_model_choices(hf_token or "")
        default = choices[-1]
        adapter_choices = _converted_cpp_lora_choices()
        return (
            gr.update(choices=choices, value=default),
            gr.update(choices=adapter_choices, value="", interactive=True),
            gr.update(visible=True),
        )
    return (
        gr.update(choices=list(TTS_MODELS), value=list(TTS_MODELS)[0]),
        gr.update(choices=lora_adapter_choices_for_model(list(TTS_MODELS)[0]), value="", interactive=True),
        gr.update(visible=False),
    )


def _tts_backend_updates(backend: str, hf_token: str = ""):
    """TTS additionally constrains generation mode and toggles llama.cpp controls."""
    model_update, adapter_update, cpp_controls_update = _backend_model_adapter_updates(backend, hf_token)
    if str(backend).startswith("llama.cpp"):
        mode_update = gr.update(
            choices=["Direct / Voice Clone"],
            value="Direct / Voice Clone",
            interactive=False,
        )
    else:
        mode_update = gr.update(
            choices=["Direct / Voice Clone", "Continuation + Voice Clone"],
            value="Direct / Voice Clone",
            interactive=True,
        )
    return model_update, adapter_update, cpp_controls_update, mode_update

def maybe_compile_model(model, use_torch_compile: bool = False, compile_mode: str = "default"):
    """No-op by design. Generic torch.compile is not part of the official MOSS-TTS/Local/TTSD/VoiceGenerator inference paths."""
    return model

def _release_runtime_memory(reset_compiler: bool = False):
    gc.collect()
    try:
        import torch
        if reset_compiler:
            try:
                torch._dynamo.reset()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()


def play_completion_chime():
    wav = ASSETS_DIR / "chime.wav"
    if not wav.exists() or os.name != "nt":
        return
    try:
        import winsound
        winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception as exc:
        log(f"Chime warning: {exc}")


def unload_model():
    with MODEL_LOCK:
        old_model = MODEL_STATE.get("model")
        old_processor = MODEL_STATE.get("processor")
        old_extra = MODEL_STATE.get("extra")
        was_compiled = bool(getattr(old_model, "_moss_compiled_backbone", False)) if old_model is not None else False
        MODEL_STATE.update(repo=None, model=None, processor=None, extra=None)
        if old_model is not None and hasattr(old_model, "close"):
            try:
                old_model.close()
            except Exception as exc:
                log(f"Model close warning: {exc}")
        # Drop GPU tokenizer/codec/inferencer references before collecting.
        if old_processor is not None and hasattr(old_processor, "audio_tokenizer"):
            try:
                old_processor.audio_tokenizer = None
            except Exception:
                pass
        del old_model, old_processor, old_extra
        _release_runtime_memory(reset_compiler=was_compiled)
    return "All loaded models were released."


def load_standard(repo: str, adapter: str = "", use_flash_attention: bool = True, use_torch_compile: bool = False, compile_mode: str = "default"):
    import torch
    from transformers import AutoModel, AutoProcessor
    from peft import PeftModel
    with MODEL_LOCK:
        runtime_key = f"{repo}|{adapter}|flash={int(bool(use_flash_attention))}"
        if MODEL_STATE["repo"] == runtime_key and MODEL_STATE["model"] is not None:
            return MODEL_STATE["model"], MODEL_STATE["processor"]
        unload_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        attention = resolve_attn(use_flash_attention)
        log(f"Loading {repo} with {attention}...")
        local_repo = resolve_hf_snapshot(repo)
        processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
        if getattr(processor, "audio_tokenizer", None) is not None:
            processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        model = AutoModel.from_pretrained(local_repo, trust_remote_code=True, attn_implementation=attention, dtype=dtype).to(device).eval()
        log_attention_verification(model, attention)
        if adapter and Path(adapter).exists():
            log(f"Applying LoRA adapter: {adapter}")
            model = PeftModel.from_pretrained(model, adapter).to(device).eval()
        MODEL_STATE.update(repo=runtime_key, model=model, processor=processor, extra=None)
        return model, processor



def split_by_periods(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [chunk.strip() for chunk in re.split(r"(?<=\.)\s+", text) if chunk.strip()]


def paragraph_sentence_split(text: str, max_chars: int = 120) -> list[str]:
    """Higgs-style automatic long-text splitter."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    for paragraph in paragraphs or [text]:
        paragraph = re.sub(r"\s+", " ", paragraph)
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def split_long_text(text: str, mode: str) -> list[str]:
    """Same long-form splitting rules used by the reference Easy GUI."""
    text = (text or "").strip()
    if not text:
        return []
    if mode == "None":
        return [text]
    if mode == "Paragraph/Sentence Auto":
        return paragraph_sentence_split(text)
    if mode == "Periods":
        return split_by_periods(text)
    if mode == "Paragraphs":
        return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if mode == "Lines":
        return [p.strip() for p in text.splitlines() if p.strip()]
    if mode == "Speaker turns":
        chunks = []
        current = []
        for line in text.splitlines():
            if re.match(r"^\s*\[?SPEAKER\d+\]?", line, flags=re.IGNORECASE) and current:
                chunks.append("\n".join(current).strip())
                current = [line.strip()]
            elif line.strip():
                current.append(line.strip())
        if current:
            chunks.append("\n".join(current).strip())
        return chunks or [text]
    return [text]


def _audio_to_time_major_array(audio):
    import numpy as np
    if hasattr(audio, "detach"):
        arr = audio.detach().cpu().float().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Torch convention is usually [channels, frames].
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        return arr
    return np.squeeze(arr)


def _concatenate_audio_chunks(chunks, sample_rate: int, gap_seconds: float):
    import numpy as np
    kept = [_audio_to_time_major_array(c) for c in chunks if c is not None]
    kept = [c for c in kept if getattr(c, "size", 0)]
    if not kept:
        raise RuntimeError("No audio chunks were generated.")

    first = kept[0]
    channels = 1 if first.ndim == 1 else first.shape[1]
    normalized = []
    for arr in kept:
        if channels == 1:
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
        else:
            if arr.ndim == 1:
                arr = np.repeat(arr[:, None], channels, axis=1)
            elif arr.shape[1] != channels:
                if arr.shape[1] == 1:
                    arr = np.repeat(arr, channels, axis=1)
                else:
                    arr = arr[:, :channels]
        normalized.append(np.asarray(arr, dtype=np.float32))

    silence_frames = max(0, int(round(float(sample_rate) * max(0.0, float(gap_seconds)))))
    silence = (
        np.zeros(silence_frames, dtype=np.float32)
        if channels == 1
        else np.zeros((silence_frames, channels), dtype=np.float32)
    )
    joined = []
    for i, arr in enumerate(normalized):
        if i and silence_frames:
            joined.append(silence)
        joined.append(arr)
    return np.concatenate(joined, axis=0)


def _decode_output_audio(processor, outputs):
    messages = processor.decode(outputs)
    if not messages:
        raise RuntimeError("The processor returned no decoded audio.")
    audio = messages[0].audio_codes_list[0]
    return _audio_to_time_major_array(audio), int(processor.model_config.sampling_rate)



def save_decoded(processor, outputs, prefix: str):
    import torchaudio
    messages = processor.decode(outputs)
    if not messages:
        raise RuntimeError("The processor returned no decoded audio.")
    audio = messages[0].audio_codes_list[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
        audio = audio.transpose(0, 1)
    out = OUTPUTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    _save_audio_file(out, audio, int(processor.model_config.sampling_rate))
    return str(out)


def generate_tts(model_label, backend, hf_token, generation_mode, text, language, voice_name, reference_audio, reference_transcript, expected_tokens, max_new_tokens, temperature, top_p, top_k, repetition_penalty, seed, adapter, cpp_kv, cpp_gpu_layers, cpp_low_memory, cpp_heads_backend, cpp_audio_gpu, chunk_mode, chunk_silence, use_flash_attention, use_torch_compile, compile_mode, progress=gr.Progress()):
    import torch

    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text to synthesize.")

    chunks = split_long_text(text, chunk_mode or "None")
    if not chunks:
        raise gr.Error("No usable text chunks were found.")
    chunked = len(chunks) > 1

    if chunked:
        log(
            f"Long text: {len(chunks)} chunks · {chunk_mode} · "
            f"{float(chunk_silence):.2f}s silence between chunks"
        )

    progress(0.06, desc="Loading model")
    voice_audio, voice_transcript = resolve_voice(VOICES_DIR, voice_name)
    reference_audio = reference_audio or voice_audio
    reference_transcript = (reference_transcript or voice_transcript or "").strip()

    if backend.startswith("llama.cpp"):
        if generation_mode != "Direct / Voice Clone":
            raise gr.Error("Continuation mode is not exposed by the bundled official llama.cpp pipeline. Use Transformers for Continuation.")

        progress(0.14, desc="Loading llama.cpp CUDA backend")
        pipeline = load_cpp_tts(
            model_label, cpp_kv, int(cpp_gpu_layers), use_flash_attention,
            hf_token or "", bool(cpp_low_memory),
            "numpy" if str(cpp_heads_backend).startswith("CPU") else "torch",
            bool(cpp_audio_gpu), adapter or ""
        )
        pipeline.sampling_config.audio_temperature = float(temperature)
        pipeline.sampling_config.audio_top_p = float(top_p)
        pipeline.sampling_config.audio_top_k = int(top_k)
        pipeline.sampling_config.audio_repetition_penalty = float(repetition_penalty)

        audio_chunks = []
        for chunk_index, chunk in enumerate(chunks, 1):
            if chunked:
                progress(
                    0.18 + 0.68 * ((chunk_index - 1) / max(1, len(chunks))),
                    desc=f"Generating chunk {chunk_index}/{len(chunks)}"
                )
                log(f"[long-form] llama.cpp chunk {chunk_index}/{len(chunks)} · {len(chunk)} chars")
            wav = pipeline.generate(
                text=chunk,
                reference_audio=reference_audio or None,
                language=_iso_from_label(language),
                tokens=int(expected_tokens) if int(expected_tokens or 0) > 0 else None,
                max_new_tokens=int(max_new_tokens),
            )
            audio_chunks.append(_audio_to_time_major_array(wav))
            del wav

        final_wav = (
            _concatenate_audio_chunks(audio_chunks, 24000, float(chunk_silence))
            if chunked else audio_chunks[0]
        )
        out = OUTPUTS_DIR / f"moss_tts_cpp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        _save_audio_file(out, final_wav, 24000)
        del final_wav, audio_chunks
        _release_runtime_memory()
        progress(1.0, desc="Done")
        play_completion_chime()
        return str(out), str(out)

    if model_label not in TTS_MODELS:
        raise gr.Error("Select a Transformers/PyTorch model for this backend.")

    repo = TTS_MODELS[model_label]
    model, processor = load_standard(
        repo, adapter or "", use_flash_attention, use_torch_compile, compile_mode
    )
    device = next(model.parameters()).device
    audio_chunks = []
    sample_rate = int(processor.model_config.sampling_rate)

    for chunk_index, chunk in enumerate(chunks, 1):
        if chunked:
            progress(
                0.16 + 0.68 * ((chunk_index - 1) / max(1, len(chunks))),
                desc=f"Generating chunk {chunk_index}/{len(chunks)}"
            )
            log(f"[long-form] PyTorch chunk {chunk_index}/{len(chunks)} · {len(chunk)} chars")

        if generation_mode == "Continuation + Voice Clone":
            if not reference_audio:
                raise gr.Error("Continuation requires reference audio.")
            if not reference_transcript:
                raise gr.Error("Continuation requires the transcript of the reference audio.")
            kwargs = {"text": reference_transcript + chunk, "reference": [reference_audio]}
            if language and language != "Auto":
                kwargs["language"] = language
            if int(expected_tokens or 0) > 0:
                kwargs["tokens"] = int(expected_tokens)
            conversation = [[
                processor.build_user_message(**kwargs),
                processor.build_assistant_message(audio_codes_list=[reference_audio]),
            ]]
            batch = processor(conversation, mode="continuation")
        else:
            kwargs = {"text": chunk}
            if language and language != "Auto":
                kwargs["language"] = language
            if reference_audio:
                kwargs["reference"] = [reference_audio]
            if int(expected_tokens or 0) > 0:
                kwargs["tokens"] = int(expected_tokens)
            conversation = [[processor.build_user_message(**kwargs)]]
            batch = processor(conversation, mode="generation")

        if repo.endswith("MOSS-TTS-Local-Transformer-v1.5") and int(seed) >= 0:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(temperature),
                audio_top_p=float(top_p),
                audio_top_k=int(top_k),
                audio_repetition_penalty=float(repetition_penalty),
            )

        audio, sample_rate = _decode_output_audio(processor, outputs)
        audio_chunks.append(audio)
        del outputs, batch

    progress(0.9, desc="Joining audio" if chunked else "Saving audio")
    final_wav = (
        _concatenate_audio_chunks(audio_chunks, sample_rate, float(chunk_silence))
        if chunked else audio_chunks[0]
    )
    out = OUTPUTS_DIR / f"moss_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    _save_audio_file(out, final_wav, sample_rate)
    del final_wav, audio_chunks
    _release_runtime_memory()
    progress(1.0, desc="Done")
    play_completion_chime()
    return str(out), str(out)



def generate_voice(text, instruction, language, max_new_tokens, temperature, top_p, top_k, repetition_penalty, use_flash_attention, use_torch_compile, compile_mode, progress=gr.Progress()):
    import torch
    if not text.strip() or not instruction.strip():
        raise gr.Error("Text and voice instruction are required.")
    if language and language != "Auto":
        log(f"VoiceGenerator language: {language} (official model supports Chinese and English; language is inferred from text).")
    progress(0.08, desc="Loading model")
    model, processor = load_standard(SPECIAL_MODELS["MOSS-VoiceGenerator  •  1.7B  •  ~8–10 GB VRAM"], use_flash_attention=use_flash_attention, use_torch_compile=use_torch_compile, compile_mode=compile_mode)
    conv = [[processor.build_user_message(text=text.strip(), instruction=instruction.strip())]]
    batch = processor(conv, mode="generation")
    device = next(model.parameters()).device
    with torch.no_grad():
        progress(0.45, desc="Generating voice")
        outputs = model.generate(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), max_new_tokens=int(max_new_tokens), audio_temperature=float(temperature), audio_top_p=float(top_p), audio_top_k=int(top_k), audio_repetition_penalty=float(repetition_penalty))
    progress(0.92, desc="Decoding audio")
    out = save_decoded(processor, outputs, "moss_voice")
    del outputs, batch
    _release_runtime_memory()
    progress(1.0, desc="Done")
    play_completion_chime()
    return out, out


def generate_sfx(model_label, prompt, duration, steps, cfg_scale, sigma_shift, seed, max_new_tokens, temperature, top_p, top_k, repetition_penalty, use_flash_attention, use_torch_compile, compile_mode, progress=gr.Progress()):
    import torch
    if not prompt.strip():
        raise gr.Error("Enter a sound description.")
    repo = SPECIAL_MODELS[model_label]
    progress(0.08, desc="Loading sound model")
    if repo.endswith("MOSS-SoundEffect-v2.0"):
        local_path = resolve_hf_snapshot(repo)
        worker_python = RUNTIME_DIR / "sfx-v2-venv" / "Scripts" / "python.exe"
        worker_script = ROOT / "tools" / "sfx_v2_infer.py"
        if not worker_python.exists():
            raise gr.Error("MOSS-SoundEffect v2.0 runtime is missing. Run 1- install.bat again.")
        seconds = max(0.1, min(30.0, float(duration)))
        out = OUTPUTS_DIR / f"moss_sfx_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        progress(0.25, desc="Starting isolated SoundEffect v2 runtime")
        cmd = [
            str(worker_python), str(worker_script),
            "--model", local_path,
            "--prompt", prompt.strip(),
            "--seconds", str(seconds),
            "--steps", str(int(steps)),
            "--cfg-scale", str(float(cfg_scale)),
            "--sigma-shift", str(float(sigma_shift)),
            "--seed", str(int(seed)),
            "--output", str(out),
        ]
        worker_env = os.environ.copy()
        # Official infer_from_pipeline.sh defaults TORCHDYNAMO_DISABLE=1.
        # The checkbox opts into the DiT's upstream @torch.compile path.
        worker_env["TORCHDYNAMO_DISABLE"] = "0" if bool(use_torch_compile) else "1"
        process = subprocess.Popen(cmd, cwd=str(ROOT), env=worker_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
        rc = process.wait()
        if rc != 0 or not out.exists():
            raise RuntimeError(f"MOSS-SoundEffect v2.0 worker failed with exit code {rc}.")
        progress(1.0, desc="Done")
        play_completion_chime()
        return str(out), str(out)

    model, processor = load_standard(repo, use_flash_attention=use_flash_attention, use_torch_compile=use_torch_compile, compile_mode=compile_mode)
    duration_tokens = max(1, int(float(duration) * 12.5))
    conv = [[processor.build_user_message(ambient_sound=prompt.strip(), tokens=duration_tokens)]]
    batch = processor(conv, mode="generation")
    device = next(model.parameters()).device
    with torch.no_grad():
        progress(0.45, desc="Generating sound")
        outputs = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )
    out = save_decoded(processor, outputs, "moss_sfx")
    del outputs, batch
    _release_runtime_memory()
    progress(1.0, desc="Done")
    play_completion_chime()
    return out, out


def _dialogue_unpack(values):
    rows=[]
    for i in range(DIALOGUE_MAX_TURNS):
        voice = values[i*2] if i*2 < len(values) else "None"
        text = values[i*2+1] if i*2+1 < len(values) else ""
        rows.append([voice or "None", text or ""])
    return rows


def _dialogue_updates(count, rows, message="Dialogue rows updated."):
    count=max(1,min(int(count),DIALOGUE_MAX_TURNS))
    rows=(rows+[["None",""] for _ in range(DIALOGUE_MAX_TURNS)])[:DIALOGUE_MAX_TURNS]
    flat=[]
    for voice,text in rows:
        flat.extend([gr.update(value=voice),gr.update(value=text)])
    vis=[gr.update(visible=i<count) for i in range(DIALOGUE_MAX_TURNS)]
    return [count,*flat,*vis,message]


def dialogue_reset_rows():
    rows=[["None",""] for _ in range(DIALOGUE_MAX_TURNS)]
    return _dialogue_updates(2,rows,"Dialogue reset to two empty turns.")


def dialogue_clear_rows(count,*values):
    rows=_dialogue_unpack(values)
    for i in range(min(int(count),DIALOGUE_MAX_TURNS)): rows[i]=["None",""]
    return _dialogue_updates(count,rows,"Visible dialogue turns cleared.")


def dialogue_compact_rows(count,*values):
    rows=_dialogue_unpack(values)
    active=[r for r in rows[:int(count)] if str(r[1]).strip()]
    if not active: active=[["None",""]]
    return _dialogue_updates(len(active),active,"Removed empty turns.")


def dialogue_add_after(index,count,*values):
    rows=_dialogue_unpack(values); count=int(count)
    if count>=DIALOGUE_MAX_TURNS: return _dialogue_updates(count,rows,f"Maximum of {DIALOGUE_MAX_TURNS} turns reached.")
    rows.insert(min(int(index)+1,count),["None",""]); rows=rows[:DIALOGUE_MAX_TURNS]
    return _dialogue_updates(count+1,rows,f"Added a turn after Turn {int(index)+1}.")


def dialogue_clone_row(index,count,*values):
    rows=_dialogue_unpack(values); count=int(count); idx=min(int(index),max(0,count-1))
    if count>=DIALOGUE_MAX_TURNS: return _dialogue_updates(count,rows,f"Maximum of {DIALOGUE_MAX_TURNS} turns reached.")
    rows.insert(idx+1,list(rows[idx])); rows=rows[:DIALOGUE_MAX_TURNS]
    return _dialogue_updates(count+1,rows,f"Cloned Turn {idx+1}.")


def dialogue_delete_row(index,count,*values):
    rows=_dialogue_unpack(values); count=max(1,int(count)); idx=min(int(index),count-1)
    if count==1:
        rows[0]=["None",""]; return _dialogue_updates(1,rows,"Last turn cleared.")
    rows.pop(idx); rows.append(["None",""])
    return _dialogue_updates(count-1,rows,f"Deleted Turn {idx+1}.")


def dialogue_clear_row(index,count,*values):
    rows=_dialogue_unpack(values); rows[int(index)]=["None",""]
    return _dialogue_updates(count,rows,f"Cleared Turn {int(index)+1}.")


def dialogue_move_row(index,direction,count,*values):
    rows=_dialogue_unpack(values); count=int(count); idx=int(index); dest=idx+int(direction)
    if 0<=idx<count and 0<=dest<count:
        rows[idx],rows[dest]=rows[dest],rows[idx]
        return _dialogue_updates(count,rows,f"Moved Turn {idx+1} {'up' if direction<0 else 'down'}.")
    return _dialogue_updates(count,rows,"Turn is already at that boundary.")


def generate_dialogue(max_new_tokens, temperature, top_p, top_k, repetition_penalty, use_flash_attention, use_torch_compile, compile_mode, row_count, *values, progress=gr.Progress()):
    import torch, torchaudio
    rows=_dialogue_unpack(values)[:max(1,min(int(row_count),DIALOGUE_MAX_TURNS))]
    active=[(v,t.strip()) for v,t in rows if str(t).strip()]
    if not active: raise gr.Error("Dialogue has no turns with text.")
    if any(not v or v=="None" for v,_ in active): raise gr.Error("Every dialogue turn with text requires a saved voice.")
    unique=[]
    for voice,_ in active:
        if voice not in unique: unique.append(voice)
    if len(unique)>DIALOGUE_NATIVE_MAX_SPEAKERS:
        raise gr.Error("MOSS-TTSD supports at most 5 distinct speakers per dialogue.")
    speaker_id={v:i+1 for i,v in enumerate(unique)}
    refs_meta=[]
    for voice in unique:
        audio_path,transcript=resolve_voice(VOICES_DIR,voice)
        if not audio_path: raise gr.Error(f"Saved voice '{voice}' has no valid audio file.")
        if not str(transcript or '').strip(): raise gr.Error(f"Saved voice '{voice}' needs a transcript for TTSD.")
        refs_meta.append((audio_path,str(transcript).strip()))
    progress(0.1,desc="Loading dialogue model")
    model,processor=load_standard(SPECIAL_MODELS["MOSS-TTSD v1.0  •  8B  •  ~21–24 GB VRAM"],use_flash_attention=use_flash_attention,use_torch_compile=use_torch_compile,compile_mode=compile_mode)
    sr=int(processor.model_config.sampling_rate); wavs=[]
    for audio_path,_ in refs_meta:
        wav,sri=torchaudio.load(audio_path); wav=wav.mean(0,keepdim=True)
        if sri!=sr: wav=torchaudio.functional.resample(wav,sri,sr)
        wavs.append(wav)
    refs=processor.encode_audios_from_wav(wavs,sampling_rate=sr)
    prompt_audio=processor.encode_audios_from_wav([torch.cat(wavs,dim=-1)],sampling_rate=sr)[0]
    prefixes=" ".join(f"[S{i}] {t}" for i,(_,t) in enumerate(refs_meta,1))
    dialogue_text=" ".join(f"[S{speaker_id[v]}] {t}" for v,t in active)
    conv=[[processor.build_user_message(text=f"{prefixes} {dialogue_text}",reference=refs),processor.build_assistant_message(audio_codes_list=[prompt_audio])]]
    batch=processor(conv,mode="continuation"); device=next(model.parameters()).device
    progress(0.42,desc="Generating dialogue")
    with torch.no_grad():
        outputs=model.generate(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device),max_new_tokens=int(max_new_tokens),audio_temperature=float(temperature),audio_top_p=float(top_p),audio_top_k=int(top_k),audio_repetition_penalty=float(repetition_penalty))
    progress(0.92,desc="Decoding audio"); out=save_decoded(processor,outputs,"moss_ttsd")
    del outputs, batch, wavs, refs, prompt_audio
    _release_runtime_memory()
    progress(1.0,desc="Done")
    play_completion_chime()
    return out,out,"Dialogue complete."


def generate_classic_dialogue(model_label, backend, hf_token, language, pause_seconds, max_new_tokens, temperature, top_p, top_k, repetition_penalty, adapter, cpp_kv, cpp_gpu_layers, cpp_low_memory, cpp_heads_backend, cpp_audio_gpu, use_flash_attention, use_torch_compile, compile_mode, row_count, *values, progress=gr.Progress()):
    import torch
    rows=_dialogue_unpack(values)[:max(1,min(int(row_count),DIALOGUE_MAX_TURNS))]
    active=[(v,t.strip()) for v,t in rows if str(t).strip()]
    if not active: raise gr.Error("Dialogue has no turns with text.")
    if any(not v or v=="None" for v,_ in active): raise gr.Error("Every dialogue turn with text requires a saved voice.")

    use_cpp = str(backend).startswith("llama.cpp")
    progress(0.05,desc="Loading TTS model")
    chunks=[]
    if use_cpp:
        pipeline=load_cpp_tts(model_label, cpp_kv, int(cpp_gpu_layers), use_flash_attention, hf_token or "", bool(cpp_low_memory), "numpy" if str(cpp_heads_backend).startswith("CPU") else "torch", bool(cpp_audio_gpu), adapter or "")
        pipeline.sampling_config.audio_temperature=float(temperature)
        pipeline.sampling_config.audio_top_p=float(top_p)
        pipeline.sampling_config.audio_top_k=int(top_k)
        pipeline.sampling_config.audio_repetition_penalty=float(repetition_penalty)
        sr=24000
        silence=torch.zeros(1,max(0,int(sr*float(pause_seconds))))
        for idx,(voice,text) in enumerate(active,1):
            ref,_=resolve_voice(VOICES_DIR,voice)
            if not ref: raise gr.Error(f"Saved voice '{voice}' has no valid audio file.")
            progress(0.08+0.78*(idx-1)/max(1,len(active)),desc=f"Generating turn {idx}/{len(active)}")
            wav=pipeline.generate(text=text,reference_audio=ref,language=_iso_from_label(language),max_new_tokens=int(max_new_tokens))
            if not torch.is_tensor(wav): wav=torch.as_tensor(wav)
            if wav.ndim==1: wav=wav.unsqueeze(0)
            chunks.append(wav.detach().cpu().float())
    else:
        if model_label not in TTS_MODELS:
            raise gr.Error("Select a Transformers/PyTorch model for this backend.")
        repo=TTS_MODELS[model_label]
        import torchaudio
        model,processor=load_standard(repo,adapter or "",use_flash_attention,use_torch_compile,compile_mode); device=next(model.parameters()).device
        sr=int(processor.model_config.sampling_rate); silence=torch.zeros(1,max(0,int(sr*float(pause_seconds))))
        for idx,(voice,text) in enumerate(active,1):
            ref,_=resolve_voice(VOICES_DIR,voice)
            if not ref: raise gr.Error(f"Saved voice '{voice}' has no valid audio file.")
            kwargs={"text":text,"reference":[ref]}
            if language and language!="Auto": kwargs["language"]=language
            batch=processor([[processor.build_user_message(**kwargs)]],mode="generation")
            progress(0.08+0.78*(idx-1)/max(1,len(active)),desc=f"Generating turn {idx}/{len(active)}")
            with torch.no_grad():
                outputs=model.generate(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device),max_new_tokens=int(max_new_tokens),audio_temperature=float(temperature),audio_top_p=float(top_p),audio_top_k=int(top_k),audio_repetition_penalty=float(repetition_penalty))
            decoded=processor.decode(outputs); wav=decoded[0] if isinstance(decoded,(list,tuple)) else decoded
            if isinstance(wav,(str,Path)):
                wav,sri=torchaudio.load(str(wav)); wav=wav.mean(0,keepdim=True)
                if sri!=sr: wav=torchaudio.functional.resample(wav,sri,sr)
            elif not torch.is_tensor(wav): wav=torch.as_tensor(wav)
            if wav.ndim==1: wav=wav.unsqueeze(0)
            chunks.append(wav.detach().cpu())
            del batch, outputs

    joined=[]
    for i,w in enumerate(chunks):
        if i and silence.numel(): joined.append(silence)
        joined.append(w)
    final=torch.cat(joined,dim=-1); out=OUTPUTS_DIR/f"moss_dialogue_classic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    import soundfile as sf
    sf.write(str(out),final.squeeze(0).numpy(),sr)
    del chunks, joined, final
    _release_runtime_memory()
    progress(1.0,desc="Done")
    play_completion_chime()
    return str(out),str(out),f"Dialogue complete. Generated {len(active)} turns."

def add_help_accordion(text):
    with gr.Accordion("📖 Quick Guide", open=False, elem_classes=["quick-guide"]):
        gr.Markdown(text)

def generate_realtime(text, language, voice_name, reference_audio, temperature, top_p, top_k, repetition_penalty, repetition_window, use_flash_attention, use_torch_compile, compile_mode, progress=gr.Progress()):
    import torch, torchaudio
    rt_dir = UPSTREAM / "moss_tts_realtime"
    if str(rt_dir) not in sys.path:
        sys.path.insert(0, str(rt_dir))
    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from inferencer import MossTTSRealtimeInference
    from transformers import AutoModel, AutoTokenizer
    if not text.strip():
        raise gr.Error("Enter text to synthesize.")
    if language and language != "Auto":
        log(f"Realtime language: {language} (official non-streaming inferencer detects it from text; selector is informational/validation).")
    progress(0.08, desc="Loading realtime model")
    voice_audio, _ = resolve_voice(VOICES_DIR, voice_name)
    reference_audio = reference_audio or voice_audio
    repo = SPECIAL_MODELS["MOSS-TTS Realtime  •  1.7B  •  ~8–10 GB VRAM"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    with MODEL_LOCK:
        runtime_key = f"{repo}|flash={int(bool(use_flash_attention))}"
        if MODEL_STATE["repo"] != runtime_key or MODEL_STATE["extra"] is None:
            unload_model()
            local_repo = resolve_hf_snapshot(repo)
            local_codec = resolve_hf_snapshot("OpenMOSS-Team/MOSS-Audio-Tokenizer")
            requested_attn = resolve_attn(use_flash_attention)
            model = MossTTSRealtime.from_pretrained(local_repo, attn_implementation=requested_attn, dtype=dtype).to(device).eval()
            log_attention_verification(model, requested_attn)
            tokenizer = AutoTokenizer.from_pretrained(local_repo)
            codec = AutoModel.from_pretrained(local_codec, trust_remote_code=True).to(device).eval()
            inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=5000, codec=codec, codec_sample_rate=24000, codec_encode_kwargs={"chunk_duration": 8})
            MODEL_STATE.update(repo=runtime_key, model=model, processor=None, extra=(inferencer, codec))
        inferencer, codec = MODEL_STATE["extra"]
    progress(0.45, desc="Generating realtime speech")
    result = inferencer.generate(text=[text.strip()], reference_audio_path=[reference_audio or ""], temperature=float(temperature), top_p=float(top_p), top_k=int(top_k), repetition_penalty=float(repetition_penalty), repetition_window=int(repetition_window), device=device)
    tokens = list(result)[0]
    output = torch.tensor(tokens).to(device)
    wav = codec.decode(output.permute(1, 0), chunk_duration=8)["audio"][0].cpu().detach()
    if wav.ndim == 1: wav = wav.unsqueeze(0)
    out = OUTPUTS_DIR / f"moss_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    _save_audio_file(out, wav, 24000)
    del tokens, output, wav
    _release_runtime_memory()
    progress(1.0, desc="Done")
    play_completion_chime()
    return str(out), str(out)


def transcribe_audio_ui(audio_path, asr_model, asr_language, asr_batch):
    if not audio_path:
        return "", "No audio selected."
    unload_model()
    try:
        text, status = ASR.transcribe(audio_path, asr_model, asr_language, int(asr_batch))
        play_completion_chime()
        return text, status
    finally:
        ASR.unload()


def load_voice_transcript_ui(name):
    _, transcript = resolve_voice(VOICES_DIR, name)
    return transcript or ""


def _powershell_dialog(script: str) -> str:
    if os.name != "nt":
        return ""
    try:
        cp = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if cp.returncode == 0:
            return (cp.stdout or "").strip().splitlines()[-1].strip() if (cp.stdout or "").strip() else ""
    except Exception as exc:
        log(f"Windows browse dialog warning: {exc}")
    return ""


def browse_dataset_folder(current_folder=""):
    """Native folder picker matching the published Easy GUI workflow."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial = current_folder if current_folder and Path(current_folder).is_dir() else str(ROOT)
        selected = filedialog.askdirectory(initialdir=initial)
        root.destroy()
        if selected:
            return selected
        return current_folder or ""
    except Exception as exc:
        log(f"Folder picker error: {exc}")
        return current_folder or ""



def scan_dataset(folder, language, reference_audio, transcribe_missing, asr_model, asr_language, asr_batch, progress=gr.Progress()):
    folder = Path(folder or "")
    if not folder.exists():
        raise gr.Error("Select an existing dataset folder.")
    progress(0.05, desc="Scanning audio files")
    audio_files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not audio_files:
        raise gr.Error("No supported audio files were found.")
    missing = [str(audio) for audio in audio_files if not audio.with_suffix(".txt").exists()]
    transcripts = {}
    if missing and transcribe_missing:
        unload_model()
        transcripts = ASR.transcribe_many(missing, asr_model, asr_language, int(asr_batch), progress_cb=lambda i,t,n: progress((i/max(1,t))*0.75+0.1, desc=n))
        ASR.unload()
    rows = []
    for audio in audio_files:
        txt = audio.with_suffix(".txt")
        text = txt.read_text(encoding="utf-8-sig").strip() if txt.exists() else transcripts.get(str(audio), "").strip()
        if text:
            if not txt.exists():
                txt.write_text(text + "\n", encoding="utf-8")
            rows.append([str(audio), text, "" if language == "Auto" else language, reference_audio or ""])
    if not rows:
        raise gr.Error("No usable transcripts were found. Add .txt sidecars or enable Faster-Whisper for missing transcripts.")
    note = f"Found {len(rows)} usable clips. Faster-Whisper filled {sum(1 for p in missing if transcripts.get(p, '').strip())} missing transcript(s)."
    play_completion_chime()
    return rows, note


def refresh_voice_choices():
    return gr.update(choices=list_voice_names(VOICES_DIR))


def load_voice_ui(name):
    audio, transcript = resolve_voice(VOICES_DIR, name)
    if audio and not Path(audio).is_file():
        audio = None
    return audio, transcript


def save_voice_ui(audio_path, name, transcript):
    status = save_voice_sample(VOICES_DIR, audio_path, name, transcript)
    choices = list_voice_names(VOICES_DIR)
    saved_name = Path(name or audio_path).stem if audio_path else "None"
    safe_value = saved_name if saved_name in choices else (choices[-1] if len(choices) > 1 else "None")
    return status, gr.update(choices=choices, value=safe_value), gr.update(choices=choices)


def delete_voice_ui(name):
    status = delete_voice(VOICES_DIR, name)
    choices = list_voice_names(VOICES_DIR)
    return status, gr.update(choices=choices, value="None"), gr.update(choices=choices, value="None")


def write_raw_jsonl(rows, profile_label):
    if not rows:
        raise gr.Error("Dataset table is empty.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_dir = DATASETS_DIR / stamp
    ds_dir.mkdir(parents=True, exist_ok=True)
    raw = ds_dir / "train_raw.jsonl"
    with raw.open("w", encoding="utf-8") as f:
        for row in rows:
            audio, text, language, ref = (list(row) + ["", "", "", ""])[:4]
            if not audio or not text:
                continue
            rec = {"audio": str(audio), "text": str(text)}
            if language: rec["language"] = str(language)
            if ref: rec["ref_audio"] = str(ref)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return raw


def prepare_dataset(rows, profile_label, progress=gr.Progress()):
    progress(0.05, desc="Preparing dataset")
    ASR.unload()
    profile, model_repo, package, codec_repo = LORA_PROFILES[profile_label]
    model_path = resolve_hf_snapshot(model_repo)
    codec_path = resolve_hf_snapshot(codec_repo)
    raw = write_raw_jsonl(rows, profile_label)
    prepared = raw.parent / "train_prepared.jsonl"
    script = UPSTREAM / package / "finetuning" / "prepare_data.py"
    cmd = [sys.executable, str(script), "--model-path", model_path, "--codec-path", codec_path, "--device", "cuda", "--input-jsonl", str(raw), "--output-jsonl", str(prepared), "--batch-size", "1"]
    log("Preparing dataset: " + subprocess.list2cmdline(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if proc.returncode != 0 or not prepared.exists():
        raise gr.Error("Dataset preparation failed. Review the console output.")
    progress(1.0, desc="Dataset prepared")
    play_completion_chime()
    return str(prepared), f"Prepared dataset: {prepared}"


def start_training(profile_label, prepared_jsonl, output_name, vram_preset, epochs, lr, micro_batch, grad_accum, lora_r, lora_alpha, lora_dropout, save_mode, save_steps, save_every_epochs, use_flash_attention, use_torch_compile, enable_eval_audio, eval_text, eval_reference_audio, eval_max_tokens, resume_checkpoint):
    global TRAIN_PROCESS
    budget = VRAM_PRESETS[vram_preset]["budget_gb"]
    physical = _gpu_vram_gb()
    if physical and physical + 0.25 < budget:
        raise gr.Error(f"{vram_preset} preset requires at least {budget:.0f} GB physical VRAM; detected {physical:.1f} GB.")
    if not prepared_jsonl or not Path(prepared_jsonl).exists():
        raise gr.Error("Prepare the dataset first.")
    profile, model_repo, _package, _codec_repo = LORA_PROFILES[profile_label]
    model_path = resolve_hf_snapshot(model_repo)
    output_name = (output_name or "moss_lora").strip().replace(" ", "_")
    output_dir = TRAINING_OUTPUTS_DIR / output_name
    schedule_steps_per_epoch, schedule_total_steps, schedule_text = estimate_training_schedule(
        prepared_jsonl, epochs, micro_batch, grad_accum
    )
    effective_save_steps = resolve_save_cadence(
        save_mode, save_steps, save_every_epochs, schedule_steps_per_epoch
    )
    log(
        f"Training schedule: {schedule_text}; checkpoint cadence={save_mode}; "
        f"effective save steps={effective_save_steps}"
    )
    cmd = [sys.executable, str(TRAINING_DIR / "train_lora.py"), "--profile", profile, "--model-path", model_path, "--train-jsonl", prepared_jsonl, "--output-dir", str(output_dir), "--epochs", str(int(epochs)), "--batch-size", str(int(micro_batch)), "--grad-accum", str(int(grad_accum)), "--learning-rate", str(float(lr)), "--lora-r", str(int(lora_r)), "--lora-alpha", str(int(lora_alpha)), "--lora-dropout", str(float(lora_dropout)), "--save-steps", str(int(effective_save_steps)), "--save-mode", str(save_mode), "--save-every-epochs", str(int(save_every_epochs))]
    if use_flash_attention:
        cmd.append("--use-flash-attention")
    if enable_eval_audio:
        cmd.extend([
            "--enable-eval-audio",
            "--eval-text", str(eval_text or "This is a MOSS-TTS training preview."),
            "--eval-max-new-tokens", str(int(eval_max_tokens)),
        ])
        if eval_reference_audio:
            eval_ref = Path(str(eval_reference_audio))
            if not eval_ref.is_file():
                raise gr.Error(f"Eval reference audio no longer exists: {eval_ref}")
            cmd.extend(["--eval-reference-audio", str(eval_ref.resolve())])
    if use_torch_compile:
        cmd.append("--torch-compile")
    if resume_checkpoint and str(resume_checkpoint) != "None":
        resume_path = Path(str(resume_checkpoint))
        if not resume_path.is_dir():
            raise gr.Error(f"Selected checkpoint no longer exists: {resume_path}")
        if not (resume_path / "adapter_config.json").is_file():
            raise gr.Error(f"Selected checkpoint has no PEFT adapter_config.json: {resume_path}")
        if not (resume_path / "trainer_state.pt").is_file():
            raise gr.Error(
                "This checkpoint contains adapter weights only and cannot perform a full training resume. "
                "Select None or a checkpoint created by patch 032+."
            )
        cmd.extend(["--resume-checkpoint", str(resume_path)])
    with TRAIN_LOCK:
        if TRAIN_PROCESS is not None and TRAIN_PROCESS.poll() is None:
            raise gr.Error("A LoRA training process is already running.")
        ASR.unload()
        unload_model()
        log("Starting LoRA: " + subprocess.list2cmdline(cmd))
        TRAIN_PROCESS = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        TRAIN_STATE.update(pct=0.0, text="Loading model / starting trainer...", running=True)
        threading.Thread(target=_training_reader, args=(TRAIN_PROCESS, int(epochs)), daemon=True).start()
    return f"Training started. PID {TRAIN_PROCESS.pid}. Output: {output_dir}", str(output_dir)


def estimate_training_schedule(prepared_jsonl, epochs_value, micro_batch_value, grad_accum_value):
    analysis = _analyze_prepared_training_jsonl(prepared_jsonl)
    sample_count = max(0, int(analysis["sample_count"]))
    effective_batch = max(1, int(micro_batch_value) * int(grad_accum_value))
    steps_per_epoch = max(1, math.ceil(max(sample_count, 1) / effective_batch))
    total_steps = max(1, int(epochs_value) * steps_per_epoch)
    return (
        steps_per_epoch,
        total_steps,
        f"≈{steps_per_epoch} optimizer steps/epoch · ≈{total_steps} total optimizer steps",
    )


def resolve_save_cadence(save_mode, save_steps, save_every_epochs, steps_per_epoch):
    if str(save_mode).startswith("Every N epoch"):
        n_epochs = max(1, int(save_every_epochs))
        return max(1, int(steps_per_epoch) * n_epochs)
    return max(1, int(save_steps))


def launch_tensorboard(output_name):
    global TENSORBOARD_PROCESS
    safe_name = (output_name or "my_moss_lora").strip().replace(" ", "_")
    run_dir = TRAINING_OUTPUTS_DIR / safe_name
    logdir = run_dir / "tensorboard"
    if not logdir.exists():
        raise gr.Error(
            f"No TensorBoard log exists yet for '{safe_name}'. Start training and wait until the trainer initializes."
        )

    url = "http://127.0.0.1:6006"
    if TENSORBOARD_PROCESS is not None and TENSORBOARD_PROCESS.poll() is None:
        webbrowser.open(url)
        return f"TensorBoard already running: {url}"

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        "127.0.0.1",
        "--port",
        "6006",
    ]
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        TENSORBOARD_PROCESS = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:
        raise gr.Error(f"Unable to launch TensorBoard: {exc}") from exc

    webbrowser.open(url)
    return f"TensorBoard started: {url}\nLogdir: {logdir}"


def stop_training():
    global TRAIN_PROCESS
    with TRAIN_LOCK:
        if TRAIN_PROCESS is None or TRAIN_PROCESS.poll() is not None:
            return "No training process is running."
        TRAIN_PROCESS.terminate()
        try:
            TRAIN_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            TRAIN_PROCESS.kill()
        TRAIN_PROCESS = None
        TRAIN_STATE.update(text="Training stopped by user.", running=False)
    return "Training stopped."


def delete_output_audios():
    removed = 0
    for path in OUTPUTS_DIR.glob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return f"Deleted {removed} output audio file(s)."


def delete_all_reference_samples():
    removed = 0
    for path in VOICES_DIR.iterdir():
        if path.is_file():
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return f"Deleted {removed} reference-library file(s).", gr.update(choices=list_voice_names(VOICES_DIR), value="None")


def model_status_text():
    repo = MODEL_STATE.get("repo")
    if repo:
        return f"**Runtime:** model loaded — `{repo.split('|', 1)[0]}`"
    return ""


def list_projects():
    return ["None", *sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())]


def _training_output_dir(output_name: str) -> Path:
    safe = (output_name or "moss_lora").strip().replace(" ", "_")
    return TRAINING_OUTPUTS_DIR / safe


def list_training_checkpoints(output_name: str):
    """Return checkpoint dropdown choices newest-first, plus None."""
    out = _training_output_dir(output_name)
    items = []
    if out.is_dir():
        for p in out.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            m = re.fullmatch(r"checkpoint-(\d+)", p.name)
            if not m:
                continue
            adapter_ok = (p / "adapter_config.json").is_file()
            state_ok = (p / "trainer_state.pt").is_file()
            if adapter_ok:
                step = int(m.group(1))
                label = f"{p.name}" + ("" if state_ok else "  (adapter only)")
                items.append((step, label, str(p)))
    items.sort(key=lambda x: x[0], reverse=True)
    return ["None", *[(label, path) for _, label, path in items]]


def refresh_training_checkpoints(output_name: str):
    choices = list_training_checkpoints(output_name)
    return gr.update(choices=choices, value="None")


def create_project(name):
    safe=re.sub(r"[^A-Za-z0-9._ -]+","_",(name or "").strip()).strip(" .")
    if not safe: raise gr.Error("Enter a project name.")
    d=PROJECTS_DIR/safe; d.mkdir(parents=True,exist_ok=True)
    (d/"project.json").write_text(json.dumps({"name":safe,"created":datetime.now().isoformat()},indent=2),encoding="utf-8")
    return gr.update(choices=list_projects(),value=safe), f"Project created: {safe}"


def delete_project(project):
    if not project or project == "None":
        raise gr.Error("Select a project to delete first.")
    target = PROJECTS_DIR / project
    if target.is_dir():
        shutil.rmtree(target)
    choices = list_projects()
    msg = f"Project deleted: {project}"
    return gr.update(choices=choices, value="None"), gr.update(choices=choices, value="None"), msg


def save_project_config(
    project,
    folder,
    language,
    profile,
    prepared,
    preset,
    output_name,
    reference_audio,
    transcribe_missing,
    asr_model,
    asr_language,
    asr_batch,
    epochs,
    lr,
    micro_batch,
    grad_accum,
    lora_r,
    lora_alpha,
    lora_dropout,
    save_steps,
    train_flash_attention,
    enable_eval_audio,
    eval_text,
    eval_reference_audio,
    eval_max_tokens,
    save_mode,
    save_every_epochs,
):
    if not project or project == "None":
        raise gr.Error("Create or select a project first.")
    d = PROJECTS_DIR / project
    d.mkdir(parents=True, exist_ok=True)

    ref = ""
    if reference_audio:
        try:
            rp = Path(str(reference_audio))
            if rp.is_file():
                ref = str(rp.resolve())
        except Exception:
            pass

    data = {
        "name": project,
        "dataset": {
            "folder": folder or "",
            "language": language or "Auto",
            "profile": profile,
            "prepared": prepared or "",
            "reference_audio": ref,
            "transcribe_missing": bool(transcribe_missing),
            "asr_model": asr_model,
            "asr_language": asr_language,
            "asr_batch": int(asr_batch),
        },
        "training": {
            "profile": profile,
            "preset": preset,
            "output_name": output_name or project,
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "micro_batch": int(micro_batch),
            "grad_accum": int(grad_accum),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "save_steps": int(save_steps),
            "flash_attention": bool(train_flash_attention),
            "enable_eval_audio": bool(enable_eval_audio),
            "eval_text": eval_text or "This is a MOSS-TTS training preview.",
            "eval_reference_audio": str(Path(str(eval_reference_audio)).resolve()) if eval_reference_audio and Path(str(eval_reference_audio)).is_file() else "",
            "eval_max_tokens": int(eval_max_tokens),
            "save_mode": save_mode or "Every N Epochs",
            "save_every_epochs": int(save_every_epochs),
        },
        "updated": datetime.now().isoformat(),
    }
    (d / "project.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return f"Project saved: {project}"


def _project_defaults(project="None"):
    profile = list(LORA_PROFILES)[0]
    default_asr = [x for x in ASR_CHOICES if "large-v3" in x and "distil" not in x][0]
    return {
        "folder": "",
        "language": "Auto",
        "profile": profile,
        "prepared": "",
        "reference_audio": None,
        "transcribe_missing": True,
        "asr_model": default_asr,
        "asr_language": "Auto-detect",
        "asr_batch": 8,
        "preset": "24 GB",
        "output_name": "my_moss_lora" if not project or project == "None" else project,
        "epochs": 30,
        "learning_rate": 5e-6,
        "micro_batch": 1,
        "grad_accum": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "save_steps": 250,
        "flash_attention": True,
        "enable_eval_audio": False,
        "eval_text": "This is a MOSS-TTS training preview.",
        "eval_reference_audio": None,
        "eval_max_tokens": 512,
        "save_mode": "Every N Epochs",
        "save_every_epochs": 1,
    }


def _read_project_state(project):
    state = _project_defaults(project)
    if not project or project == "None":
        return state, "No project selected."
    f = PROJECTS_DIR / project / "project.json"
    if not f.exists():
        return state, f"Project {project} has no saved configuration."

    d = json.loads(f.read_text(encoding="utf-8"))

    # Backward compatibility with projects created before the full-state format.
    if "dataset" not in d and "training" not in d:
        state.update({
            "folder": d.get("dataset_folder", ""),
            "language": d.get("language", "Auto"),
            "profile": d.get("profile", state["profile"]),
            "prepared": d.get("prepared", ""),
            "preset": d.get("preset", "24 GB"),
            "output_name": d.get("output_name", project),
        })
    else:
        ds = d.get("dataset", {}) or {}
        tr = d.get("training", {}) or {}
        state.update({
            "folder": ds.get("folder", ""),
            "language": ds.get("language", "Auto"),
            "profile": tr.get("profile", ds.get("profile", state["profile"])),
            "prepared": ds.get("prepared", ""),
            "reference_audio": ds.get("reference_audio") or None,
            "transcribe_missing": bool(ds.get("transcribe_missing", True)),
            "asr_model": ds.get("asr_model", state["asr_model"]),
            "asr_language": ds.get("asr_language", "Auto-detect"),
            "asr_batch": int(ds.get("asr_batch", 8)),
            "preset": tr.get("preset", "24 GB"),
            "output_name": tr.get("output_name", project),
            "epochs": int(tr.get("epochs", 3)),
            "learning_rate": float(tr.get("learning_rate", 1e-5)),
            "micro_batch": int(tr.get("micro_batch", 1)),
            "grad_accum": int(tr.get("grad_accum", 16)),
            "lora_r": int(tr.get("lora_r", 8)),
            "lora_alpha": int(tr.get("lora_alpha", 16)),
            "lora_dropout": float(tr.get("lora_dropout", 0.05)),
            "save_steps": int(tr.get("save_steps", 250)),
            "flash_attention": bool(tr.get("flash_attention", True)),
            "enable_eval_audio": bool(tr.get("enable_eval_audio", False)),
            "eval_text": tr.get("eval_text", state["eval_text"]),
            "eval_reference_audio": tr.get("eval_reference_audio") or None,
            "eval_max_tokens": int(tr.get("eval_max_tokens", 512)),
            "save_mode": tr.get("save_mode", "Every N Epochs"),
            "save_every_epochs": int(tr.get("save_every_epochs", 1)),
        })

    if state["preset"] not in VRAM_PRESETS:
        state["preset"] = "24 GB"
    if state["profile"] not in LORA_PROFILES:
        state["profile"] = list(LORA_PROFILES)[0]
    ref = state.get("reference_audio")
    if ref and not Path(str(ref)).is_file():
        state["reference_audio"] = None
    eval_ref = state.get("eval_reference_audio")
    if eval_ref and not Path(str(eval_ref)).is_file():
        state["eval_reference_audio"] = None
    return state, f"Loaded project: {project}"


def load_project_full(project):
    s, status = _read_project_state(project)
    return (
        s["folder"],
        s["language"],
        s["profile"],
        s["prepared"],
        s["reference_audio"],
        s["transcribe_missing"],
        s["asr_model"],
        s["asr_language"],
        s["asr_batch"],
        s["profile"],
        s["prepared"],
        s["preset"],
        s["output_name"],
        s["epochs"],
        s["learning_rate"],
        s["micro_batch"],
        s["grad_accum"],
        s["lora_r"],
        s["lora_alpha"],
        s["lora_dropout"],
        s["save_steps"],
        s["flash_attention"],
        s["enable_eval_audio"],
        s["eval_text"],
        s["eval_reference_audio"],
        s["eval_max_tokens"],
        s["save_mode"],
        s["save_every_epochs"],
        status,
        status,
    )



def _gpu_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _prepared_size_hint(prepared_jsonl):
    """Return a conservative sequence-size hint from prepared audio code lengths."""
    if not prepared_jsonl or not Path(prepared_jsonl).exists():
        return 1.0
    longest = 0
    try:
        with open(prepared_jsonl, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 256:
                    break
                rec = json.loads(line)
                codes = rec.get("audio_codes") or []
                if isinstance(codes, list):
                    def max_len(v):
                        if not isinstance(v, list): return 0
                        if not v: return 0
                        if isinstance(v[0], list): return max((max_len(x) for x in v), default=0)
                        return len(v)
                    longest = max(longest, max_len(codes))
    except Exception:
        return 1.0
    if longest <= 750: return 0.85
    if longest <= 1500: return 1.0
    if longest <= 3000: return 1.15
    return 1.3


def _analyze_prepared_training_jsonl(prepared_jsonl: str) -> dict:
    """Analyze the prepared MOSS JSONL using encoded audio length as duration evidence."""
    result = {
        "sample_count": 0,
        "total_seconds": 0.0,
        "avg_seconds": 5.0,
        "max_seconds": 0.0,
        "source": "defaults",
        "warnings": [],
    }
    p = Path(prepared_jsonl or "")
    if not p.is_file():
        result["warnings"].append("Prepared JSONL not found; assuming 5.0s average clips.")
        return result

    durations = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                result["sample_count"] += 1

                # prepare_data.py stores audio_codes. At ~12.5 audio frames/s,
                # the time axis gives a robust duration estimate without
                # reopening the original audio files.
                codes = record.get("audio_codes")
                frames = 0
                if isinstance(codes, list) and codes:
                    if isinstance(codes[0], list):
                        # Usually [time][n_vq]; tolerate [n_vq][time].
                        rows = len(codes)
                        cols = len(codes[0]) if codes[0] else 0
                        frames = rows if rows >= cols else cols
                    else:
                        frames = len(codes)
                if frames > 0:
                    durations.append(frames / 12.5)
    except Exception as exc:
        result["warnings"].append(f"Could not fully analyze prepared JSONL: {exc}")

    if durations:
        result["total_seconds"] = float(sum(durations))
        result["avg_seconds"] = float(sum(durations) / len(durations))
        result["max_seconds"] = float(max(durations))
        result["source"] = "prepared audio token lengths"
    else:
        result["warnings"].append("No audio-code lengths found; assuming 5.0s average clips.")
    return result


def _round_up_to_multiple(value: float, multiple: int, minimum: int = 1) -> int:
    value = max(float(value), float(minimum))
    multiple = max(1, int(multiple))
    return int(math.ceil(value / multiple) * multiple)


def _nearest_power_of_two_at_least(value: float, minimum: int = 1, maximum: int = 64) -> int:
    value = max(float(value), float(minimum))
    p = 1
    while p < value and p < maximum:
        p *= 2
    return max(minimum, min(maximum, p))


def _choose_moss_autotune_batch(profile: str, vram_preset: str, avg_duration: float, sample_count: int):
    """Higgs-style memory planning with deliberately round/stable values.

    Micro batch and grad accumulation are powers of two whenever practical.
    Their product remains a clean effective batch so scheduler/update behavior
    is easy to reason about.
    """
    is_8b = "8B" in profile
    is_32 = str(vram_preset).startswith("32")

    if is_8b:
        if is_32 and avg_duration <= 4.0 and sample_count >= 64:
            micro = 2
        else:
            micro = 1
        target_effective_batch = 16 if is_32 else 8
    else:
        if is_32 and avg_duration <= 6.0 and sample_count >= 64:
            micro = 4
        elif avg_duration <= 10.0 and sample_count >= 32:
            micro = 2
        else:
            micro = 1
        target_effective_batch = 16 if is_32 else 8

    grad = max(1, target_effective_batch // micro)
    grad = _nearest_power_of_two_at_least(grad, minimum=1, maximum=64)
    return micro, grad


def autotune_training(profile, vram_preset, prepared_jsonl):
    preset = VRAM_PRESETS[vram_preset]
    budget = preset["budget_gb"]
    physical = _gpu_vram_gb()
    if physical and physical + 0.25 < budget:
        raise gr.Error(
            f"{vram_preset} preset requires at least {budget:.0f} GB physical VRAM; "
            f"detected {physical:.1f} GB."
        )

    analysis = _analyze_prepared_training_jsonl(prepared_jsonl)
    sample_count = int(analysis["sample_count"])
    avg_duration = float(analysis["avg_seconds"] or 5.0)
    total_minutes = float(analysis["total_seconds"]) / 60.0
    max_duration = float(analysis["max_seconds"] or 0.0)

    micro, grad = _choose_moss_autotune_batch(profile, vram_preset, avg_duration, sample_count)
    effective_batch = max(1, micro * grad)
    steps_per_epoch = max(1, math.ceil(max(sample_count, 1) / effective_batch))

    # Evidence anchors:
    # - official MOSS SFT: batch 1 x grad 8, 3 epochs;
    # - published community Delay LoRA: r16/alpha32/dropout .05,
    #   AdamW + cosine, weight decay .01, max-grad-norm .5, long training.
    # For small single-speaker LoRA, target useful optimizer-update counts rather
    # than copying the very short 3-epoch full-SFT recipe.
    if total_minutes <= 0:
        target_updates = 1200
    elif total_minutes < 10:
        target_updates = 2500
    elif total_minutes < 45:
        target_updates = 2000
    elif total_minutes < 120:
        target_updates = 1500
    else:
        target_updates = 1000

    # Smaller Local models generally need fewer updates than 8B Delay.
    if "1.7b" in profile.lower():
        target_updates = int(target_updates * 0.75)
    elif "4b" in profile.lower():
        target_updates = int(target_updates * 0.85)

    raw_epochs = max(3, math.ceil(target_updates / steps_per_epoch))
    # Round upward to a stable multiple of 5 epochs.
    epochs_value = _round_up_to_multiple(raw_epochs, 5, minimum=5)
    total_steps = epochs_value * steps_per_epoch

    # Conservative LoRA LR: full SFT upstream uses 1e-5 (Delay/Local 1.0) and
    # 2e-5 (Local v1.5); successful community Delay LoRA uses 2e-6.
    if "local-v1.5" in profile.lower():
        lr_value = 1e-5
    else:
        lr_value = 5e-6

    rank = 16
    if budget >= 32 and total_minutes >= 45:
        rank = 32
    alpha = rank * 2
    dropout = 0.05

    # Default save cadence approximates five evenly-spaced checkpoints when
    # the user chooses step-based saving; epoch-based saving remains preferred.
    raw_save_steps = max(1, math.ceil(total_steps / 5))
    save_steps_value = _round_up_to_multiple(raw_save_steps, 25, minimum=25)

    warnings_list = list(analysis["warnings"])
    if sample_count and sample_count < 20:
        warnings_list.append("Very small dataset; overfitting risk is high.")
    if avg_duration > 15:
        warnings_list.append("Long average clips detected; splitting clips can reduce VRAM pressure.")
    if max_duration > 30:
        warnings_list.append("Some prepared clips exceed about 30 seconds.")

    report = (
        f"AutoTune: {vram_preset} · {profile}\n"
        f"Dataset: {sample_count} samples · {total_minutes:.1f} min · "
        f"avg {avg_duration:.1f}s · max {max_duration:.1f}s\n"
        f"Micro batch {micro} × grad accum {grad} = effective batch {effective_batch}\n"
        f"≈{steps_per_epoch} optimizer steps/epoch · {epochs_value} epochs · "
        f"≈{total_steps} total optimizer steps\n"
        f"Target updates ≈{target_updates} · LR {lr_value:g} · "
        f"LoRA r/alpha {rank}/{alpha} · dropout {dropout:.2f}"
    )
    if physical:
        report += f"\nPhysical GPU VRAM: {physical:.1f} GB"
    if warnings_list:
        report += "\nWarnings: " + " | ".join(warnings_list)

    return (
        epochs_value,
        lr_value,
        micro,
        grad,
        rank,
        alpha,
        dropout,
        save_steps_value,
        report,
    )



def _training_reader(proc, total_epochs):
    global TRAIN_PROCESS
    step_re=re.compile(r"epoch=(\d+)/(\d+) step=(\d+)/(\d+) loss=([0-9.eE+-]+)")
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line=raw.rstrip(); log(line)
            m=step_re.search(line)
            if m:
                step,total=int(m.group(3)),max(1,int(m.group(4)))
                with TRAIN_LOCK:
                    TRAIN_STATE.update(pct=100.0*step/total,text=f"Epoch {m.group(1)}/{m.group(2)} · step {step}/{total} · loss {m.group(5)}",running=True)
        rc=proc.wait()
        with TRAIN_LOCK:
            TRAIN_STATE.update(pct=100.0 if rc==0 else TRAIN_STATE.get("pct",0), text="Training completed." if rc==0 else f"Training exited with code {rc}.", running=False)
        if rc == 0:
            _release_runtime_memory(reset_compiler=True)
            play_completion_chime()
    except Exception as e:
        log(f"Training monitor error: {e}")
        with TRAIN_LOCK: TRAIN_STATE.update(text=str(e),running=False)


def _normalize_repo_identity(value: str) -> str:
    return str(value or "").replace("\\", "/").strip().lower()


def _adapter_config_matches_repo(adapter_dir: Path, repo: str) -> bool:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    base = _normalize_repo_identity(cfg.get("base_model_name_or_path", ""))
    expected = _normalize_repo_identity(repo)
    if base == expected:
        return True
    try:
        org, name = repo.split("/", 1)
        snapshot_marker = f"models--{org.lower()}--{name.lower()}"
        if snapshot_marker in base:
            return True
    except ValueError:
        pass
    return False


def _project_profile_repo(project_dir: Path) -> tuple[str, str]:
    cfg_path = project_dir / "project.json"
    if not cfg_path.is_file():
        return "", ""
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        training = data.get("training", {}) or {}
        profile_label = training.get("profile", "")
        output_name = training.get("output_name", project_dir.name)
        profile = LORA_PROFILES.get(profile_label)
        if profile:
            return profile[1], str(output_name or project_dir.name).strip().replace(" ", "_")
    except Exception:
        pass
    return "", ""


def lora_adapter_choices_for_repo(repo: str):
    """List final adapters and checkpoints compatible with the selected base model."""
    choices = [("None", "")]
    if not repo:
        return choices

    seen: set[str] = set()
    candidates: list[tuple[str, Path, int, int]] = []

    # Prefer project metadata because locally cached base_model paths in PEFT
    # adapter_config.json are not always human-readable repo IDs.
    if PROJECTS_DIR.is_dir():
        for project_dir in sorted(PROJECTS_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not project_dir.is_dir():
                continue
            project_repo, output_name = _project_profile_repo(project_dir)
            if project_repo != repo or not output_name:
                continue
            run_dir = TRAINING_OUTPUTS_DIR / output_name
            if (run_dir / "adapter_config.json").is_file():
                candidates.append((f"{project_dir.name} — Final", run_dir, 0, 10**12))
            for cp in run_dir.glob("checkpoint-*"):
                if not (cp / "adapter_config.json").is_file():
                    continue
                try:
                    step = int(cp.name.rsplit("-", 1)[1])
                except Exception:
                    step = -1
                candidates.append((f"{project_dir.name} — {cp.name}", cp, 1, step))

    # Also discover adapters copied manually into training/outputs.
    if TRAINING_OUTPUTS_DIR.is_dir():
        for cfg in TRAINING_OUTPUTS_DIR.rglob("adapter_config.json"):
            adapter_dir = cfg.parent
            key = str(adapter_dir.resolve()).lower()
            if key in seen or not _adapter_config_matches_repo(adapter_dir, repo):
                continue
            rel = adapter_dir.relative_to(TRAINING_OUTPUTS_DIR).as_posix()
            cp_match = re.search(r"/checkpoint-(\d+)$", rel)
            rank = 1 if cp_match else 0
            step = int(cp_match.group(1)) if cp_match else 10**12
            candidates.append((rel, adapter_dir, rank, step))

    # Final project adapter first, then newest checkpoints.
    candidates.sort(key=lambda x: (x[0].split(" — ")[0].lower(), x[2], -x[3], x[0].lower()))
    for label, path, _rank, _step in candidates:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        choices.append((label, str(path)))
    return choices


def lora_adapter_choices_for_model(model_label: str):
    if model_label in CPP_MODEL_VARIANTS:
        return _converted_cpp_lora_choices()
    return lora_adapter_choices_for_repo(TTS_MODELS.get(model_label, ""))


def refresh_lora_adapter_for_model(model_label: str):
    return gr.update(choices=lora_adapter_choices_for_model(model_label), value="")


def refresh_lora_adapter_for_backend(model_label: str, backend: str, current_value: str = ""):
    """Refresh adapter choices without dropping the selected C++ LoRA on quantization changes."""
    if str(backend).startswith("llama.cpp"):
        choices = _converted_cpp_lora_choices()
    else:
        choices = lora_adapter_choices_for_model(model_label)
    current = str(current_value or "")
    valid_values = {str(v) for _label, v in choices} if choices and isinstance(choices[0], tuple) else {str(v) for v in choices}
    value = current if current and current in valid_values else ""
    return gr.update(choices=choices, value=value, interactive=True)



def build_ui():
    voices=list_voice_names(VOICES_DIR)
    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Row(elem_classes="title-section"):
            with gr.Column(scale=7, min_width=420):
                gr.Markdown("# 🗣️ MOSS-TTS Family Easy GUI: Inference + LoRA Training")
            with gr.Column(scale=3, min_width=430):
                with gr.Row():
                    unload_all_btn=gr.Button("🧹 Unload All Models",size="sm",variant="secondary")
                    delete_outputs_btn=gr.Button("🗑️ Clear Outputs",size="sm",variant="stop")
                    delete_voices_btn=gr.Button("🗑️ Clear Samples",size="sm",variant="stop")
        top_status=gr.Markdown("",elem_classes="compact-status")

        with gr.Tabs(elem_classes="tabs"):
            with gr.Tab("🎙️ Prep Samples"):
                gr.Markdown("*Create and manage reusable reference voices for cloning, continuation and dialogue workflows.*", elem_classes=["tab-subtitle"])
                add_help_accordion("""Save reusable reference voices here. Use a clean clip with one speaker and little background noise. A transcript is optional for normal voice cloning, but it is needed for Continuation and some dialogue workflows. **Transcribe Now** can fill the transcript with Faster-Whisper when you want it; nothing is transcribed automatically. Selecting **None** clears the saved voice so you can work with a new sample.""")
                with gr.Row():
                    with gr.Column(scale=1,elem_classes="form-section"):
                        gr.Markdown("#### 📚 Voice Library")
                        with gr.Row():
                            with gr.Column(scale=8):
                                voice_saved=gr.Dropdown(voices,value="None",label="Saved Voice")
                            with gr.Column(scale=1,min_width=48):
                                voice_refresh=gr.Button("🔄",size="sm")
                        voice_delete=gr.Button("🗑️ Delete",size="sm",variant="stop")
                        voice_preview=gr.Audio(type="filepath",label="Preview",elem_classes=["audio-safe-space"])
                        voice_saved_text=gr.Textbox(label="Transcript",lines=4,interactive=False)
                    with gr.Column(scale=2,elem_classes="form-section"):
                        gr.Markdown("#### 🎙️ Prepare Sample")
                        voice_audio=gr.Audio(label="Reference Audio",type="filepath",interactive=True,sources=["upload","microphone"],elem_classes=["audio-safe-space"])
                        with gr.Accordion("🛰️ Faster-Whisper Transcription (Optional)",open=False):
                            with gr.Row():
                                voice_asr_model=gr.Dropdown(ASR_CHOICES,value=[x for x in ASR_CHOICES if "large-v3" in x and "distil" not in x][0],label="ASR Model")
                                voice_asr_language=gr.Dropdown(ASR_LANGUAGES,value="Auto-detect",label="Language")
                                voice_asr_batch=gr.Slider(1,32,value=8,step=1,label="Batch Size")
                            voice_transcribe=gr.Button("Transcribe Now")
                        voice_name=gr.Textbox(label="Voice Name")
                        voice_transcript=gr.Textbox(label="Reference Transcript (optional)",lines=4)
                        voice_save=gr.Button("💾 Save Voice",variant="primary",elem_classes="green-btn")
                        voice_status=gr.Markdown()
            with gr.Tab("🔊 Inference"):
                gr.Markdown("*Generate speech, dialogue, voices, sound effects and realtime audio with the MOSS-TTS family.*", elem_classes=["tab-subtitle"])
                add_help_accordion("""Choose the type of audio you want to create, then open the matching subtab. **FlashAttention 2** can speed up compatible PyTorch models. The llama.cpp backend is configured inside the TTS workflows and is intended for the fastest MOSS-TTS v1.5 inference.""")
                gr.Markdown("### 🚀 Acceleration Engines")
                with gr.Row():
                    use_flash_attention=gr.Checkbox(True,label="FlashAttention 2")
                # Kept as non-visual state only to preserve internal callback signatures.
                # Generic torch.compile is intentionally not applied to MOSS-TTS/Local/TTSD/VoiceGenerator.
                use_torch_compile=gr.State(False)
                compile_mode=gr.State("default")
                with gr.Tabs():
                    with gr.Tab("TTS / Voice Clone"):
                        gr.Markdown("*Single-speaker synthesis and voice cloning with Transformers or the accelerated llama.cpp backend.*", elem_classes=["tab-subtitle"])
                        add_help_accordion("""Choose a model, write the text and optionally select a saved reference voice. **Direct / Voice Clone** only needs the reference audio. **Continuation + Voice Clone** also needs the exact transcript of the reference.

### llama.cpp CUDA
This is the fastest backend for **MOSS-TTS v1.5 Delay 8B**.

On the first C++ use, the GUI downloads the official v1.5 model once and prepares a reusable F16 GGUF. If the quantization you selected does not exist yet, the GUI creates it from that local F16 file.

- Switching from **Q8_0 to Q4_K_M, Q5_K_M, Q6_K or another quantization does not download the model again**.
- The first time you select a new quantization, expect a one-time conversion that can take a few minutes and use extra disk space.
- Once created, each quantization is cached and reused on later launches.
- Changing **KV Cache**, GPU Layers or other memory controls does not rebuild the model.
- A VRAM preset only triggers a new conversion if it selects a quantization you have never prepared before.
- LoRA voices are converted separately in **GGUF Conversion** and the same converted LoRA can be used with every compatible v1.5 quantization.

If you want the fastest startup, keep using a quantization that is already prepared.

### Long Text / Chunking
For long text, open **Long Text / Chunking**. **Paragraph/Sentence Auto** is recommended for normal prose. You can also split strictly by periods, paragraphs, lines or `[SPEAKER1]`-style speaker turns.

Each chunk is generated separately with the same selected voice and settings and then joined into one WAV. **Silence Between Chunks** controls the pause between clips; the default is **0.5 seconds**. Select **None** when you do not want the text split.""")
                        with gr.Row():
                            with gr.Column(scale=1,elem_classes="form-section"):
                                with gr.Row():
                                    with gr.Column(scale=8):
                                        tts_voice=gr.Dropdown(voices,value="None",label="Saved Voice")
                                    with gr.Column(scale=1,min_width=48):
                                        tts_voice_refresh=gr.Button("🔄",size="sm")
                                tts_ref=gr.Audio(type="filepath",label="Reference Audio (optional)",sources=["upload","microphone"],elem_classes=["audio-safe-space"])
                                tts_ref_text=gr.Textbox(label="Reference Transcript",lines=4)
                            with gr.Column(scale=2,elem_classes="form-section"):
                                tts_backend=gr.Dropdown(TTS_BACKENDS,value=TTS_BACKENDS[0],label="Inference Backend")
                                tts_model=gr.Dropdown(list(TTS_MODELS),value=list(TTS_MODELS)[0],label="Model / Quantization")
                                with gr.Group(visible=False, elem_classes=["cpp-backend-controls"]) as tts_cpp_controls:
                                    gr.Markdown("**llama.cpp CUDA memory controls**")
                                    cpp_vram_preset=gr.Dropdown(CPP_VRAM_PRESET_CHOICES,value="24 GB",label="VRAM Preset")
                                    cpp_preset_note=gr.Markdown(CPP_VRAM_PRESETS["24 GB"]["description"])
                                    with gr.Row():
                                        cpp_kv=gr.Dropdown(["f16","q8_0","q4_0"],value="q8_0",label="KV Cache Type")
                                        cpp_gpu_layers=gr.Slider(-1,80,value=-1,step=1,label="GPU Layers (-1 = All)")
                                    with gr.Row():
                                        cpp_low_memory=gr.Checkbox(False,label="Low Memory / Staged Loading")
                                        cpp_heads_backend=gr.Dropdown(["GPU / Torch","CPU / NumPy"],value="GPU / Torch",label="LM Heads")
                                        cpp_audio_gpu=gr.Checkbox(True,label="Audio Tokenizer on GPU")
                                with gr.Row():
                                    with gr.Column(scale=2):
                                        tts_mode=gr.Dropdown(["Direct / Voice Clone","Continuation + Voice Clone"],value="Direct / Voice Clone",label="Mode")
                                    with gr.Column(scale=2):
                                        tts_language=gr.Dropdown(LANGUAGES,value="Auto",label="Language")
                                    with gr.Column(scale=3):
                                        tts_adapter=gr.Dropdown(
                                            choices=lora_adapter_choices_for_model(list(TTS_MODELS)[0]),
                                            value="",
                                            label="LoRA Adapter / Checkpoint",
                                            info="Filtered automatically for the selected base model."
                                        )
                                    with gr.Column(scale=1,min_width=48):
                                        tts_adapter_refresh=gr.Button("🔄",size="sm")
                                tts_text=gr.Textbox(label="Text",lines=8)
                                with gr.Accordion("⚙️ Generation Parameters",open=False):
                                    with gr.Row():
                                        tts_temp=gr.Slider(.1,3,value=1.7,step=.05,label="Temperature")
                                        tts_top_p=gr.Slider(.1,1,value=.8,step=.01,label="Top P")
                                    with gr.Row():
                                        tts_top_k=gr.Slider(1,200,value=25,step=1,label="Top K")
                                        tts_rep=gr.Slider(.8,2,value=1,step=.05,label="Repetition Penalty")
                                    with gr.Row():
                                        tts_expected_tokens=gr.Number(value=0,precision=0,label="Expected Audio Tokens (0 = Auto)",info="Duration control: about 12.5 tokens ≈ 1 second")
                                        tts_tokens=gr.Slider(256,8192,value=2048,step=128,label="Max New Tokens")
                                    tts_seed=gr.Number(value=-1,precision=0,label="Seed (-1 = Random; Local v1.5 only)")
                                with gr.Accordion("📚 Long Text / Chunking",open=False):
                                    with gr.Row():
                                        tts_chunk_mode=gr.Dropdown(
                                            CHUNK_CHOICES,
                                            value="None",
                                            label="Chunk Mode",
                                            info="None disables splitting. Paragraph/Sentence Auto is recommended for long prose."
                                        )
                                        tts_chunk_silence=gr.Slider(
                                            0,5,value=.5,step=.1,
                                            label="Silence Between Chunks (seconds)"
                                        )
                                    gr.Markdown(
                                        "Each chunk is synthesized separately with the same voice and settings, "
                                        "then all chunks are joined into one WAV."
                                    )
                                tts_go=gr.Button("Generate Speech",variant="primary")
                                tts_audio=gr.Audio(label="Output",elem_classes=["output-clean","audio-safe-space"])
                                tts_path=gr.Textbox(label="Output Path",interactive=False,elem_classes=["output-path"])
                    with gr.Tab("Dialogue Builder"):
                        gr.Markdown("*Build multi-speaker dialogue with the dedicated MOSS-TTSD model and independent visual turns.*", elem_classes=["tab-subtitle"])
                        gr.Markdown("MOSS-TTSD dialogue with independent turns. Up to **5 distinct saved voices** can be reused across the turns.")
                        dialogue_components=[]; dialogue_rows=[]; d_voice_refresh_btns=[]; d_add_btns=[]; d_clone_btns=[]; d_delete_btns=[]; d_clear_btns=[]; d_up_btns=[]; d_down_btns=[]
                        d_row_count=gr.State(2)
                        with gr.Row(elem_classes="dialogue-toolbar"):
                            d_reset=gr.Button("Reset rows",size="sm",variant="secondary")
                            d_clear_all=gr.Button("Clear rows",size="sm",variant="secondary")
                            d_compact=gr.Button("Remove empty rows",size="sm",variant="secondary")
                        for i in range(DIALOGUE_MAX_TURNS):
                            with gr.Group(visible=i<2,elem_classes=["dialogue-turn-card"]) as row_group:
                                with gr.Row():
                                    with gr.Column(scale=2,min_width=180):
                                        with gr.Row():
                                            d_sp=gr.Dropdown(voices,value="None",label=f"Turn {i+1} · Voice")
                                            d_voice_refresh=gr.Button("🔄",size="sm")
                                    with gr.Column(scale=6):
                                        d_tx=gr.Textbox(label=f"Text {i+1}",placeholder=f"Enter dialogue text for turn {i+1}...",lines=2)
                                    with gr.Column(scale=2,min_width=210,elem_classes=["dialogue-actions"]):
                                        with gr.Row():
                                            b_add=gr.Button("➕",size="sm"); b_clone=gr.Button("📋",size="sm"); b_up=gr.Button("⬆️",size="sm"); b_down=gr.Button("⬇️",size="sm")
                                        with gr.Row():
                                            b_clear=gr.Button("🧹",size="sm"); b_delete=gr.Button("🗑️",size="sm",variant="stop")
                            dialogue_components.extend([d_sp,d_tx]); dialogue_rows.append(row_group); d_voice_refresh_btns.append(d_voice_refresh); d_add_btns.append(b_add); d_clone_btns.append(b_clone); d_up_btns.append(b_up); d_down_btns.append(b_down); d_clear_btns.append(b_clear); d_delete_btns.append(b_delete)
                        with gr.Accordion("⚙️ Generation Parameters",open=False):
                            with gr.Row():
                                d_temp=gr.Slider(.1,3,value=1.1,step=.05,label="Temperature"); d_top_p=gr.Slider(.1,1,value=.9,step=.01,label="Top P")
                            with gr.Row():
                                d_top_k=gr.Slider(1,200,value=50,step=1,label="Top K"); d_rep=gr.Slider(.8,2,value=1.1,step=.05,label="Repetition Penalty")
                            d_tokens=gr.Slider(256,8192,value=2048,step=128,label="Max New Tokens")
                        d_go=gr.Button("⚡ Generate Dialogue",variant="primary",size="lg"); d_audio=gr.Audio(label="Generated Dialogue",elem_classes=["output-clean","audio-safe-space"]); d_path=gr.Textbox(label="Saved WAV",interactive=False); d_status=gr.Textbox(label="Dialogue Status",interactive=False,lines=2)
                        add_help_accordion("Each row is an independent dialogue turn. Select a saved voice and write only that speaker's text for the row; row controls can insert, clone, reorder, clear or delete turns. The dedicated TTSD workflow supports **1–5 distinct speakers** per dialogue, while the same saved voice can be reused across any number of turns. Every distinct voice used by TTSD needs both reference audio and its saved transcript because the model builds speaker prefixes from those references before generating the requested dialogue.")

                    with gr.Tab("Dialogue Builder (Classic)"):
                        gr.Markdown("*Generate each dialogue turn independently with standard TTS models, then join the results.*", elem_classes=["tab-subtitle"])
                        gr.Markdown("Turn-by-turn dialogue generated with the standard TTS models. Each row is synthesized independently and then joined.")
                        classic_components=[]; classic_rows=[]; c_voice_refresh_btns=[]; c_add_btns=[]; c_clone_btns=[]; c_delete_btns=[]; c_clear_btns=[]; c_up_btns=[]; c_down_btns=[]
                        with gr.Column(elem_classes="form-section"):
                            c_backend=gr.Dropdown(TTS_BACKENDS,value=TTS_BACKENDS[0],label="Inference Backend")
                            c_model=gr.Dropdown(list(TTS_MODELS),value=list(TTS_MODELS)[0],label="Model / Quantization")
                            with gr.Group(visible=False, elem_classes=["cpp-backend-controls"]) as c_cpp_controls:
                                gr.Markdown("**llama.cpp CUDA memory controls**")
                                c_cpp_vram_preset=gr.Dropdown(CPP_VRAM_PRESET_CHOICES,value="24 GB",label="VRAM Preset")
                                c_cpp_preset_note=gr.Markdown(CPP_VRAM_PRESETS["24 GB"]["description"])
                                with gr.Row():
                                    c_cpp_kv=gr.Dropdown(["f16","q8_0","q4_0"],value="q8_0",label="KV Cache Type")
                                    c_cpp_gpu_layers=gr.Slider(-1,80,value=-1,step=1,label="GPU Layers (-1 = All)")
                                with gr.Row():
                                    c_cpp_low_memory=gr.Checkbox(False,label="Low Memory / Staged Loading")
                                    c_cpp_heads_backend=gr.Dropdown(["GPU / Torch","CPU / NumPy"],value="GPU / Torch",label="LM Heads")
                                    c_cpp_audio_gpu=gr.Checkbox(True,label="Audio Tokenizer on GPU")
                            with gr.Row():
                                with gr.Column(scale=2):
                                    c_language=gr.Dropdown(LANGUAGES,value="Auto",label="Language")
                                with gr.Column(scale=2):
                                    c_pause=gr.Slider(0,5,value=.5,step=.05,label="Pause Between Turns (seconds)")
                                with gr.Column(scale=3):
                                    c_adapter=gr.Dropdown(
                                        choices=lora_adapter_choices_for_model(list(TTS_MODELS)[0]),
                                        value="",
                                        label="LoRA Adapter / Checkpoint",
                                        info="Filtered automatically for the selected base model."
                                    )
                                with gr.Column(scale=1,min_width=48):
                                    c_adapter_refresh=gr.Button("🔄",size="sm")
                        c_row_count=gr.State(2)
                        with gr.Row(elem_classes="dialogue-toolbar"):
                            c_reset=gr.Button("Reset rows",size="sm",variant="secondary"); c_clear_all=gr.Button("Clear rows",size="sm",variant="secondary"); c_compact=gr.Button("Remove empty rows",size="sm",variant="secondary")
                        for i in range(DIALOGUE_MAX_TURNS):
                            with gr.Group(visible=i<2,elem_classes=["dialogue-turn-card"]) as row_group:
                                with gr.Row():
                                    with gr.Column(scale=2,min_width=180):
                                        with gr.Row():
                                            c_sp=gr.Dropdown(voices,value="None",label=f"Turn {i+1} · Voice")
                                            c_voice_refresh=gr.Button("🔄",size="sm")
                                    with gr.Column(scale=6): c_tx=gr.Textbox(label=f"Text {i+1}",placeholder=f"Enter dialogue text for turn {i+1}...",lines=2)
                                    with gr.Column(scale=2,min_width=210,elem_classes=["dialogue-actions"]):
                                        with gr.Row():
                                            b_add=gr.Button("➕",size="sm"); b_clone=gr.Button("📋",size="sm"); b_up=gr.Button("⬆️",size="sm"); b_down=gr.Button("⬇️",size="sm")
                                        with gr.Row(): b_clear=gr.Button("🧹",size="sm"); b_delete=gr.Button("🗑️",size="sm",variant="stop")
                            classic_components.extend([c_sp,c_tx]); classic_rows.append(row_group); c_voice_refresh_btns.append(c_voice_refresh); c_add_btns.append(b_add); c_clone_btns.append(b_clone); c_up_btns.append(b_up); c_down_btns.append(b_down); c_clear_btns.append(b_clear); c_delete_btns.append(b_delete)
                        with gr.Accordion("⚙️ Generation Parameters",open=False):
                            with gr.Row(): c_temp=gr.Slider(.1,3,value=1.7,step=.05,label="Temperature"); c_top_p=gr.Slider(.1,1,value=.8,step=.01,label="Top P")
                            with gr.Row(): c_top_k=gr.Slider(1,200,value=25,step=1,label="Top K"); c_rep=gr.Slider(.8,2,value=1,step=.05,label="Repetition Penalty")
                            c_tokens=gr.Slider(256,8192,value=2048,step=128,label="Max New Tokens / Turn")

                        c_go=gr.Button("⚡ Generate Dialogue",variant="primary",size="lg"); c_audio=gr.Audio(label="Generated Dialogue",elem_classes=["output-clean","audio-safe-space"]); c_path=gr.Textbox(label="Saved WAV",interactive=False); c_status=gr.Textbox(label="Dialogue Status",interactive=False,lines=2)
                        add_help_accordion("Each visible row is synthesized independently with the selected standard TTS model, then the WAVs are concatenated in row order. This mode is useful when you want more than five speakers or prefer the normal voice-cloning path instead of TTSD. **Language** and **Pause Between Turns (seconds)** are global for the whole dialogue; each row chooses its own saved voice. The saved transcript is not required for ordinary per-turn voice cloning. Sampling controls apply identically to every generated turn. With **llama.cpp CUDA**, Model / Quantization exposes the official F16, Q8_0, Q6_K, Q5_K_M and Q4_K_M backbones; LoRA Adapter / Checkpoint supports MOSS-TTS v1.5 Delay LoRAs: the selected PEFT adapter is converted once to a cached GGUF LoRA and then applied dynamically by llama.cpp without merging it into the quantized backbone.")

                    with gr.Tab("Voice Generator"):
                        gr.Markdown("*Create speech from a natural-language description of the desired voice, style and delivery.*", elem_classes=["tab-subtitle"])
                        add_help_accordion("""Describe the kind of voice you want in plain language, then enter the text to speak. Use this when you want to create a voice from a description instead of cloning an existing reference clip. Start with the default generation settings and adjust them only if needed.""")
                        with gr.Row():
                            with gr.Column(scale=1,elem_classes="form-section"):
                                vg_instruction=gr.Textbox(label="Voice / Style Instruction",lines=10)
                            with gr.Column(scale=2,elem_classes="form-section"):
                                gr.Markdown("**Model:** MOSS-VoiceGenerator · 1.7B · ~8–10 GB VRAM")
                                vg_language=gr.Dropdown(VOICE_GENERATOR_LANGUAGES,value="Auto",label="Language")
                                vg_text=gr.Textbox(label="Text",lines=8)
                                with gr.Accordion("⚙️ Generation Parameters",open=False):
                                    with gr.Row():
                                        vg_temp=gr.Slider(.1,3,value=1.5,step=.05,label="Temperature")
                                        vg_top_p=gr.Slider(.1,1,value=.6,step=.01,label="Top P")
                                    with gr.Row():
                                        vg_top_k=gr.Slider(1,200,value=50,step=1,label="Top K")
                                        vg_rep=gr.Slider(.8,2,value=1.1,step=.05,label="Repetition Penalty")
                                    vg_tokens=gr.Slider(256,8192,value=2048,step=128,label="Max New Tokens")
                                vg_go=gr.Button("Generate Voice",variant="primary")
                                vg_audio=gr.Audio(label="Output",elem_classes=["output-clean","audio-safe-space"]); vg_path=gr.Textbox(label="Output Path",interactive=False)
                    with gr.Tab("Sound Effect"):
                        gr.Markdown("*Generate non-speech audio and sound effects from text descriptions.*", elem_classes=["tab-subtitle"])
                        add_help_accordion("""Describe a sound and generate it directly from text. Use this for ambience, impacts, environmental sounds and other non-speech audio. Pick the model, choose the target duration and start from the default settings.""")
                        with gr.Row():
                            with gr.Column(scale=2,elem_classes="form-section"):
                                sfx_model=gr.Dropdown(
                                    ["MOSS-SoundEffect v2.0  •  DiT 1.3B + Qwen3 1.7B  •  ~10–14 GB VRAM", "MOSS-SoundEffect  •  8B  •  ~21–24 GB VRAM"],
                                    value="MOSS-SoundEffect v2.0  •  DiT 1.3B + Qwen3 1.7B  •  ~10–14 GB VRAM",
                                    label="Model",
                                )
                                sfx_prompt=gr.Textbox(label="Sound Description",lines=10)
                            with gr.Column(scale=1,elem_classes="form-section"):
                                sfx_duration=gr.Slider(.5,30,value=10,step=.5,label="Duration (seconds)")
                                sfx_v2_compile=gr.Checkbox(False,label="torch.compile (SoundEffect v2.0)",info="Upstream DiT compile path. Disabled by default, matching the bundled official inference script fallback on systems with Dynamo/Triton issues.")
                                with gr.Accordion("⚙️ Sound Generation Parameters",open=False):
                                    gr.Markdown("**v2.0 diffusion controls**")
                                    sfx_steps=gr.Slider(10,150,value=100,step=1,label="Inference Steps")
                                    with gr.Row():
                                        sfx_cfg=gr.Slider(1,8,value=4,step=.1,label="CFG Scale")
                                        sfx_sigma=gr.Slider(0,10,value=5,step=.1,label="Sigma Shift")
                                    sfx_seed=gr.Number(value=0,precision=0,label="Seed")
                                    gr.Markdown("**Legacy autoregressive controls**")
                                    with gr.Row():
                                        sfx_temp=gr.Slider(.1,3,value=1.5,step=.05,label="Temperature")
                                        sfx_top_p=gr.Slider(.1,1,value=.6,step=.01,label="Top P")
                                    with gr.Row():
                                        sfx_top_k=gr.Slider(1,200,value=50,step=1,label="Top K")
                                        sfx_rep=gr.Slider(.8,2,value=1.2,step=.05,label="Repetition Penalty")
                                    sfx_max_tokens=gr.Slider(256,8192,value=2048,step=128,label="Max New Tokens")
                                sfx_go=gr.Button("Generate Sound",variant="primary")
                                sfx_audio=gr.Audio(label="Output",elem_classes=["output-clean","audio-safe-space"]); sfx_path=gr.Textbox(label="Output Path",interactive=False)
                    with gr.Tab("Realtime"):
                        gr.Markdown("*Use the low-latency MOSS-TTS Realtime model with saved voices and multilingual controls.*", elem_classes=["tab-subtitle"])
                        add_help_accordion("""Generate speech with the low-latency Realtime model. Select a saved voice or reference clip, enter the text and choose the language when known. The default decoding settings are the recommended starting point for stable realtime speech.""")
                        with gr.Row():
                            with gr.Column(scale=1,elem_classes="form-section"):
                                with gr.Row():
                                    rt_voice=gr.Dropdown(voices,value="None",label="Saved Voice")
                                    rt_voice_refresh=gr.Button("🔄",size="sm")
                                rt_ref=gr.Audio(type="filepath",label="Reference Audio (optional)",elem_classes=["audio-safe-space"])
                            with gr.Column(scale=2,elem_classes="form-section"):
                                gr.Markdown("**Model:** MOSS-TTS Realtime · 1.7B · ~8–10 GB VRAM")
                                rt_language=gr.Dropdown(REALTIME_LANGUAGES,value="Auto",label="Language")
                                rt_text=gr.Textbox(label="Text",lines=8)
                                with gr.Accordion("⚙️ Decoding & Sampling Parameters",open=False):
                                    with gr.Row():
                                        rt_temp=gr.Slider(.1,1.5,value=.8,step=.05,label="Temperature"); rt_top_p=gr.Slider(.1,1,value=.6,step=.05,label="Top P")
                                    with gr.Row():
                                        rt_top_k=gr.Slider(1,100,value=30,step=1,label="Top K"); rt_rep=gr.Slider(1,1.5,value=1.1,step=.01,label="Repetition Penalty")
                                    rt_rep_window=gr.Slider(1,512,value=50,step=1,label="Repetition Window")
                                rt_go=gr.Button("Generate",variant="primary")
                                rt_audio=gr.Audio(label="Output",elem_classes=["output-clean","audio-safe-space"]); rt_path=gr.Textbox(label="Output Path",interactive=False)
            with gr.Tab("📂 Dataset Preparation"):
                gr.Markdown("*Organize project audio/transcripts and encode them into training-ready MOSS datasets.*", elem_classes=["tab-subtitle"])
                add_help_accordion("""Create or select a project, choose the folder that contains your audio and matching `.txt` transcripts, then scan it. Review the clips and text before pressing **Prepare Dataset**. The prepared dataset and project settings are restored automatically when you return later.""")
                with gr.Row(elem_classes="project-strip"):
                    project=gr.Dropdown(list_projects(),value="None",label="Project")
                    new_project=gr.Textbox(label="New Project Name")
                    create_project_btn=gr.Button("Create Project")
                    save_project_btn=gr.Button("Save Project")
                    delete_project_btn=gr.Button("🗑️ Delete Project",variant="stop")
                project_status=gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=1,elem_classes="form-section"):
                        with gr.Row():
                            dataset_folder=gr.Textbox(label="Source Audio Folder",placeholder=r"D:\dataset\speaker")
                            dataset_browse_btn=gr.Button("📁 Browse",size="sm")
                        dataset_language=gr.Dropdown(LANGUAGES,value="Auto",label="Language")
                        train_ref=gr.Audio(type="filepath",label="Shared Reference Audio (optional)",elem_classes=["audio-safe-space"])
                    with gr.Column(scale=1,elem_classes="form-section"):
                        transcribe_missing=gr.Checkbox(value=True,label="Transcribe missing TXT files")
                        dataset_asr_model=gr.Dropdown(ASR_CHOICES,value=[x for x in ASR_CHOICES if "large-v3" in x and "distil" not in x][0],label="Faster-Whisper Model")
                        dataset_asr_language=gr.Dropdown(ASR_LANGUAGES,value="Auto-detect",label="Language")
                        dataset_asr_batch=gr.Slider(1,32,value=8,step=1,label="ASR Batch Size")
                scan_btn=gr.Button("Build / Scan Dataset",variant="primary")
                dataset_status=gr.Markdown()
                dataset_table=gr.Dataframe(headers=["audio","text","language","ref_audio"],datatype=["str"]*4,type="array",interactive=True,label="Dataset")
                with gr.Row():
                    dataset_profile=gr.Dropdown(list(LORA_PROFILES),value=list(LORA_PROFILES)[0],label="Training Model")
                    prepare_btn=gr.Button("Prepare Dataset")
                prepared_path=gr.Textbox(label="Prepared JSONL",interactive=False)
            with gr.Tab("🚀 LoRA Training"):
                gr.Markdown("*Fine-tune supported MOSS-TTS checkpoints with project persistence, AutoTune, checkpoints and TensorBoard.*", elem_classes=["tab-subtitle"])
                add_help_accordion("""Select a prepared project and a trainable MOSS-TTS base model, then use **AutoTune for VRAM** as the starting point. AutoTune adjusts batch size, accumulation, epochs and LoRA settings from the dataset and selected VRAM budget.

Training can save checkpoints by steps or by epochs. Optional preview audio lets you hear how the voice is evolving in TensorBoard. Use **Resume Checkpoint** to continue a previous run from its saved training state; select `None` to start a new run.

The training sliders intentionally have wide ranges. Very small datasets can require hundreds of epochs to reach the optimizer-update target calculated by AutoTune, so the GUI will not clamp those values to a short preset-oriented range.""")
                with gr.Row(elem_classes="project-strip"):
                    train_project=gr.Dropdown(list_projects(),value="None",label="Project")
                    train_preset=gr.Dropdown(list(VRAM_PRESETS),value="24 GB",label="VRAM Preset")
                    autotune_btn=gr.Button("⚡ AutoTune for VRAM")
                    save_train_project_btn=gr.Button("💾 Save Project",variant="secondary")
                    delete_train_project_btn=gr.Button("🗑️ Delete Project",variant="stop")
                autotune_status=gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=1,elem_classes="form-section"):
                        train_profile=gr.Dropdown(list(LORA_PROFILES),value=list(LORA_PROFILES)[0],label="Base Model")
                        train_prepared_path=gr.Textbox(label="Prepared JSONL")
                        output_name=gr.Textbox(label="Adapter Name",value="my_moss_lora")
                        with gr.Row():
                            resume_checkpoint=gr.Dropdown(
                                choices=["None"],
                                value="None",
                                label="Resume Checkpoint",
                                info="None starts from the base model. Select a checkpoint to restore adapter + optimizer + scheduler + training position."
                            )
                            resume_refresh_btn=gr.Button("🔄",size="sm")
                    with gr.Column(scale=1,elem_classes="form-section"):
                        with gr.Row():
                            epochs=gr.Slider(1,2000,value=30,step=1,label="Epochs",info="LoRA usually needs many more optimizer updates than the official 3-epoch full-SFT baseline; use AutoTune after preparing the dataset.")
                            lr=gr.Slider(1e-8,5e-4,value=5e-6,step=1e-7,label="Learning Rate",info="Official full-SFT baseline is 1e-5 for Delay/Local 1.0 and 2e-5 for Local v1.5; the published community Delay LoRA used 2e-6. AutoTune uses a conservative LoRA-specific value.")
                        with gr.Row():
                            micro_batch=gr.Slider(1,128,value=1,step=1,label="Micro Batch Size")
                            grad_accum=gr.Slider(1,2048,value=8,step=1,label="Gradient Accumulation")
                        with gr.Row(): lora_r=gr.Slider(2,1024,value=16,step=2,label="LoRA Rank"); lora_alpha=gr.Slider(2,4096,value=32,step=2,label="LoRA Alpha")
                        with gr.Row():
                            lora_dropout=gr.Slider(0,.95,value=.05,step=.01,label="Dropout")
                            save_mode=gr.Dropdown(["Every N Steps","Every N Epochs"],value="Every N Steps",label="Checkpoint Cadence")
                        with gr.Row():
                            save_steps=gr.Slider(1,200000,value=250,step=25,label="Save Every N Steps")
                            save_every_epochs=gr.Slider(1,500,value=1,step=1,label="Save Every N Epochs")
                        steps_per_epoch_state=gr.State(1)
                        total_steps_state=gr.State(1)
                        training_schedule_info=gr.Markdown("Training schedule will be calculated from the prepared dataset.")
                        with gr.Row(): train_flash_attention=gr.Checkbox(True,label="FlashAttention 2")
                        train_torch_compile=gr.State(False)
                        with gr.Accordion("🎧 Eval Audio + TensorBoard", open=False):
                            enable_eval_audio=gr.Checkbox(False,label="Generate eval audio preview at each save")
                            eval_text=gr.Textbox(
                                value="This is a MOSS-TTS training preview.",
                                label="Eval Preview Text",
                                lines=2,
                            )
                            eval_reference_audio=gr.Audio(
                                label="Eval Reference Audio (optional)",
                                type="filepath",
                            )
                            gr.Markdown(
                                "The eval preview uses this clip as the voice reference. Leave it empty for unconditioned evaluation.",
                                elem_classes=["small-note"],
                            )
                            eval_max_tokens=gr.Slider(
                                128, 8192, value=512, step=128,
                                label="Eval Max New Tokens",
                            )
                with gr.Row(): train_btn=gr.Button("🚀 Start Training",variant="primary"); stop_btn=gr.Button("⏹️ Stop",variant="stop")
                train_progress=gr.HTML(training_progress_html())
                with gr.Row():
                    train_status=gr.Markdown()
                    tensorboard_btn=gr.Button("📊 Open TensorBoard",variant="secondary")
                tensorboard_status=gr.Markdown()
                adapter_path=gr.Textbox(label="Adapter Output",interactive=False)
            with gr.Tab("🧩 GGUF Conversion"):
                gr.Markdown("*Convert trained MOSS-TTS v1.5 Delay 8B PEFT LoRAs into reusable llama.cpp GGUF adapters.*", elem_classes=["tab-subtitle"])
                gr.Markdown("### Convert trained LoRA adapters for llama.cpp")
                with gr.Row():
                    with gr.Column(scale=3,elem_classes="form-section"):
                        with gr.Row():
                            with gr.Column(scale=8):
                                gguf_checkpoint=gr.Dropdown(
                                    choices=lora_adapter_choices_for_repo(CPP_V15_REPO),
                                    value="",
                                    label="v1.5 Delay 8B Checkpoint / Final Adapter",
                                )
                            with gr.Column(scale=1,min_width=48):
                                gguf_checkpoint_refresh=gr.Button("🔄",size="sm")
                        gguf_lora_name=gr.Textbox(
                            label="Converted LoRA Name",
                            placeholder="Example: PL_voice_v1",
                        )
                        gguf_convert_btn=gr.Button("Convert Checkpoint to GGUF LoRA",variant="primary")
                        gguf_convert_status=gr.Markdown()
                    with gr.Column(scale=2,elem_classes="form-section"):
                        gguf_registered=gr.Dropdown(
                            choices=_converted_cpp_lora_choices(),
                            value="",
                            label="Registered GGUF LoRAs",
                            interactive=False,
                        )
                        gr.Markdown(
                            "Converted adapters are stored in `models/moss-tts-cpp-v1.5/lora_adapters/` "
                            "and become available automatically when the inference backend is switched to llama.cpp."
                        )
                add_help_accordion("""Use this tab only when you want to use a trained **MOSS-TTS v1.5 Delay 8B LoRA** with the fast llama.cpp backend.

1. Select the final LoRA or one of its checkpoints.
2. Give the converted voice a clear name.
3. Press **Convert Checkpoint to GGUF LoRA**.
4. Switch inference to llama.cpp and select the converted voice from the LoRA dropdown.

You only need to convert a LoRA once. The converted LoRA is separate from the main model, so the same file can be reused with **Q4_K_M, Q5_K_M, Q6_K, Q8_0 or F16** without converting it again.""")
            with gr.Tab("🔑 HuggingFace Token"):
                gr.Markdown("*Store and verify a Hugging Face read token for model downloads that require authenticated access.*", elem_classes=["tab-subtitle"])
                add_help_accordion("""Paste a Hugging Face **Read** token here only when a model download requires account access. Public downloads can work without one. After the token is accepted, the GUI reuses it automatically for later downloads.""")
                gr.Markdown("""
## HuggingFace Token

The C++ backend now builds its backbone from the official **MOSS-TTS-v1.5** checkpoint locally and uses the official ONNX audio tokenizer. The token remains available for any Hugging Face download that requires authentication; the old preconverted `MOSS-TTS-GGUF` repository is no longer used as the v1.5 backbone.  
The normal Transformers backend does **not** require this field unless Hugging Face itself requires authentication for a specific model.

### Setup — one time per Hugging Face account

1. **Create or sign in to a Hugging Face account.**
   - Go to the Hugging Face website and sign in with the account you want to use.

2. **Open the official MOSS-TTS GGUF model page.**
   - Repository: `OpenMOSS-Team/MOSS-TTS-GGUF`
   - Read and accept/request the model access terms while signed into that account.
   - Wait until the page shows that access has been granted.

3. **Create a Read token.**
   - In Hugging Face, open **Settings → Access Tokens**.
   - Create a new token with **Read** permission.
   - A Write token is not required.

4. **Paste the token below.**
   - Tokens normally start with `hf_`.
   - The field is password-masked.
   - The Easy GUI uses it only to authenticate Hugging Face API/list/download requests.
   - The token is kept in the current GUI session; it is not written into project source files.

5. **Press `Check Token & GGUF Access`.**
   - `Token valid + model access working` means the C++ backend can enumerate/download its GGUF files.
   - If the token is valid but model access fails, make sure the **same account** accepted the GGUF repository terms.

6. **Use the C++ backend normally.**
   - Go to **Inference → TTS / Voice Clone** or **Dialogue Builder (Classic)**.
   - Select `llama.cpp CUDA + ONNX`.
   - The token entered here is reused automatically.
   - Model files remain **on-demand**: choosing a quantization does not download it until inference starts.

> **Security:** treat your Hugging Face token like a password. Do not publish it, put it in screenshots, commit it to Git, or share it with other users.
""")
                with gr.Column(elem_classes="form-section"):
                    hf_token_global=gr.Textbox(
                        label="Hugging Face Read Token",
                        type="password",
                        placeholder="hf_...",
                        info="Used for gated GGUF access by the llama.cpp backend."
                    )
                    hf_check_btn=gr.Button("Check Token & GGUF Access", variant="primary")
                    hf_access_status=gr.Markdown("Token not checked yet.")

        with gr.Accordion("🖥️ Console", open=True, elem_classes=["console-accordion"]):
            cmd_console=gr.HTML(console_html())
        timer=gr.Timer(0.5,active=True)
        timer.tick(console_html,outputs=cmd_console)
        timer.tick(training_progress_html,outputs=train_progress)

        all_dialogue_voice_dropdowns = [dialogue_components[i] for i in range(0, len(dialogue_components), 2)] + [classic_components[i] for i in range(0, len(classic_components), 2)]
        all_voice_refresh_pairs = [(tts_voice_refresh, tts_voice), (rt_voice_refresh, rt_voice), *list(zip(d_voice_refresh_btns, [dialogue_components[i] for i in range(0, len(dialogue_components), 2)])), *list(zip(c_voice_refresh_btns, [classic_components[i] for i in range(0, len(classic_components), 2)]))]
        def _refresh_secondary_voice_dropdowns():
            choices=list_voice_names(VOICES_DIR)
            return [gr.update(choices=choices) for _ in [rt_voice,*all_dialogue_voice_dropdowns]]

        # Voice Library
        voice_transcribe.click(transcribe_audio_ui,[voice_audio,voice_asr_model,voice_asr_language,voice_asr_batch],[voice_transcript,voice_status])
        voice_saved.change(load_voice_ui,voice_saved,[voice_preview,voice_saved_text])
        voice_refresh.click(lambda: gr.update(choices=list_voice_names(VOICES_DIR)),outputs=voice_saved)
        for refresh_btn, voice_dropdown in all_voice_refresh_pairs:
            refresh_btn.click(refresh_voice_choices,outputs=voice_dropdown,queue=False)
        voice_save.click(save_voice_ui,[voice_audio,voice_name,voice_transcript],[voice_status,voice_saved,tts_voice]).then(_refresh_secondary_voice_dropdowns,outputs=[rt_voice,*all_dialogue_voice_dropdowns])
        voice_delete.click(delete_voice_ui,[voice_saved],[voice_status,voice_saved,tts_voice]).then(_refresh_secondary_voice_dropdowns,outputs=[rt_voice,*all_dialogue_voice_dropdowns])
        # None semantics + waveform
        tts_voice.change(load_voice_ui,tts_voice,[tts_ref,tts_ref_text])
        rt_voice.change(lambda n: (lambda a: a if a and Path(a).is_file() else None)(resolve_voice(VOICES_DIR,n)[0]),rt_voice,rt_ref)
        # Hugging Face gated-access helper
        hf_check_btn.click(check_hf_token_access,hf_token_global,hf_access_status)
        gguf_checkpoint_refresh.click(
            refresh_cpp_checkpoint_choices,
            outputs=gguf_checkpoint,
            queue=False,
        )
        gguf_convert_btn.click(
            convert_cpp_lora_ui,
            [gguf_checkpoint,gguf_lora_name],
            [gguf_convert_status,gguf_registered],
        )
        # inference
        tts_backend.change(_tts_backend_updates,[tts_backend,hf_token_global],[tts_model,tts_adapter,tts_cpp_controls,tts_mode])
        tts_model.change(refresh_lora_adapter_for_backend,[tts_model,tts_backend,tts_adapter],tts_adapter,queue=False)
        tts_adapter_refresh.click(refresh_lora_adapter_for_model,tts_model,tts_adapter,queue=False)
        c_model.change(refresh_lora_adapter_for_backend,[c_model,c_backend,c_adapter],c_adapter,queue=False)
        c_adapter_refresh.click(refresh_lora_adapter_for_model,c_model,c_adapter,queue=False)

        hf_token_global.change(_tts_backend_updates,[tts_backend,hf_token_global],[tts_model,tts_adapter,tts_cpp_controls,tts_mode])
        c_backend.change(_backend_model_adapter_updates,[c_backend,hf_token_global],[c_model,c_adapter,c_cpp_controls])
        hf_token_global.change(_backend_model_adapter_updates,[c_backend,hf_token_global],[c_model,c_adapter,c_cpp_controls])
        cpp_vram_preset.change(_cpp_vram_preset_updates,cpp_vram_preset,[tts_model,cpp_kv,cpp_low_memory,cpp_heads_backend,cpp_audio_gpu,cpp_preset_note])
        c_cpp_vram_preset.change(_cpp_vram_preset_updates,c_cpp_vram_preset,[c_model,c_cpp_kv,c_cpp_low_memory,c_cpp_heads_backend,c_cpp_audio_gpu,c_cpp_preset_note])
        tts_go.click(generate_tts,[tts_model,tts_backend,hf_token_global,tts_mode,tts_text,tts_language,tts_voice,tts_ref,tts_ref_text,tts_expected_tokens,tts_tokens,tts_temp,tts_top_p,tts_top_k,tts_rep,tts_seed,tts_adapter,cpp_kv,cpp_gpu_layers,cpp_low_memory,cpp_heads_backend,cpp_audio_gpu,tts_chunk_mode,tts_chunk_silence,use_flash_attention,use_torch_compile,compile_mode],[tts_audio,tts_path])
        d_edit_outputs=[d_row_count,*dialogue_components,*dialogue_rows,d_status]; d_edit_inputs=[d_row_count,*dialogue_components]
        d_reset.click(dialogue_reset_rows,outputs=d_edit_outputs); d_clear_all.click(dialogue_clear_rows,[d_row_count,*dialogue_components],d_edit_outputs); d_compact.click(dialogue_compact_rows,[d_row_count,*dialogue_components],d_edit_outputs)
        for i in range(DIALOGUE_MAX_TURNS):
            d_add_btns[i].click(lambda count,*vals,_i=i: dialogue_add_after(_i,count,*vals),d_edit_inputs,d_edit_outputs)
            d_clone_btns[i].click(lambda count,*vals,_i=i: dialogue_clone_row(_i,count,*vals),d_edit_inputs,d_edit_outputs)
            d_delete_btns[i].click(lambda count,*vals,_i=i: dialogue_delete_row(_i,count,*vals),d_edit_inputs,d_edit_outputs)
            d_clear_btns[i].click(lambda count,*vals,_i=i: dialogue_clear_row(_i,count,*vals),d_edit_inputs,d_edit_outputs)
            d_up_btns[i].click(lambda count,*vals,_i=i: dialogue_move_row(_i,-1,count,*vals),d_edit_inputs,d_edit_outputs)
            d_down_btns[i].click(lambda count,*vals,_i=i: dialogue_move_row(_i,1,count,*vals),d_edit_inputs,d_edit_outputs)
        d_go.click(generate_dialogue,[d_tokens,d_temp,d_top_p,d_top_k,d_rep,use_flash_attention,use_torch_compile,compile_mode,d_row_count,*dialogue_components],[d_audio,d_path,d_status])
        c_edit_outputs=[c_row_count,*classic_components,*classic_rows,c_status]; c_edit_inputs=[c_row_count,*classic_components]
        c_reset.click(dialogue_reset_rows,outputs=c_edit_outputs); c_clear_all.click(dialogue_clear_rows,[c_row_count,*classic_components],c_edit_outputs); c_compact.click(dialogue_compact_rows,[c_row_count,*classic_components],c_edit_outputs)
        for i in range(DIALOGUE_MAX_TURNS):
            c_add_btns[i].click(lambda count,*vals,_i=i: dialogue_add_after(_i,count,*vals),c_edit_inputs,c_edit_outputs)
            c_clone_btns[i].click(lambda count,*vals,_i=i: dialogue_clone_row(_i,count,*vals),c_edit_inputs,c_edit_outputs)
            c_delete_btns[i].click(lambda count,*vals,_i=i: dialogue_delete_row(_i,count,*vals),c_edit_inputs,c_edit_outputs)
            c_clear_btns[i].click(lambda count,*vals,_i=i: dialogue_clear_row(_i,count,*vals),c_edit_inputs,c_edit_outputs)
            c_up_btns[i].click(lambda count,*vals,_i=i: dialogue_move_row(_i,-1,count,*vals),c_edit_inputs,c_edit_outputs)
            c_down_btns[i].click(lambda count,*vals,_i=i: dialogue_move_row(_i,1,count,*vals),c_edit_inputs,c_edit_outputs)
        c_go.click(generate_classic_dialogue,[c_model,c_backend,hf_token_global,c_language,c_pause,c_tokens,c_temp,c_top_p,c_top_k,c_rep,c_adapter,c_cpp_kv,c_cpp_gpu_layers,c_cpp_low_memory,c_cpp_heads_backend,c_cpp_audio_gpu,use_flash_attention,use_torch_compile,compile_mode,c_row_count,*classic_components],[c_audio,c_path,c_status])
        vg_go.click(generate_voice,[vg_text,vg_instruction,vg_language,vg_tokens,vg_temp,vg_top_p,vg_top_k,vg_rep,use_flash_attention,use_torch_compile,compile_mode],[vg_audio,vg_path])
        sfx_go.click(generate_sfx,[sfx_model,sfx_prompt,sfx_duration,sfx_steps,sfx_cfg,sfx_sigma,sfx_seed,sfx_max_tokens,sfx_temp,sfx_top_p,sfx_top_k,sfx_rep,use_flash_attention,sfx_v2_compile,compile_mode],[sfx_audio,sfx_path])
        rt_go.click(generate_realtime,[rt_text,rt_language,rt_voice,rt_ref,rt_temp,rt_top_p,rt_top_k,rt_rep,rt_rep_window,use_flash_attention,use_torch_compile,compile_mode],[rt_audio,rt_path])
        # projects + dataset
        dataset_browse_btn.click(browse_dataset_folder,dataset_folder,dataset_folder,queue=False)
        delete_project_btn.click(delete_project,project,[project,train_project,project_status])
        delete_train_project_btn.click(delete_project,train_project,[project,train_project,autotune_status])
        save_train_project_btn.click(save_project_config,[train_project,dataset_folder,dataset_language,train_profile,train_prepared_path,train_preset,output_name,train_ref,transcribe_missing,dataset_asr_model,dataset_asr_language,dataset_asr_batch,epochs,lr,micro_batch,grad_accum,lora_r,lora_alpha,lora_dropout,save_steps,train_flash_attention,enable_eval_audio,eval_text,eval_reference_audio,eval_max_tokens,save_mode,save_every_epochs],autotune_status)
        create_project_btn.click(create_project,new_project,[project,project_status]).then(lambda: gr.update(choices=list_projects()),outputs=train_project)
        project.change(load_project_full,project,[dataset_folder,dataset_language,dataset_profile,prepared_path,train_ref,transcribe_missing,dataset_asr_model,dataset_asr_language,dataset_asr_batch,train_profile,train_prepared_path,train_preset,output_name,epochs,lr,micro_batch,grad_accum,lora_r,lora_alpha,lora_dropout,save_steps,train_flash_attention,enable_eval_audio,eval_text,eval_reference_audio,eval_max_tokens,save_mode,save_every_epochs,project_status,autotune_status])
        scan_btn.click(scan_dataset,[dataset_folder,dataset_language,train_ref,transcribe_missing,dataset_asr_model,dataset_asr_language,dataset_asr_batch],[dataset_table,dataset_status])
        prepare_btn.click(prepare_dataset,[dataset_table,dataset_profile],[prepared_path,dataset_status])
        prepared_path.change(lambda x:x or "",prepared_path,train_prepared_path); dataset_profile.change(lambda x:x,dataset_profile,train_profile)
        save_project_btn.click(save_project_config,[project,dataset_folder,dataset_language,dataset_profile,prepared_path,train_preset,output_name,train_ref,transcribe_missing,dataset_asr_model,dataset_asr_language,dataset_asr_batch,epochs,lr,micro_batch,grad_accum,lora_r,lora_alpha,lora_dropout,save_steps,train_flash_attention,enable_eval_audio,eval_text,eval_reference_audio,eval_max_tokens,save_mode,save_every_epochs],project_status)
        train_project.change(load_project_full,train_project,[dataset_folder,dataset_language,dataset_profile,prepared_path,train_ref,transcribe_missing,dataset_asr_model,dataset_asr_language,dataset_asr_batch,train_profile,train_prepared_path,train_preset,output_name,epochs,lr,micro_batch,grad_accum,lora_r,lora_alpha,lora_dropout,save_steps,train_flash_attention,enable_eval_audio,eval_text,eval_reference_audio,eval_max_tokens,save_mode,save_every_epochs,project_status,autotune_status])
        autotune_btn.click(autotune_training,[train_profile,train_preset,train_prepared_path],[epochs,lr,micro_batch,grad_accum,lora_r,lora_alpha,lora_dropout,save_steps,autotune_status])
        for _schedule_trigger in (train_prepared_path, epochs, micro_batch, grad_accum):
            _schedule_trigger.change(
                estimate_training_schedule,
                [train_prepared_path,epochs,micro_batch,grad_accum],
                [steps_per_epoch_state,total_steps_state,training_schedule_info],
                queue=False,
            )

        resume_refresh_btn.click(refresh_training_checkpoints,output_name,resume_checkpoint,queue=False)
        output_name.change(refresh_training_checkpoints,output_name,resume_checkpoint,queue=False)
        train_project.change(lambda _p, name: refresh_training_checkpoints(name),[train_project,output_name],resume_checkpoint,queue=False)
        train_btn.click(start_training,[train_profile,train_prepared_path,output_name,train_preset,epochs,lr,micro_batch,grad_accum,lora_r,lora_alpha,lora_dropout,save_mode,save_steps,save_every_epochs,train_flash_attention,train_torch_compile,enable_eval_audio,eval_text,eval_reference_audio,eval_max_tokens,resume_checkpoint],[train_status,adapter_path])
        stop_btn.click(stop_training,outputs=train_status)
        tensorboard_btn.click(launch_tensorboard,output_name,tensorboard_status,queue=False)
        # maintenance actions
        unload_all_btn.click(unload_model,outputs=top_status)
        delete_outputs_btn.click(delete_output_audios,outputs=top_status)
        def _clear_voice_library_updates():
            choices=list_voice_names(VOICES_DIR)
            return ["Reference samples cleared.", gr.update(choices=choices,value="None"), gr.update(choices=choices,value="None"), gr.update(choices=choices,value="None"), *[gr.update(choices=choices,value="None") for _ in all_dialogue_voice_dropdowns]]
        delete_voices_btn.click(delete_all_reference_samples,outputs=[top_status,voice_saved]).then(_clear_voice_library_updates,outputs=[top_status,voice_saved,tts_voice,rt_voice,*all_dialogue_voice_dropdowns])
    return demo


if __name__ == "__main__":
    build_ui().queue(default_concurrency_limit=1).launch(inbrowser=True, server_name="127.0.0.1", css=CSS)
