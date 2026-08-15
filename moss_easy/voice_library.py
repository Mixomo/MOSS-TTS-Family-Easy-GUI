from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
TEXT_KEYS = ("Text", "text", "Transcript", "transcript", "ReferenceText", "reference_text")


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", (name or "").strip()).strip(" .")
    return name or "voice"


def _audio_files(voices_dir: Path):
    voices_dir.mkdir(parents=True, exist_ok=True)
    for p in voices_dir.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p


def _same_stem_audio(voices_dir: Path, stem: str) -> Path | None:
    for p in _audio_files(voices_dir):
        if p.stem.lower() == stem.lower():
            return p
    return None


def _metadata_audio(voices_dir: Path, stem: str, data: dict) -> Path | None:
    raw = data.get("audio") if isinstance(data, dict) else None
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = voices_dir / candidate
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
    # Higgs-style metadata stores text only; audio is the same-stem file.
    return _same_stem_audio(voices_dir, stem)


def _read_transcript(audio: Path, meta: dict | None = None) -> str:
    txt = audio.with_suffix(".txt")
    if txt.is_file():
        try:
            value = txt.read_text(encoding="utf-8-sig").strip()
            if value:
                return value
        except OSError:
            pass
    data = meta
    if data is None:
        js = audio.with_suffix(".json")
        if js.is_file():
            try:
                data = json.loads(js.read_text(encoding="utf-8-sig"))
            except Exception:
                data = None
    if isinstance(data, dict):
        for key in TEXT_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def list_voice_names(voices_dir: Path) -> list[str]:
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    names = {p.stem for p in _audio_files(voices_dir)}
    # Also accept current-MOSS JSON metadata when its referenced audio is valid.
    for meta in voices_dir.glob("*.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8-sig"))
            audio = _metadata_audio(voices_dir, meta.stem, data)
            if audio is not None and audio.is_file():
                names.add(meta.stem)
        except Exception:
            continue
    return ["None", *sorted(names, key=str.lower)]


def save_voice_sample(voices_dir: Path, audio_path: str, name: str, transcript: str) -> str:
    if not audio_path:
        raise ValueError("Select or record reference audio first.")
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    src = Path(audio_path)
    if not src.is_file():
        raise ValueError(f"Reference audio is not a valid file: {src}")
    safe = _safe_name(name or src.stem)
    suffix = src.suffix.lower() if src.suffix.lower() in AUDIO_EXTS else ".wav"
    dst = voices_dir / f"{safe}{suffix}"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    transcript = (transcript or "").strip()
    # Higgs-compatible TXT sidecar.
    dst.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    # Superset JSON: Higgs reads Type/Text; current MOSS reads audio/transcript.
    dst.with_suffix(".json").write_text(
        json.dumps(
            {"Type": "Sample", "Text": transcript, "audio": str(dst), "transcript": transcript},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return f"Saved voice preset: {safe}"


def resolve_voice(voices_dir: Path, name: str) -> tuple[str | None, str]:
    if not name or name == "None":
        return None, ""
    voices_dir = Path(voices_dir)
    meta_path = voices_dir / f"{name}.json"
    meta = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            meta = None
    audio = _metadata_audio(voices_dir, name, meta or {})
    if audio is None or not audio.is_file():
        return None, ""
    return str(audio), _read_transcript(audio, meta)


def delete_voice(voices_dir: Path, name: str) -> str:
    if not name or name == "None":
        return "Select a saved voice preset first."
    voices_dir = Path(voices_dir)
    audio, _ = resolve_voice(voices_dir, name)
    candidates = {voices_dir / f"{name}.json", voices_dir / f"{name}.txt"}
    if audio:
        ap = Path(audio)
        try:
            if ap.is_file() and ap.parent.resolve() == voices_dir.resolve():
                candidates.add(ap)
                candidates.add(ap.with_suffix(".txt"))
                candidates.add(ap.with_suffix(".json"))
        except OSError:
            pass
    for path in candidates:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass
    return f"Deleted voice preset: {name}"
