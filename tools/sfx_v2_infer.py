from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--sigma-shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-SoundEffect v2.0 requires CUDA in this Easy GUI.")

    pipe = MossSoundEffectPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    audio = pipe(
        prompt=args.prompt,
        seconds=max(0.1, min(30.0, float(args.seconds))),
        num_inference_steps=int(args.steps),
        cfg_scale=float(args.cfg_scale),
        sigma_shift=float(args.sigma_shift),
        seed=int(args.seed),
    )[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    elif audio.ndim == 2:
        audio = audio.T
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, int(pipe.sample_rate))
    print(f"[SFX v2] Saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
