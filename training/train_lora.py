from __future__ import annotations
import json

import argparse
import importlib.util
import math
import random
import os
import time
import sys
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "moss_tts_upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

PROFILES = {
    "delay-v1.5-8b": {
        "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
        "dataset_module": "moss_tts_delay.finetuning.dataset",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "trainable_prefixes": ["language_model.layers."],
        "eval_sampling": (1.7, 0.8, 25, 1.0),
    },
    "delay-v1.0-8b": {
        "model": "OpenMOSS-Team/MOSS-TTS",
        "dataset_module": "moss_tts_delay.finetuning.dataset",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "trainable_prefixes": ["language_model.layers."],
        "eval_sampling": (1.7, 0.8, 25, 1.0),
    },
    "local-v1.5-4b": {
        "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "dataset_module": "moss_tts_local_v1.5.finetuning.dataset",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "trainable_prefixes": ["transformer.layers."],
        "eval_sampling": (1.7, 0.8, 25, 1.0),
    },
    "local-v1.0-1.7b": {
        "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        "dataset_module": "moss_tts_local.finetuning.dataset",
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "trainable_prefixes": ["language_model.layers."],
        "eval_sampling": (1.0, 0.95, 50, 1.1),
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU LoRA trainer for supported MOSS-TTS Easy GUI profiles")
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--save-mode", choices=["Every N Steps", "Every N Epochs"], default="Every N Steps")
    p.add_argument("--save-every-epochs", type=int, default=1)
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--use-flash-attention", action="store_true")
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--enable-eval-audio", action="store_true")
    p.add_argument("--eval-text", default="This is a MOSS-TTS training preview.")
    p.add_argument("--eval-reference-audio", default="")
    p.add_argument("--eval-max-new-tokens", type=int, default=512)
    return p.parse_args()


def resolve_attn():
    if importlib.util.find_spec("flash_attn") is not None and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "sdpa"


def import_dataset(profile_name: str):
    module = __import__(PROFILES[profile_name]["dataset_module"], fromlist=["*"])
    dataset_cls = getattr(module, "MossTTSSFTDataset", None)
    if dataset_cls is None:
        raise RuntimeError(f"Could not locate MossTTSSFTDataset in {module.__name__}")
    package = PROFILES[profile_name]["dataset_module"].rsplit(".", 1)[0]
    common = __import__(package + ".common", fromlist=["load_jsonl"])
    return module, dataset_cls, common.load_jsonl


def _patch_moss_embedding_accessor_for_peft(model):
    """Normalize custom MOSS get_input_embeddings() only for PEFT utilities."""
    base_cls = type(model)
    original = getattr(base_cls, "get_input_embeddings", None)
    if original is None:
        return

    try:
        original(model)
        return
    except TypeError:
        pass
    except Exception:
        return

    def patched(self, *args, **kwargs):
        if args or kwargs:
            return original(self, *args, **kwargs)
        if hasattr(self, "language_model"):
            return self.language_model.get_input_embeddings()
        if hasattr(self, "model") and hasattr(self.model, "language_model"):
            return self.model.language_model.get_input_embeddings()
        if hasattr(self, "embedding_list") and len(self.embedding_list):
            return self.embedding_list[0]
        return original(self)

    base_cls.get_input_embeddings = patched
    print("[LoRA] patched custom MOSS get_input_embeddings() for PEFT compatibility", flush=True)


def apply_lora(model, targets, args):
    for param in model.parameters():
        param.requires_grad = False

    _patch_moss_embedding_accessor_for_peft(model)

    # IMPORTANT: do not set task_type="CAUSAL_LM".
    # That creates PeftModelForCausalLM, whose HF-standard forward signature
    # injects output_hidden_states and other kwargs into MOSS custom forward().
    # Generic PeftModel applies the exact same LoRA modules without rewriting
    # the model's forward contract.
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
    )

    resume_dir = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_dir:
        if not (resume_dir / "adapter_config.json").is_file():
            raise RuntimeError(f"Resume checkpoint has no adapter_config.json: {resume_dir}")
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        print(f"[LoRA] loaded adapter checkpoint: {resume_dir}", flush=True)
    else:
        model = get_peft_model(model, cfg)

    # Keep adapters only on the architecture's global/temporal Qwen backbone.
    allowed_prefixes = PROFILES[args.profile].get("trainable_prefixes", [])
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = any(prefix in name for prefix in allowed_prefixes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("PEFT created no trainable LoRA parameters in language_model.layers for this profile.")

    print(
        f"[LoRA] wrapper={type(model).__name__} trainable={trainable:,} / "
        f"total={total:,} ({100*trainable/total:.4f}%)",
        flush=True,
    )
    if type(model).__name__ == "PeftModelForCausalLM":
        raise RuntimeError(
            "Incompatible PEFT wrapper detected: PeftModelForCausalLM. "
            "MOSS custom models must use generic PeftModel."
        )
    return model


def _audio_from_decoded(messages):
    if not messages:
        raise RuntimeError("Processor returned no decoded messages.")
    msg = messages[0]
    audio_list = getattr(msg, "audio_codes_list", None)
    if not audio_list:
        raise RuntimeError("Processor returned no decoded audio.")
    audio = audio_list[0]
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()
    else:
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            audio = audio[0]
    return audio


def generate_eval_audio(model, processor, text: str, out_path: Path, max_new_tokens: int, profile_name: str = "delay-v1.5-8b", reference_audio: str = ""):
    """Best-effort preview using the same MOSS generation/decoding path as inference."""
    was_training = model.training
    model.eval()

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = getattr(base, "config", None)
    old_use_cache = getattr(config, "use_cache", None) if config is not None else None
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = True

    try:
        reference = None
        if reference_audio:
            ref_path = Path(reference_audio)
            if ref_path.is_file():
                reference = [str(ref_path.resolve())]
        conversation = [[processor.build_user_message(text=(text or "").strip(), reference=reference)]]
        batch = processor(conversation, mode="generation")
        device = next(base.parameters()).device
        generation_model = model if hasattr(model, "generate") else base
        with torch.no_grad():
            outputs = generation_model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(PROFILES[profile_name]["eval_sampling"][0]),
                audio_top_p=float(PROFILES[profile_name]["eval_sampling"][1]),
                audio_top_k=int(PROFILES[profile_name]["eval_sampling"][2]),
                audio_repetition_penalty=float(PROFILES[profile_name]["eval_sampling"][3]),
            )
        messages = processor.decode(outputs)
        audio = _audio_from_decoded(messages)
        sample_rate = int(processor.model_config.sampling_rate)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sample_rate)
        return sample_rate, audio
    finally:
        if config is not None and old_use_cache is not None:
            config.use_cache = old_use_cache
        if was_training:
            model.train()
        torch.cuda.empty_cache()




def _sanitize_moss_forward_kwargs(profile: str, batch: dict) -> dict:
    """Build safe kwargs for custom MOSS forward methods.

    MOSS Delay forces several Qwen3 arguments internally and forwards **kwargs.
    Re-sending those keys from the trainer causes duplicate-keyword failures.
    """
    kwargs = dict(batch)

    # Reserved by the outer MOSS model / unsafe to forward from collators.
    reserved = (
        "output_hidden_states",
        "return_dict",
        "output_attentions",
        "cache_position",
        "past_key_values",
        "inputs_embeds",
        "position_ids",
    )
    for key in reserved:
        kwargs.pop(key, None)

    # Raw metadata must never reach model.forward().
    metadata_keys = (
        "audio",
        "audio_path",
        "reference_audio",
        "reference_audio_path",
        "text",
        "language",
        "metadata",
        "sample_id",
    )
    for key in metadata_keys:
        kwargs.pop(key, None)

    kwargs["use_cache"] = False

    # Official Delay recipe uses weighted text/audio-channel loss.
    if "delay" in profile.lower() and "channelwise_loss_weight" not in kwargs:
        kwargs["channelwise_loss_weight"] = [1.0] + [32.0] * 32

    return kwargs


def _assert_training_batch_compatible(profile: str, kwargs: dict):
    """Fail early with a concise message if the prepared dataset is mismatched."""
    if "input_ids" not in kwargs:
        raise RuntimeError("Prepared batch is missing input_ids.")
    if "labels" not in kwargs:
        raise RuntimeError("Prepared batch is missing labels.")

    input_ids = kwargs["input_ids"]
    labels = kwargs["labels"]

    if not torch.is_tensor(input_ids) or not torch.is_tensor(labels):
        raise RuntimeError("Prepared input_ids/labels must be torch tensors.")

    if "delay" in profile.lower():
        if input_ids.ndim != 3 or input_ids.shape[-1] != 33:
            raise RuntimeError(
                f"MOSS-TTS Delay expects input_ids [B, S, 33], got {tuple(input_ids.shape)}. "
                "Re-run Dataset Preparation with the Delay 8B profile."
            )
        if labels.ndim != 3 or labels.shape[-1] != 33:
            raise RuntimeError(
                f"MOSS-TTS Delay expects labels [B, S, 33], got {tuple(labels.shape)}."
            )




def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save_training_checkpoint(
    accelerator,
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    global_step: int,
    epoch_index: int,
    next_batch_index: int,
):
    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(checkpoint_dir)

    state = {
        "format_version": 1,
        "global_step": int(global_step),
        "epoch_index": int(epoch_index),
        "next_batch_index": int(next_batch_index),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "global_step": int(global_step),
                "epoch_index": int(epoch_index),
                "next_batch_index": int(next_batch_index),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_dir


def _load_training_checkpoint_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    device,
):
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.is_file():
        raise RuntimeError(
            f"Checkpoint is adapter-only and cannot fully resume training: {checkpoint_dir}"
        )

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    _optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(state["scheduler"])

    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])

    return (
        int(state.get("global_step", 0)),
        int(state.get("epoch_index", 0)),
        int(state.get("next_batch_index", 0)),
    )






def _training_feature_self_check():
    """Fail before model loading if a packaging regression breaks trainer helpers."""
    required_globals = {
        "json": json,
        "math": math,
        "random": random,
        "np": np,
        "torch": torch,
        "sf": sf,
    }
    missing = [name for name, value in required_globals.items() if value is None]
    if missing:
        raise RuntimeError(
            "Trainer feature imports are incomplete: " + ", ".join(missing)
        )

    # Validate serializer availability before spending minutes loading 8B.
    probe = json.dumps({"trainer": "ok", "version": 1})
    if not probe:
        raise RuntimeError("Trainer JSON serializer self-check failed.")




def _tensorboard_add_audio(writer, tag: str, audio, global_step: int, sample_rate: int):
    """Scope-safe TensorBoard audio writer using module-level NumPy/Torch imports."""
    if writer is None:
        return
    audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim == 1:
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
    elif audio_np.ndim == 2:
        # TensorBoard expects [channels, frames].
        tensor = torch.from_numpy(audio_np)
        if tensor.shape[0] > tensor.shape[1] and tensor.shape[1] <= 8:
            tensor = tensor.transpose(0, 1)
    else:
        audio_np = np.squeeze(audio_np)
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
    writer.add_audio(tag, tensor, global_step, sample_rate=int(sample_rate))
    writer.flush()



def main():
    _training_feature_self_check()
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA Training is configured for a single NVIDIA CUDA GPU.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    profile = PROFILES[args.profile]
    model_path = args.model_path or profile["model"]
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=resolve_attn() if args.use_flash_attention else "sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = apply_lora(model, profile["targets"], args)
    print(f"[LoRA] active PEFT wrapper: {type(model).__name__}", flush=True)

    # Do not call transformers.PreTrainedModel.enable_input_require_grads().
    # MOSS custom model classes do not all expose the standard HF
    # get_input_embeddings() contract (some remote-code revisions take
    # multimodal input_ids), so the generic helper can raise TypeError.
    #
    # Gradient checkpointing only needs the tensor entering the transformer
    # stack to require grad. Register a forward hook on the real text embedding
    # module used by each supported MOSS architecture.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    def _resolve_input_embedding_module(module):
        candidates = [
            "language_model.embed_tokens",          # MossTTSDelayModel
            "model.language_model.embed_tokens",    # wrapped Delay variants
            "transformer.embed_tokens",             # MossTTSLocal v1.5
            "model.transformer.embed_tokens",       # wrapped Local v1.5
            "embedding_list.0",                     # MosiTTSModel / Local
            "model.embedding_list.0",               # outer Local wrapper
        ]
        for dotted in candidates:
            cur = module
            try:
                for part in dotted.split("."):
                    if part.isdigit():
                        cur = cur[int(part)]
                    else:
                        cur = getattr(cur, part)
                if isinstance(cur, torch.nn.Module):
                    return cur, dotted
            except Exception:
                continue
        return None, None

    embedding_module, embedding_path = _resolve_input_embedding_module(base)
    if embedding_module is None:
        raise RuntimeError(
            "Could not locate the MOSS text embedding module required for "
            "gradient checkpointing."
        )

    def _require_grad_on_embedding_output(_module, _inputs, output):
        if torch.is_tensor(output) and torch.is_grad_enabled():
            output.requires_grad_(True)
        return output

    embedding_module.register_forward_hook(_require_grad_on_embedding_output)
    print(f"[LoRA] gradient-checkpoint input hook: {embedding_path}", flush=True)

    # Transformers' generic gradient_checkpointing_enable() calls
    # enable_input_require_grads(), which assumes a standard zero-argument
    # get_input_embeddings(). MOSS remote-code models do not consistently
    # implement that contract, so calling the generic helper crashes.
    #
    # Prefer the lower-level checkpointing setter used internally by
    # PreTrainedModel. It toggles the checkpointing flags on compatible
    # submodules without re-entering enable_input_require_grads().
    checkpointing_enabled = False
    gc_setter = getattr(base, "_set_gradient_checkpointing", None)
    if callable(gc_setter):
        try:
            import functools
            from torch.utils.checkpoint import checkpoint
            gradient_checkpointing_func = functools.partial(
                checkpoint,
                use_reentrant=True,
            )
            gc_setter(
                enable=True,
                gradient_checkpointing_func=gradient_checkpointing_func,
            )
            checkpointing_enabled = True
            print("[LoRA] gradient checkpointing: enabled via _set_gradient_checkpointing", flush=True)
        except TypeError:
            # Older/custom Transformers signature.
            try:
                gc_setter(base, True)
                checkpointing_enabled = True
                print("[LoRA] gradient checkpointing: enabled via legacy _set_gradient_checkpointing", flush=True)
            except Exception as exc:
                print(f"[LoRA] gradient checkpointing unavailable: {exc}", flush=True)
        except Exception as exc:
            print(f"[LoRA] gradient checkpointing unavailable: {exc}", flush=True)

    if not checkpointing_enabled:
        # Do not call base.gradient_checkpointing_enable(): for MOSS Delay it
        # re-enters the incompatible get_input_embeddings() method. Training
        # remains valid without checkpointing, at the cost of higher VRAM.
        print(
            "[LoRA] gradient checkpointing disabled for this MOSS revision "
            "(custom get_input_embeddings API is incompatible with the generic Transformers helper).",
            flush=True,
        )
    if hasattr(base, "config") and hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    module, dataset_cls, load_jsonl = import_dataset(args.profile)
    records = load_jsonl(args.train_jsonl)
    if not records:
        raise RuntimeError(f"No records found in {args.train_jsonl}")
    dataset = dataset_cls(records=records, processor=processor)
    collate = getattr(dataset, "collate_fn", None) or getattr(module, "collate_fn", None)
    if collate is None:
        raise RuntimeError("Upstream dataset does not expose a collate function.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = max(1, args.epochs * updates_per_epoch)
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    global_step = 0
    start_epoch = 0
    resume_batch_index = 0
    if args.resume_checkpoint:
        resume_dir = Path(args.resume_checkpoint)
        global_step, start_epoch, resume_batch_index = _load_training_checkpoint_state(
            resume_dir,
            optimizer,
            scheduler,
            accelerator.device,
        )
        if resume_batch_index >= len(loader):
            start_epoch += resume_batch_index // max(1, len(loader))
            resume_batch_index = resume_batch_index % max(1, len(loader))
        print(
            f"[resume] checkpoint={resume_dir} global_step={global_step} "
            f"epoch_index={start_epoch} next_batch_index={resume_batch_index}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if accelerator.is_main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_run = output_dir / "tensorboard" / f"{time.strftime('%Y%m%d_%H%M%S')}"
            writer = SummaryWriter(log_dir=str(tb_run))
            print(f"[tensorboard] run={tb_run}", flush=True)
        except Exception as exc:
            print(f"[tensorboard] disabled: {exc}", flush=True)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch_index:
                continue
            with accelerator.accumulate(model):
                kwargs = _sanitize_moss_forward_kwargs(args.profile, batch)
                kwargs = {
                    k: (v.to(accelerator.device) if torch.is_tensor(v) else v)
                    for k, v in kwargs.items()
                }
                _assert_training_batch_compatible(args.profile, kwargs)

                if global_step == 0 and accelerator.is_main_process:
                    tensor_shapes = {
                        k: tuple(v.shape) for k, v in kwargs.items() if torch.is_tensor(v)
                    }
                    non_tensor_keys = [
                        k for k, v in kwargs.items() if not torch.is_tensor(v)
                    ]
                    print(f"[LoRA] first batch tensor shapes: {tensor_shapes}", flush=True)
                    if non_tensor_keys:
                        print(f"[LoRA] first batch non-tensor kwargs: {non_tensor_keys}", flush=True)

                outputs = model(**kwargs)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise RuntimeError(
                        "MOSS forward returned no loss. Check that the prepared dataset "
                        "contains labels and matches the selected training architecture."
                    )
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"Non-finite training loss detected: {loss.detach().float().item()}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                loss_value = loss.detach().float().item()
                if accelerator.is_main_process:
                    print(f"[train] epoch={epoch+1}/{args.epochs} step={global_step}/{total_steps} loss={loss_value:.6f}", flush=True)
                    if writer is not None:
                        writer.add_scalar("train/loss", loss_value, global_step)
                        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], global_step)

                if args.save_mode == "Every N Steps" and args.save_steps > 0 and global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_dir = _save_training_checkpoint(
                            accelerator,
                            model,
                            optimizer,
                            scheduler,
                            output_dir,
                            global_step,
                            epoch,
                            batch_index + 1,
                        )
                        unwrapped = accelerator.unwrap_model(model)
                        print(f"[checkpoint] saved full state={checkpoint_dir}", flush=True)

                        if args.enable_eval_audio:
                            try:
                                wav_path = output_dir / "eval_audio" / f"step_{global_step:07d}.wav"
                                sr, audio = generate_eval_audio(
                                    unwrapped,
                                    processor,
                                    args.eval_text,
                                    wav_path,
                                    args.eval_max_new_tokens,
                                    args.profile,
                                    args.eval_reference_audio,
                                )
                                print(f"[eval-audio] step={global_step} file={wav_path}", flush=True)
                                if writer is not None:
                                    _tensorboard_add_audio(writer, "eval/generated_audio", audio, global_step, sr)
                            except Exception as exc:
                                print(f"[eval-audio] preview failed at step {global_step}: {exc}", flush=True)
                                unwrapped.train()
        accelerator.wait_for_everyone()

        # Exact epoch-based cadence. This is intentionally independent of
        # optimizer-step arithmetic so partial final accumulation batches cannot
        # shift checkpoint timing.
        if (
            args.save_mode == "Every N Epochs"
            and args.save_every_epochs > 0
            and (epoch + 1) % args.save_every_epochs == 0
        ):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_dir = _save_training_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    output_dir,
                    global_step,
                    epoch + 1,
                    0,
                )
                unwrapped = accelerator.unwrap_model(model)
                print(
                    f"[checkpoint] epoch={epoch+1} step={global_step} saved full state={checkpoint_dir}",
                    flush=True,
                )

                if args.enable_eval_audio:
                    try:
                        wav_path = output_dir / "eval_audio" / f"epoch_{epoch+1:03d}_step_{global_step:07d}.wav"
                        sr, audio = generate_eval_audio(
                            unwrapped,
                            processor,
                            args.eval_text,
                            wav_path,
                            args.eval_max_new_tokens,
                            args.profile,
                            args.eval_reference_audio,
                        )
                        print(
                            f"[eval-audio] epoch={epoch+1} step={global_step} file={wav_path}",
                            flush=True,
                        )
                        if writer is not None:
                            _tensorboard_add_audio(writer, "eval/generated_audio", audio, global_step, sr)
                    except Exception as exc:
                        print(
                            f"[eval-audio] preview failed at epoch {epoch+1}, step {global_step}: {exc}",
                            flush=True,
                        )
                        unwrapped.train()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir)
        print(f"[done] LoRA adapter saved to {output_dir}")

        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    main()
