MOSS-TTS Easy GUI accelerator wheel override

The fixed Windows runtime is:
- Python 3.12.10
- PyTorch 2.9.0 + cu128
- FlashAttention 2.8.3
- Triton Windows 3.5.1.post24

If the installer cannot download FlashAttention, place this exact compatible wheel here:
flash_attn-2.8.3+cu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl

The installer always prefers a local flash_attn*.whl in this folder.
