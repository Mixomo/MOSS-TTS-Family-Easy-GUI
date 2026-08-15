from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
DOWNSAMPLE_RATE = 1920
SAMPLE_RATE = 24000
N_QUANTIZERS = 32


def _load_ort_session(onnx_path: str | Path, use_gpu: bool = True):
    import onnxruntime as ort
    providers = []
    available = ort.get_available_providers()
    if use_gpu:
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    log.info("Loaded %s with providers=%s", Path(onnx_path).name, session.get_providers())
    return session


class OnnxAudioTokenizer:
    def __init__(self, encoder_path, decoder_path, n_quantizers=N_QUANTIZERS, use_gpu=True):
        self.n_quantizers = int(n_quantizers)
        self.sample_rate = SAMPLE_RATE
        self._encoder = _load_ort_session(encoder_path, use_gpu)
        self._decoder = _load_ort_session(decoder_path, use_gpu)
        self._enc_in = [x.name for x in self._encoder.get_inputs()]
        self._enc_out = [x.name for x in self._encoder.get_outputs()]
        self._dec_in = [x.name for x in self._decoder.get_inputs()]
        self._dec_out = [x.name for x in self._decoder.get_outputs()]

    def encode(self, waveform, n_quantizers=None):
        nq = self.n_quantizers if n_quantizers is None else int(n_quantizers)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 1:
            waveform = waveform[None, None, :]
        elif waveform.ndim == 2:
            waveform = waveform[None, :]
        length = waveform.shape[-1]
        padded = ((length + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        if padded != length:
            waveform = np.pad(waveform, ((0, 0), (0, 0), (0, padded - length)))
        outputs = self._encoder.run(self._enc_out, {self._enc_in[0]: waveform, self._enc_in[1]: np.array(nq, dtype=np.int64)})
        valid = int(outputs[1][0])
        return outputs[0][:, 0, :valid].T.astype(np.int64)

    def decode(self, audio_codes, n_quantizers=None):
        nq = self.n_quantizers if n_quantizers is None else int(n_quantizers)
        codes = np.asarray(audio_codes, dtype=np.int64)
        if codes.ndim == 2:
            if codes.shape[1] == self.n_quantizers and codes.shape[0] != self.n_quantizers:
                codes = codes.T
            codes = codes[:, None, :]
        elif codes.ndim != 3:
            raise ValueError(f"Expected 2D or 3D audio codes, got {codes.ndim}D")
        outputs = self._decoder.run(self._dec_out, {self._dec_in[0]: codes, self._dec_in[1]: np.array(nq, dtype=np.int64)})
        valid = int(outputs[1][0])
        return outputs[0][0, 0, :valid].astype(np.float32)

    def close(self):
        self._encoder = None
        self._decoder = None
