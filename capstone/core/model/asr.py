import logging
from typing import Dict, List, Tuple

import torch

from core.config import ASR_conf


class Transcriber:
    def __init__(self, config: ASR_conf = ASR_conf()):
        self.config = config
        self._model = None   ## lazy: never load whisper when a course has no video

    def _resolve_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            device = self._resolve_device()
            ## int8 asked on cpu too — faster-whisper falls back cleanly
            self._model = WhisperModel(
                self.config.model_name, device=device, compute_type=self.config.compute_type
            )
            logging.info(f"Whisper {self.config.model_name} loaded on {device}")
        return self._model

    def transcribe(self, path: str) -> Tuple[float, List[Dict]]:
        ## native segment timestamps -> (t_lo, t_hi); no VAD tuning in v1
        segments, info = self.model.transcribe(path)
        out = [{"t_lo": s.start, "t_hi": s.end, "text": s.text.strip()}
               for s in segments if s.text.strip()]
        return float(info.duration), out
