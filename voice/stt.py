"""Speech recognition (faster-whisper). Only runs on audio captured after the wake word."""

import logging

import numpy as np

from voice.config import VoiceConfig

logger = logging.getLogger(__name__)

MIN_SAMPLES = 16000 // 4  # shorter than 250 ms is noise, not a command
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0


class FasterWhisperTranscriber:
    def __init__(self, config: VoiceConfig):
        from faster_whisper import WhisperModel

        # One language: use it. Several: detect which one was spoken, among just those. None: Whisper decides.
        self.candidates = config.stt_languages
        self.language = self.candidates[0] if len(self.candidates) == 1 else None
        config.model_dir.mkdir(parents=True, exist_ok=True)
        self._model = WhisperModel(
            config.stt_model,
            device=config.stt_device,
            compute_type=config.stt_compute_type,
            download_root=str(config.model_dir / "whisper"),
        )

    def _language_for(self, samples: np.ndarray) -> str | None:
        """With several candidate languages, pick the likeliest. Forcing the wrong one makes
        Whisper translate: English forced as Vietnamese came out as Vietnamese words."""
        if len(self.candidates) < 2:
            return self.language

        _, _, probabilities = self._model.detect_language(samples)
        scores = dict(probabilities)
        return max(self.candidates, key=lambda code: scores.get(code, 0.0))

    def transcribe(self, audio: np.ndarray) -> str:
        """Text of a 16 kHz mono int16 recording; '' if there is no usable speech."""
        if len(audio) < MIN_SAMPLES:
            return ""

        samples = audio.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            samples,
            language=self._language_for(samples),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        kept = []

        for segment in segments:
            # Whisper invents text for noise; drop segments it is itself unsure about.
            if segment.no_speech_prob > NO_SPEECH_PROB and segment.avg_logprob < MIN_AVG_LOGPROB:
                continue

            kept.append(segment.text.strip())

        return " ".join(kept).strip()
