"""Speech recognition (faster-whisper). Only runs on audio captured after the wake word."""

import logging

import numpy as np

from voice.config import VoiceConfig

logger = logging.getLogger(__name__)

MIN_SAMPLES = 16000 // 4  # shorter than 250 ms is noise, not a command
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
BEAM_SIZE = 5  # greedy decoding (1) dropped and misspelled Vietnamese words: CER 23.8% -> 15.9%

# Whisper's detector leans English on short Vietnamese ("Không" vi 0.01 / en 0.67), while real
# English scores >= 0.8 unless very short ("Okay" spoken slowly: 0.66). So another listed language
# wins outright only above SURE; between UNSURE and SURE both are transcribed and the more
# confident transcript wins; below UNSURE the main (first listed) language is used.
OTHER_LANGUAGE_SURE = 0.8
OTHER_LANGUAGE_UNSURE = 0.3


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

    def _languages_for(self, samples: np.ndarray) -> list[str | None]:
        """The language(s) to transcribe in. Forcing the wrong one makes Whisper translate:
        Vietnamese forced as English came out as an English translation."""
        if len(self.candidates) < 2:
            return [self.language]

        _, _, probabilities = self._model.detect_language(samples)
        scores = dict(probabilities)
        main = self.candidates[0]
        other = max(self.candidates[1:], key=lambda code: scores.get(code, 0.0))
        odds = scores.get(other, 0.0)

        if odds >= OTHER_LANGUAGE_SURE and odds > scores.get(main, 0.0):
            return [other]

        if odds >= OTHER_LANGUAGE_UNSURE:
            return [main, other]

        return [main]

    def _decode(self, samples: np.ndarray, language: str | None) -> tuple[str, float]:
        """(text, mean log-probability of the kept segments); an empty text scores -inf."""
        segments, _ = self._model.transcribe(
            samples,
            language=language,
            beam_size=BEAM_SIZE,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        kept = []
        logprobs = []

        for segment in segments:
            # Whisper invents text for noise; drop segments it is itself unsure about.
            if segment.no_speech_prob > NO_SPEECH_PROB and segment.avg_logprob < MIN_AVG_LOGPROB:
                continue

            kept.append(segment.text.strip())
            logprobs.append(segment.avg_logprob)

        text = " ".join(kept).strip()
        return text, (sum(logprobs) / len(logprobs) if text else float("-inf"))

    def transcribe(self, audio: np.ndarray) -> str:
        """Text of a 16 kHz mono int16 recording; '' if there is no usable speech."""
        if len(audio) < MIN_SAMPLES:
            return ""

        samples = audio.astype(np.float32) / 32768.0
        results = [self._decode(samples, language) for language in self._languages_for(samples)]
        # max keeps the first (main language) on a tie
        return max(results, key=lambda result: result[1])[0]
