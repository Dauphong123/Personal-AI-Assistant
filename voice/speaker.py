"""Speaker verification: is this the enrolled user's voice?

A voiceprint is the average speaker embedding of several enrollment phrases
(`python -m voice.enroll`). Audio is accepted when its embedding is close enough to it
(cosine similarity). This is a convenience lock, not strong authentication: a recording
or a cloned voice can fool it.
"""

import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

MODEL_FILE = "wespeaker_en_voxceleb_resnet34.onnx"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{MODEL_FILE}"  # sic: the release tag is misspelled upstream
)
MIN_MODEL_BYTES = 5_000_000
DOWNLOAD_TIMEOUT = 30  # seconds without data before giving up

SAMPLE_RATE = 16000
MIN_SAMPLES = SAMPLE_RATE * 6 // 10  # under 0.6 s of audio is too little to judge a voice
MAX_SAMPLES = SAMPLE_RATE * 20  # only the last 20 s are used, to bound the cost
MIN_ENROLLMENT_CLIPS = 3

# Cosine similarity of different people is usually below ~0.35 and of the same person
# well above 0.6 with this model; the threshold is placed between the two.
IMPOSTOR_LEVEL = 0.35
THRESHOLD_RANGE = (0.4, 0.75)


class Embedder(Protocol):
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """L2-normalized embedding of 16 kHz mono int16 audio."""
        ...


class SpeakerEmbedder:
    """WeSpeaker ResNet34 (ONNX) through sherpa-onnx. The model (~26 MB) downloads on first use."""

    def __init__(self, model_dir: Path):
        import sherpa_onnx

        model = self._ensure_model(model_dir / "speaker")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=1, provider="cpu")
        )

    @staticmethod
    def _ensure_model(directory: Path) -> Path:
        model = directory / MODEL_FILE

        if model.is_file() and model.stat().st_size >= MIN_MODEL_BYTES:
            return model

        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading speaker model %s", MODEL_FILE)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".part")
        os.close(fd)

        try:
            with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT) as response, open(tmp, "wb") as out:
                expected = int(response.headers.get("Content-Length") or 0)
                shutil.copyfileobj(response, out)

            size = os.path.getsize(tmp)

            if size < MIN_MODEL_BYTES or (expected and size != expected):
                raise RuntimeError("downloaded speaker model is incomplete")

            os.replace(tmp, model)  # never leave a half-written model behind
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        return model

    def embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=audio.astype(np.float32) / 32768.0)
        stream.input_finished()
        vector = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def calibrate_threshold(similarities: list[float]) -> float:
    """Pick a threshold between how consistent the user sounds and a typical impostor."""
    low, high = THRESHOLD_RANGE
    return float(np.clip((float(np.mean(similarities)) + IMPOSTOR_LEVEL) / 2, low, high))


@dataclass
class VoicePrint:
    embeddings: np.ndarray  # (n, dim), one per enrollment clip
    threshold: float
    consistency: list[float]  # each clip against the average of the others

    @property
    def centroid(self) -> np.ndarray:
        return _unit(self.embeddings.mean(axis=0))

    @classmethod
    def from_embeddings(cls, embeddings: list[np.ndarray]) -> "VoicePrint":
        if len(embeddings) < MIN_ENROLLMENT_CLIPS:
            raise ValueError(f"need at least {MIN_ENROLLMENT_CLIPS} clips, got {len(embeddings)}")

        matrix = np.stack([_unit(e) for e in embeddings])
        consistency = []

        for i in range(len(matrix)):
            others = _unit(np.delete(matrix, i, axis=0).mean(axis=0))
            consistency.append(float(matrix[i] @ others))

        return cls(matrix, calibrate_threshold(consistency), consistency)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, embeddings=self.embeddings, threshold=self.threshold, consistency=self.consistency)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "VoicePrint":
        with np.load(path) as data:
            return cls(
                embeddings=data["embeddings"],
                threshold=float(data["threshold"]),
                consistency=[float(x) for x in data["consistency"]],
            )


@dataclass(frozen=True)
class VerifyResult:
    accepted: bool
    score: float
    reason: str = ""


class SpeakerVerifier:
    def __init__(self, embedder: Embedder, voiceprint: VoicePrint, threshold: float | None = None):
        self.embedder = embedder
        self.voiceprint = voiceprint
        self.centroid = voiceprint.centroid
        self.threshold = voiceprint.threshold if threshold is None else threshold

    def score(self, audio: np.ndarray) -> float:
        return float(_unit(self.embedder.embed(audio[-MAX_SAMPLES:])) @ self.centroid)

    def verify(self, audio: np.ndarray) -> VerifyResult:
        """Fails closed: too little audio, or an embedding error, is a rejection."""
        if len(audio) < MIN_SAMPLES:
            return VerifyResult(False, 0.0, "too little speech to recognize the voice")

        try:
            score = self.score(audio)
        except Exception:
            logger.exception("Speaker verification failed")
            return VerifyResult(False, 0.0, "voice check failed")

        if score < self.threshold:
            return VerifyResult(False, score, "voice not recognized")

        return VerifyResult(True, score)
