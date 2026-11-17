"""Web voice: browser recordings to text, answers to WAV audio. Fakes only (no models)."""

import io
import json
import unittest
import wave

import numpy as np

from tests import stubs

stubs.install()

from backend.server import MAX_AUDIO, PROTOCOL, make_server  # noqa: E402
from backend.sessions import SessionManager  # noqa: E402
from backend.voice import WebVoice  # noqa: E402
from tests.test_backend_server import TOKEN, ServerCase  # noqa: E402
from tests.test_backend_session import scripted  # noqa: E402


class FakeTranscriber:
    def __init__(self):
        self.heard = []

    def transcribe(self, audio):
        self.heard.append(audio)
        return "hello from the mic"


class FakeSynth:
    def __init__(self, name):
        self.name = name
        self.said = []

    def synthesize(self, text):
        self.said.append(text)
        return np.array([0, 1000, -1000, 0], dtype=np.int16), 22050


def fake_decode(data):
    assert isinstance(data, bytes)
    return np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)


def make_voice(stt=True, tts=True):
    transcriber = FakeTranscriber()
    synths = {"en": FakeSynth("en"), "vi": FakeSynth("vi")}
    voice = WebVoice(
        transcriber=(lambda: transcriber) if stt else None,
        synthesizer=(lambda language: synths[language]) if tts else None,
        decode=fake_decode,
    )
    return voice, transcriber, synths


class WebVoiceTests(unittest.TestCase):
    def test_capabilities_follow_the_engines(self):
        self.assertEqual((make_voice()[0].can_transcribe, make_voice()[0].can_speak), (True, True))
        self.assertEqual((make_voice(stt=False)[0].can_transcribe, make_voice(tts=False)[0].can_speak), (False, False))

    def test_transcribe_decodes_to_16khz_int16(self):
        voice, transcriber, _ = make_voice()
        self.assertEqual(voice.transcribe(b"webm bytes"), "hello from the mic")
        audio = transcriber.heard[0]
        self.assertEqual(audio.dtype, np.int16)
        self.assertEqual(list(audio), [0, 16383, -16383, 32767])

    def test_engines_are_built_once_and_lazily(self):
        built = []

        def factory():
            built.append(1)
            return FakeTranscriber()

        voice = WebVoice(transcriber=factory, decode=fake_decode)
        self.assertEqual(built, [])
        voice.transcribe(b"a")
        voice.transcribe(b"b")
        self.assertEqual(built, [1])

    def test_speak_returns_a_wav_file(self):
        voice, _, synths = make_voice()
        data = voice.speak("**Done.** Ran `pytest`.")
        with wave.open(io.BytesIO(data)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 22050))
            self.assertEqual(wav.getnframes(), 4)
        self.assertEqual(synths["en"].said, ["Done. Ran pytest."])  # markdown cleaned for speech

    def test_code_is_not_read_aloud(self):
        voice, _, synths = make_voice()
        voice.speak("Here:\n```python\nprint(1)\n```")
        self.assertNotIn("print", synths["en"].said[0])

    def test_vietnamese_uses_the_vietnamese_voice(self):
        voice, _, synths = make_voice()
        voice.speak("Tôi đã chạy các bài kiểm tra và tất cả đều đạt.")
        self.assertEqual(len(synths["vi"].said), 1)
        self.assertEqual(synths["en"].said, [])

    def test_nothing_to_say_is_empty(self):
        voice, _, synths = make_voice()
        self.assertEqual(voice.speak("   "), b"")
        self.assertEqual(voice.speak("***"), b"")
        self.assertEqual(synths["en"].said, [])

    def test_unavailable_engines_raise(self):
        voice = WebVoice(decode=fake_decode)
        with self.assertRaises(RuntimeError):
            voice.transcribe(b"x")
        with self.assertRaises(RuntimeError):
            voice.speak("hi")

    def test_from_env_without_packages_offers_nothing(self):
        voice = WebVoice.from_env(find_spec=lambda name: None)
        self.assertFalse(voice.can_transcribe)
        self.assertFalse(voice.can_speak)

    def test_from_env_with_packages_offers_both_without_loading_models(self):
        voice = WebVoice.from_env(find_spec=lambda name: object())
        self.assertTrue(voice.can_transcribe)
        self.assertTrue(voice.can_speak)


class VoiceRouteTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.voice, self.transcriber, self.synths = make_voice()
        self.server.voice = self.voice

    def post_audio(self, data=b"opus data", content_type="audio/webm;codecs=opus", **kwargs):
        return self.request("POST", "/api/voice/transcribe", raw=data, headers={"Content-Type": content_type}, **kwargs)

    def test_handshake_reports_voice(self):
        status, data = self.api("POST", "/api/handshake", {"protocol": PROTOCOL})
        self.assertEqual(data["features"]["voice"], {"transcribe": True, "speak": True})
        self.assertEqual(data["limits"]["max_audio"], MAX_AUDIO)

    def test_transcribe(self):
        status, _, body = self.post_audio()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"text": "hello from the mic"})

    def test_transcribe_needs_audio_content_type(self):
        self.assertEqual(self.post_audio(content_type="application/json")[0], 415)
        self.assertEqual(self.post_audio(content_type="text/plain")[0], 415)

    def test_transcribe_size_limits(self):
        self.assertEqual(self.post_audio(data=b"")[0], 400)
        self.assertEqual(self.post_audio(data=b"x" * (MAX_AUDIO + 1))[0], 413)

    def test_transcribe_needs_token_and_origin(self):
        self.assertEqual(self.post_audio(token=False)[0], 401)
        status, _, _ = self.request(
            "POST", "/api/voice/transcribe", raw=b"x",
            headers={"Content-Type": "audio/webm", "Origin": "http://evil.example"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(self.transcriber.heard, [])

    def test_unreadable_audio_is_a_422(self):
        def broken(data):
            raise ValueError("not audio")

        self.voice._decode = broken
        self.assertEqual(self.post_audio()[0], 422)

    def test_speak_returns_wav(self):
        status, resp, body = self.request("POST", "/api/voice/speak", {"text": "All done."})
        self.assertEqual(status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "audio/wav")
        self.assertTrue(body.startswith(b"RIFF"))

    def test_speak_needs_text(self):
        for body in ({}, {"text": 3}, {"text": "  "}):
            self.assertEqual(self.request("POST", "/api/voice/speak", body)[0], 400, body)

    def test_speak_with_nothing_speakable_is_204(self):
        self.assertEqual(self.request("POST", "/api/voice/speak", {"text": "***"})[0], 204)

    def test_csp_allows_playing_blob_audio(self):
        _, resp, _ = self.request("GET", "/", token=False)
        self.assertIn("media-src 'self' blob:", resp.getheader("Content-Security-Policy"))

    def test_wrong_method_and_unknown_voice_route(self):
        self.assertEqual(self.request("GET", "/api/voice/speak")[0], 405)
        self.assertEqual(self.request("POST", "/api/voice/other", {})[0], 404)


class NoVoiceTests(ServerCase):
    def test_voice_routes_are_503_without_voice(self):
        self.assertEqual(self.request("POST", "/api/voice/transcribe", raw=b"x", headers={"Content-Type": "audio/webm"})[0], 503)
        self.assertEqual(self.request("POST", "/api/voice/speak", {"text": "hi"})[0], 503)


class RealDecodeTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("av"), "PyAV not installed")
    def test_a_wav_recording_decodes_to_16khz_samples(self):
        from backend.voice import decode_recording

        rate = 48000
        tone = (np.sin(np.linspace(0, 2 * np.pi * 440, rate)) * 12000).astype(np.int16)
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(tone.tobytes())
        samples = decode_recording(out.getvalue())
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(len(samples), 16000, delta=200)
        self.assertGreater(float(np.abs(samples).max()), 0.2)

    @unittest.skipUnless(__import__("importlib").util.find_spec("av"), "PyAV not installed")
    def test_a_webm_opus_recording_decodes(self):
        # What Chrome's MediaRecorder produces.
        import av

        from backend.voice import decode_recording

        out = io.BytesIO()
        try:
            with av.open(out, mode="w", format="webm") as container:
                stream = container.add_stream("libopus", rate=48000)
                stream.layout = "mono"
                t = np.arange(48000) / 48000
                tone = (np.sin(2 * np.pi * 440 * t) * 0.4).astype(np.float32).reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(tone, format="flt", layout="mono")
                frame.sample_rate = 48000
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
        except Exception as exc:  # this PyAV build has no opus encoder
            self.skipTest(f"cannot encode webm/opus here: {exc}")
        samples = decode_recording(out.getvalue())
        self.assertAlmostEqual(len(samples), 16000, delta=800)
        self.assertGreater(float(np.abs(samples).max()), 0.2)

    @unittest.skipUnless(__import__("importlib").util.find_spec("av"), "PyAV not installed")
    def test_junk_is_an_error_not_a_crash(self):
        from backend.voice import decode_recording

        with self.assertRaises(Exception):
            decode_recording(b"this is not audio at all")


class PartialVoiceTests(unittest.TestCase):
    def test_speak_only_voice_refuses_transcribe(self):
        voice, _, _ = make_voice(stt=False)
        server = make_server(SessionManager(app=None, run=scripted()), TOKEN, port=0, voice=voice)
        self.addCleanup(server.server_close)
        self.assertEqual(server.describe()["features"]["voice"], {"transcribe": False, "speak": True})
        self.assertEqual(server.describe()["limits"]["max_audio"], 0)


if __name__ == "__main__":
    unittest.main()
