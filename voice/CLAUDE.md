# Voice mode (`voice/`)

Optional: `python main.py --voice` or `VOICE_ENABLED=true`. Deps in `requirements_voice.txt`, settings in `.env.example`.

```powershell
python -m voice.enroll [--test]             # record voiceprint (6 phrases) / try it live
python -m voice.check [device] [seconds]    # mic troubleshooting: inputs, live level meter, wake-word score
RUN_VOICE_ENGINE_TESTS=1 python -m unittest tests.test_voice_engines   # real-model test (downloads ~200 MB once)
```

## Design
Voice is only another input/output for `ChatLoop`: transcripts go to `chat.submit()` (same path as keyboard, which keeps working); each finished turn's final answer returns via `chat.on_reply`. Tools, memory, checkpointer and thread are shared. Streamed sentences come from `turns.SpeechStream` (see root CLAUDE.md); the assistant queues them and speaks in bursts from `PROCESSING`, so the wake word is still checked between bursts.

`assistant.py` is the state machine (`VoiceState`, explicit `TRANSITIONS`). All audio runs on its one thread, except the PortAudio callback and a TTS synthesis helper thread. Components are injected, so tests use fakes: `wakeword.py` (openWakeWord ONNX behind `WakeWordDetector`: consecutive-frame hits + cooldown), `vad.py` (Silero VAD + `CommandRecorder`), `stt.py` (faster-whisper, only after wake word), `tts.py` (Piper; `Speaker` synthesizes the next sentence during playback), `audio.py` (sounddevice), `text.py` (wake-word stripping, filler/yes-no, markdown cleanup), `config.py` (`VoiceConfig.from_env`, validated).

## Behavior to preserve
- Wake word is monitored while the agent works ("Jarvis, stop" cancels a turn) but never while speaking.
- Mic queue flushed and detector reset after every playback (avoids self-activation).
- Recorder keeps ~240 ms pre-roll so "Jarvis, do X" in one breath works; wake word alone → beep and listen again.
- Tool approvals are spoken and answered by voice; silence or unclear answer declines.
- Pretrained openWakeWord model is `hey_jarvis` (no plain "jarvis"); a custom `.onnx` path works in `WAKE_WORD`.
- Models cached in `~/.cache/langchain-voice` (override `VOICE_MODEL_DIR`).

## Speaker verification (`speaker.py`, `enroll.py`)
`voice.enroll` records fixed phrases, makes WeSpeaker embeddings (sherpa-onnx, 26 MB model downloaded on first use), saves the average + calibrated cosine threshold as `voiceprint.npz`. If that file exists (or `SPEAKER_VERIFICATION=true`) the assistant verifies the audio of the whole interaction (wake word + command, so short utterances have enough speech) after transcription finds a real command. Rejected audio is dropped (`[REJECTED]`, transcript not shown); spoken approvals are checked too; under 0.6 s of speech fails closed. Convenience lock, not authentication (replays and voice clones pass).

## Speech language
`SPEECH_LANGUAGE` is `en`, `vi`, a list (`vi,en`: Whisper detection restricted to the listed codes in `stt.py`, because forcing one language makes Whisper translate the other) or `auto`. Non-English defaults to the `small` Whisper model. `Speaker` picks the Piper voice per reply (`TTS_VOICE_VI` for Vietnamese; voices load lazily).
