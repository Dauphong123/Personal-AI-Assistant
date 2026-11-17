# Voice mode (`voice/`)

Optional: `python main.py --voice` or `VOICE_ENABLED=true`. Wake word and/or push-to-talk with hands-free (Wispr Flow style, F022). Deps in `requirements_voice.txt`, settings in `.env.example`.

```powershell
python -m voice.enroll [--test]             # record voiceprint (6 phrases) / try it live
python -m voice.check [device] [seconds]    # mic troubleshooting: inputs, live level meter, wake-word score
RUN_VOICE_ENGINE_TESTS=1 python -m unittest tests.test_voice_engines   # real-model test (downloads ~200 MB once)
```

## Design
The web chat reuses only `stt.py`, `tts.py` (Piper synthesizer), `text.clean_for_speech` and `config.py`, through `backend/voice.py` (push-to-talk from the browser, no wake word); see `backend/CLAUDE.md`. Keep those interfaces stable.

Voice is only another input/output for `ChatLoop`: transcripts go to `chat.submit()` (same path as keyboard, which keeps working); each finished turn's final answer returns via `chat.on_reply`. Tools, memory, checkpointer and thread are shared. Streamed sentences come from `turns.SpeechStream` (see root CLAUDE.md); the assistant queues them and speaks in bursts from `PROCESSING`, so the wake word is still checked between bursts.

`assistant.py` is the state machine (`VoiceState`, explicit `TRANSITIONS`). All audio runs on its one thread, except the PortAudio callback and a TTS synthesis helper thread. Components are injected, so tests use fakes: `wakeword.py` (openWakeWord ONNX behind `WakeWordDetector`: consecutive-frame hits + cooldown), `vad.py` (Silero VAD + `CommandRecorder`), `stt.py` (faster-whisper, only after wake word), `tts.py` (Piper; `Speaker` synthesizes the next sentence during playback), `audio.py` (sounddevice), `text.py` (wake-word stripping, filler/yes-no, markdown cleanup), `dictation.py` (cleanup of dictated prompts, hands-free phrases), `hotkey.py` (push-to-talk key), `config.py` (`VoiceConfig.from_env`, validated).

## Behavior to preserve
- Wake word is monitored while the agent works ("Jarvis, stop" cancels a turn) but never while speaking.
- Mic queue flushed and detector reset after every playback (avoids self-activation).
- Recorder keeps ~240 ms pre-roll so "Jarvis, do X" in one breath works; wake word alone → beep and listen again.
- Tool approvals are spoken and answered by voice; silence or unclear answer declines.
- Pretrained openWakeWord model is `hey_jarvis` (no plain "jarvis"); a custom `.onnx` path works in `WAKE_WORD`.
- Models cached in `~/.cache/langchain-voice` (override `VOICE_MODEL_DIR`).

## Push-to-talk and hands-free (`hotkey.py`, `dictation.py`)
- `VOICE_TRIGGER`: `both` (default), `push_to_talk` (the wake-word model is not loaded; `detector` is None) or `wake_word` (`key` is None). `build_key()` falls back to the wake word if the key cannot be read (not Windows).
- `KeyWatcher` polls `PUSH_TO_TALK_KEY` (default `right_ctrl`, see `hotkey.KEYS`) with `GetAsyncKeyState` (stdlib ctypes; global, any window has focus) on a daemon thread; `PushToTalkKey` turns samples into `down` / `hold` / `tap` (< 0.3 s) / `double_tap` (second tap within 0.4 s).
- Hold: `CommandRecorder.record_held` records until release (pauses kept, no silence cut-off) → transcribe → `dictated_prompt` (cleaned; `stop`/`exit`/`quit` kept plain for ChatLoop) → `chat.submit`. A tap records nothing.
- Hands-free (double tap, or saying "hands free" / "rảnh tay"): `LISTENING_FOR_COMMAND` loops on VAD `record(give_up=...)`; `give_up` stops the wait (never mid-utterance) to speak queued replies or an approval question, or to follow the key. Each prompt is submitted and listening resumes at once (ChatLoop queues prompts behind the running turn). Ends on a tap, "stop listening" / "dừng nghe", or `HANDS_FREE_IDLE_TIMEOUT` (120 s) without a prompt. Approvals are still answered by voice (`normalize_answer`).
- Pressing the key while the assistant speaks cuts the speech (`_on_key_event`, on the watcher thread).
- Only dictated prompts (hold, hands-free) are cleaned; wake-word prompts go through unchanged.
- `dictation.clean_dictation` is rule-based (an LLM pass would cost seconds at ~4 tok/s): fillers (um/uh/ờ/ừm; not "à"/"ừ"), stutters (not "that that"/"had had", not in Vietnamese where reduplication means something), corrections only after a pause mark (", no wait, 3 pm" replaces from a repeated word, else as many words, max 3, else keeps both), "scratch that" / "xóa cái đó" drops the sentence before, capital + final period.

## Speaker verification (`speaker.py`, `enroll.py`)
`voice.enroll` records fixed phrases, makes WeSpeaker embeddings (sherpa-onnx, 26 MB model downloaded on first use), saves the average + calibrated cosine threshold as `voiceprint.npz`. If that file exists (or `SPEAKER_VERIFICATION=true`) the assistant verifies the audio of the whole interaction (wake word + command, so short utterances have enough speech) after transcription finds a real command. Rejected audio is dropped (`[REJECTED]`, transcript not shown); spoken approvals are checked too; under 0.6 s of speech fails closed. Convenience lock, not authentication (replays and voice clones pass).

## Speech language
`SPEECH_LANGUAGE` is `en`, `vi`, a list (`vi,en`: Whisper detection restricted to the listed codes in `stt.py`, because forcing one language makes Whisper translate the other) or `auto`. Non-English defaults to the `small` Whisper model. `Speaker` picks the Piper voice per reply (`TTS_VOICE_VI` for Vietnamese; voices load lazily).
