"""Push-to-talk key (F022): hold, tap, double tap; the watcher thread; the trigger settings."""

import threading
import time
import unittest

from voice.config import VoiceConfig, VoiceConfigError
from voice.hotkey import KEYS, KeyWatcher, PushToTalkKey


class PushToTalkKeyTests(unittest.TestCase):
    def feed(self, key, steps):
        return [event for down, now in steps if (event := key.update(down, now))]

    def test_hold_reports_down_then_hold(self):
        key = PushToTalkKey()
        self.assertEqual(self.feed(key, [(True, 0.0), (True, 0.3), (False, 0.8)]), ["down", "hold"])

    def test_short_press_is_a_tap(self):
        self.assertEqual(self.feed(PushToTalkKey(), [(True, 0.0), (False, 0.1)]), ["down", "tap"])

    def test_two_quick_taps_are_a_double_tap(self):
        events = self.feed(PushToTalkKey(), [(True, 0.0), (False, 0.1), (True, 0.25), (False, 0.35)])
        self.assertEqual(events, ["down", "tap", "down", "double_tap"])

    def test_slow_taps_stay_single(self):
        events = self.feed(PushToTalkKey(), [(True, 0.0), (False, 0.1), (True, 2.0), (False, 2.1)])
        self.assertEqual(events, ["down", "tap", "down", "tap"])

    def test_a_third_tap_starts_over(self):
        steps = [(True, 0.0), (False, 0.1), (True, 0.2), (False, 0.3), (True, 0.4), (False, 0.5)]
        self.assertEqual(self.feed(PushToTalkKey(), steps)[-1], "tap")

    def test_a_hold_between_taps_breaks_the_double_tap(self):
        steps = [(True, 0.0), (False, 0.1), (True, 0.2), (False, 0.9)]
        self.assertEqual(self.feed(PushToTalkKey(), steps)[-1], "hold")

    def test_no_change_means_no_event(self):
        self.assertEqual(self.feed(PushToTalkKey(), [(False, 0.0), (False, 0.1)]), [])


class KeyWatcherTests(unittest.TestCase):
    def test_events_reach_the_queue_and_the_callback(self):
        states = iter([False, True, True, False] + [False] * 1000)
        seen = []
        done = threading.Event()

        def on_event(event):
            seen.append(event)

            if event == "tap":
                done.set()

        watcher = KeyWatcher(lambda: next(states), on_event=on_event, interval=0.001)
        watcher.start()

        try:
            self.assertTrue(done.wait(2))
        finally:
            watcher.close()

        self.assertEqual(seen, ["down", "tap"])
        self.assertEqual([watcher.poll(), watcher.poll(), watcher.poll()], ["down", "tap", None])

    def test_held_follows_the_key(self):
        down = threading.Event()
        watcher = KeyWatcher(down.is_set, interval=0.001)
        watcher.start()

        try:
            down.set()
            deadline = time.monotonic() + 2

            while not watcher.held and time.monotonic() < deadline:
                time.sleep(0.005)

            self.assertTrue(watcher.held)
        finally:
            watcher.close()

    def test_a_failing_key_reader_stops_quietly(self):
        def broken():
            raise OSError("no keyboard")

        watcher = KeyWatcher(broken, interval=0.001)
        watcher.start()
        watcher.close()
        self.assertIsNone(watcher.poll())

    def test_known_key_names(self):
        self.assertIn("right_ctrl", KEYS)
        self.assertEqual(KEYS["right_ctrl"], 0xA3)


class TriggerConfigTests(unittest.TestCase):
    def test_defaults_use_both_triggers(self):
        config = VoiceConfig.from_env({})
        self.assertEqual(config.voice_trigger, "both")
        self.assertTrue(config.uses_wake_word)
        self.assertTrue(config.uses_push_to_talk)
        self.assertEqual(config.push_to_talk_key, "right_ctrl")
        self.assertEqual(config.hands_free_idle_timeout, 120.0)

    def test_push_to_talk_only(self):
        config = VoiceConfig.from_env({"VOICE_TRIGGER": "push_to_talk", "PUSH_TO_TALK_KEY": "F9"})
        self.assertFalse(config.uses_wake_word)
        self.assertTrue(config.uses_push_to_talk)
        self.assertEqual(config.push_to_talk_key, "f9")

    def test_wake_word_only(self):
        config = VoiceConfig.from_env({"VOICE_TRIGGER": "wake_word"})
        self.assertTrue(config.uses_wake_word)
        self.assertFalse(config.uses_push_to_talk)

    def test_bad_values_are_rejected(self):
        for env in (
            {"VOICE_TRIGGER": "clap"},
            {"PUSH_TO_TALK_KEY": "space_bar"},
            {"HANDS_FREE_IDLE_TIMEOUT": "1"},
        ):
            with self.subTest(env=env), self.assertRaises(VoiceConfigError):
                VoiceConfig.from_env(env)


if __name__ == "__main__":
    unittest.main()
