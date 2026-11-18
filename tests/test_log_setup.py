import logging
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path

from log_setup import configure_logging


class ConfigureLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger(f"test_log_setup.{self.id()}")
        self.root.propagate = False
        self.addCleanup(self.close_handlers)

    def close_handlers(self):
        for handler in list(self.root.handlers):
            handler.close()
            self.root.removeHandler(handler)

    def test_default_is_warning_on_the_console(self):
        configure_logging({}, root=self.root)
        [handler] = self.root.handlers
        self.assertIsInstance(handler, logging.StreamHandler)
        self.assertEqual(handler.level, logging.WARNING)

    def test_log_level_sets_the_console_level(self):
        configure_logging({"LOG_LEVEL": "debug"}, root=self.root)
        self.assertEqual(self.root.handlers[0].level, logging.DEBUG)
        self.assertEqual(self.root.level, logging.DEBUG)

    def test_an_unknown_level_is_refused_with_the_valid_names(self):
        with self.assertRaises(ValueError) as caught:
            configure_logging({"LOG_LEVEL": "loud"}, root=self.root)
        self.assertIn("LOG_LEVEL", str(caught.exception))
        self.assertIn("WARNING", str(caught.exception))

    def test_log_file_gets_info_with_timestamps_and_rotates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "agent.log"
            configure_logging({"LOG_FILE": str(path)}, root=self.root)
            files = [h for h in self.root.handlers if isinstance(h, RotatingFileHandler)]
            self.assertEqual(len(files), 1)
            self.assertGreater(files[0].maxBytes, 0)
            self.assertGreater(files[0].backupCount, 0)

            self.root.info("tool failed")
            self.root.debug("too detailed")
            files[0].flush()
            text = path.read_text(encoding="utf-8")
            self.close_handlers()

        self.assertIn("INFO", text)
        self.assertIn("tool failed", text)
        self.assertNotIn("too detailed", text)
        self.assertRegex(text, r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")

    def test_console_stays_quiet_when_the_file_takes_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            configure_logging({"LOG_FILE": str(Path(tmp) / "a.log")}, root=self.root)
            console = [h for h in self.root.handlers if not isinstance(h, RotatingFileHandler)]
            self.assertEqual(console[0].level, logging.WARNING)
            self.assertEqual(self.root.level, logging.INFO)
            self.close_handlers()

    def test_calling_twice_does_not_duplicate_handlers(self):
        configure_logging({}, root=self.root)
        configure_logging({}, root=self.root)
        self.assertEqual(len(self.root.handlers), 1)


if __name__ == "__main__":
    unittest.main()
