"""Dictation cleanup (F022): what Whisper heard becomes the prompt you meant, as Wispr Flow does."""

import unittest

from voice.dictation import clean_dictation


class FillerTests(unittest.TestCase):
    def test_english_fillers_are_removed(self):
        self.assertEqual(clean_dictation("um, can you, uh, open the readme"), "Can you open the readme.")

    def test_vietnamese_hesitations_are_removed(self):
        self.assertEqual(clean_dictation("ờ, mở file readme ừm giúp tôi"), "Mở file readme giúp tôi.")

    def test_meaningful_vietnamese_particles_stay(self):
        # "à" asks a question and "ừ" means yes; neither is a hesitation here.
        self.assertEqual(clean_dictation("xong rồi à"), "Xong rồi à.")
        self.assertEqual(clean_dictation("ừ làm đi"), "Ừ làm đi.")

    def test_words_containing_filler_letters_stay(self):
        self.assertEqual(clean_dictation("we were humming"), "We were humming.")

    def test_only_fillers_give_nothing(self):
        self.assertEqual(clean_dictation("um, uh..."), "")


class StutterTests(unittest.TestCase):
    def test_repeated_words_are_collapsed(self):
        self.assertEqual(clean_dictation("open the the file"), "Open the file.")
        self.assertEqual(clean_dictation("I I I think so"), "I think so.")

    def test_grammatical_repeats_stay(self):
        self.assertEqual(clean_dictation("she said that that was fine"), "She said that that was fine.")

    def test_vietnamese_reduplication_stays(self):
        self.assertEqual(clean_dictation("đi từ từ thôi"), "Đi từ từ thôi.")

    def test_repeated_numbers_stay(self):
        self.assertEqual(clean_dictation("dial 1 1 2"), "Dial 1 1 2.")


class CorrectionTests(unittest.TestCase):
    def test_short_correction_replaces_the_words_it_corrects(self):
        self.assertEqual(clean_dictation("meet at 2 pm, no wait, 3 pm"), "Meet at 3 pm.")

    def test_correction_restarting_from_a_repeated_word(self):
        self.assertEqual(
            clean_dictation("open the file, I mean the config file in src"), "Open the config file in src."
        )
        self.assertEqual(clean_dictation("I want pizza, no wait, I want a burger"), "I want a burger.")

    def test_vietnamese_correction(self):
        self.assertEqual(clean_dictation("hẹn lúc 2 giờ, à không, 3 giờ"), "Hẹn lúc 3 giờ.")

    def test_scratch_that_drops_the_previous_sentence(self):
        self.assertEqual(clean_dictation("Open the readme. Scratch that. Open the license."), "Open the license.")
        self.assertEqual(clean_dictation("open the readme, scratch that, open the license"), "Open the license.")

    def test_scratch_that_alone_gives_nothing(self):
        self.assertEqual(clean_dictation("scratch that"), "")

    def test_long_unanchored_correction_keeps_both_parts(self):
        # Nothing tells what is being corrected, so nothing is thrown away.
        self.assertEqual(
            clean_dictation("list the files, no wait, show me what changed since yesterday"),
            "List the files, show me what changed since yesterday.",
        )

    def test_unpunctuated_i_mean_is_ordinary_speech(self):
        self.assertEqual(clean_dictation("I mean it when I say thanks"), "I mean it when I say thanks.")

    def test_a_plain_no_is_not_a_correction(self):
        self.assertEqual(clean_dictation("no, that is wrong"), "No, that is wrong.")


class FormattingTests(unittest.TestCase):
    def test_first_letter_capitalized_and_period_added(self):
        self.assertEqual(clean_dictation("  run the tests  "), "Run the tests.")

    def test_existing_end_punctuation_is_kept(self):
        self.assertEqual(clean_dictation("what changed today?"), "What changed today?")

    def test_empty_input(self):
        self.assertEqual(clean_dictation(""), "")
        self.assertEqual(clean_dictation("  ,. "), "")


if __name__ == "__main__":
    unittest.main()
