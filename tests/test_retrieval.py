import unittest

from retrieval import BM25, best_passages, fold, reciprocal_rank_fusion, split_passages, tokenize


class TokenizeTests(unittest.TestCase):
    def test_accents_and_case_are_folded(self):
        self.assertEqual(fold("Tệp Đường"), "tep duong")
        self.assertEqual(tokenize("Hello, World_2!"), ["hello", "world_2"])

    def test_vietnamese_matches_with_or_without_accents(self):
        bm25 = BM25(["liệt kê các tệp trong thư mục", "thời tiết hôm nay"])
        self.assertEqual(bm25.top("tep thu muc", 2), [0])


class BM25Tests(unittest.TestCase):
    DOCS = [
        "docker volumes keep data after a container is removed",
        "a network bridge connects containers on one host",
        "postgres stores rows in tables and uses indexes",
        "the invoice number INV-48213 belongs to customer Zeta",
    ]

    def test_the_document_with_the_rare_word_ranks_first(self):
        self.assertEqual(BM25(self.DOCS).top("INV-48213", 3), [3])

    def test_documents_without_any_query_word_are_left_out(self):
        self.assertEqual(BM25(self.DOCS).top("kubernetes", 4), [])

    def test_more_matching_words_rank_higher(self):
        order = BM25(self.DOCS).top("docker container volumes", 4)
        self.assertEqual(order[0], 0)

    def test_filler_words_alone_match_nothing(self):
        docs = ["this is the story of the river", "what is the answer to the question", "cats and dogs"]
        self.assertEqual(BM25(docs).top("what is the capital of France", 3), [])

    def test_filler_words_do_not_outweigh_the_topic(self):
        docs = ["the story of the river is long", "the capital of France is Paris", "cats and dogs"]
        self.assertEqual(BM25(docs).top("what is the capital of France", 3), [1])

    def test_vietnamese_filler_is_ignored_too(self):
        docs = ["đây là một câu chuyện của và là", "tệp cấu hình nằm trong thư mục gốc"]
        self.assertEqual(BM25(docs).top("là và của", 2), [])
        self.assertEqual(BM25(docs).top("tệp ở đâu", 2), [1])

    def test_a_word_found_in_most_passages_is_ignored_in_a_larger_corpus(self):
        docs = [f"every passage says common and has its own word{i}" for i in range(10)]
        bm25 = BM25(docs)
        self.assertEqual(bm25.top("common", 10), [])
        self.assertEqual(bm25.top("common word3", 10), [3])

    def test_a_small_corpus_keeps_every_word(self):
        self.assertEqual(BM25(["common alpha", "common beta"]).top("common", 2), [0, 1])

    def test_empty_inputs_do_not_crash(self):
        self.assertEqual(BM25([]).top("anything", 3), [])
        self.assertEqual(BM25(self.DOCS).top("", 3), [])
        self.assertEqual(BM25(["", ""]).top("x", 3), [])

    def test_k_limits_the_results(self):
        self.assertEqual(len(BM25(["alpha b", "alpha c", "alpha d"]).top("alpha", 2)), 2)


class FusionTests(unittest.TestCase):
    def test_an_item_ranked_high_in_both_lists_wins(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
        self.assertEqual([item for item, _ in fused][:2], [1, 3])
        self.assertEqual({item for item, _ in fused}, {1, 2, 3, 4})

    def test_weights_decide_ties_and_scale_a_list(self):
        tied = reciprocal_rank_fusion([["vector"], ["keyword"]])
        self.assertEqual(len(tied), 2)
        weighted = reciprocal_rank_fusion([["vector"], ["keyword"]], weights=[1.0, 1.2])
        self.assertEqual(weighted[0][0], "keyword")
        self.assertGreater(weighted[0][1], weighted[1][1])

    def test_an_item_found_by_only_one_method_still_appears(self):
        fused = dict(reciprocal_rank_fusion([[], [7]]))
        self.assertIn(7, fused)

    def test_no_lists_means_nothing(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])


class PassageTests(unittest.TestCase):
    def test_passages_stay_near_the_size(self):
        text = "\n\n".join(f"Paragraph number {i} has a few words in it." for i in range(60))
        passages = split_passages(text, size=200, overlap=0)
        self.assertGreater(len(passages), 3)
        self.assertTrue(all(len(p) <= 260 for p in passages), [len(p) for p in passages])
        self.assertIn("Paragraph number 0", passages[0])
        self.assertIn("Paragraph number 59", passages[-1])

    def test_nothing_is_lost(self):
        text = "\n".join(f"line{i}" for i in range(100))
        joined = " ".join(split_passages(text, size=120, overlap=0))
        for i in range(100):
            self.assertIn(f"line{i}", joined)

    def test_a_long_unbroken_string_is_cut(self):
        passages = split_passages("x" * 2500, size=1000, overlap=0)
        self.assertGreaterEqual(len(passages), 3)
        self.assertTrue(all(len(p) <= 1000 for p in passages))

    def test_long_paragraphs_are_cut_at_sentences(self):
        text = " ".join(f"This is sentence {i}." for i in range(80))
        passages = split_passages(text, size=300, overlap=0)
        self.assertGreater(len(passages), 3)
        self.assertTrue(all(p.endswith(".") for p in passages))

    def test_empty_text(self):
        self.assertEqual(split_passages("", 100), [])
        self.assertEqual(split_passages("  \n\n  ", 100), [])


class BestPassagesTests(unittest.TestCase):
    TEXT = "\n\n".join(
        [f"Filler paragraph {i} about nothing in particular, padding the page out." for i in range(10)]
        + ["The secret setting is return_exceptions and it changes error handling."]
        + [f"More filler {i}, still nothing relevant to the question here." for i in range(10)]
    )

    def test_the_matching_passage_is_returned_not_the_start(self):
        result = best_passages(self.TEXT, "return_exceptions", max_chars=300, size=200)
        self.assertIn("return_exceptions", result)
        self.assertNotIn("Filler paragraph 0", result)

    def test_passages_keep_their_order_and_gaps_are_marked(self):
        filler = "\n\n".join(f"{'filler ' * 8}{i}" for i in range(3))  # each stays its own passage
        text = f"alpha one is here.\n\n{filler}\n\nalpha two is here."
        result = best_passages(text, "alpha", max_chars=1000, size=60)
        self.assertLess(result.index("alpha one"), result.index("alpha two"))
        self.assertIn("[...]", result)

    def test_no_match_falls_back_to_the_start(self):
        result = best_passages(self.TEXT, "zzzz", max_chars=300, size=200)
        self.assertIn("Filler paragraph 0", result)

    def test_the_budget_is_respected(self):
        text = "\n\n".join(f"word{i} appears here with some padding text." for i in range(100))
        result = best_passages(text, "padding", max_chars=500, size=100)
        self.assertLess(len(result), 700)

    def test_empty_text(self):
        self.assertEqual(best_passages("", "x", 100), "")


if __name__ == "__main__":
    unittest.main()
