import tempfile
import unittest
from pathlib import Path

from langchain_core.embeddings import Embeddings

from rag.rag import RAG


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), float(sum(map(ord, text)) % 11), 1.0]


class RagIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.docs = Path(self._tmp.name) / "docs"
        self.docs.mkdir()
        self.rag = RAG(
            persist_directory=str(Path(self._tmp.name) / "db"),
            embeddings=FakeEmbeddings(),
        )

    def count(self):
        return len(self.rag.vectorstores.get()["ids"])

    def test_documents_are_split_and_reindexing_is_idempotent(self):
        (self.docs / "a.txt").write_text("sentence about docker. " * 100)
        self.rag.add_document(str(self.docs))
        first = self.count()
        self.assertGreater(first, 1, "long file must be split into chunks")

        self.rag.add_document(str(self.docs))
        self.rag.add_document(str(self.docs))
        self.assertEqual(self.count(), first)

    def test_changed_and_removed_files_leave_no_stale_chunks(self):
        (self.docs / "a.txt").write_text("alpha " * 200)
        (self.docs / "b.txt").write_text("beta " * 200)
        self.rag.add_document(str(self.docs))

        (self.docs / "b.txt").unlink()
        (self.docs / "a.txt").write_text("gamma")
        self.rag.add_document(str(self.docs))

        self.assertEqual(self.rag.vectorstores.get()["documents"], ["gamma"])

    def test_retriever_uses_k(self):
        for i in range(8):
            (self.docs / f"{i}.txt").write_text(f"document number {i} " * 3)
        self.rag.add_document(str(self.docs))
        self.assertEqual(len(self.rag.search("document")), 5)
        self.assertEqual(len(self.rag.get_retriever({"k": 2}).invoke("document")), 2)


class CountingEmbeddings(FakeEmbeddings):
    """Records every text that gets embedded, to prove unchanged chunks are not re-embedded."""

    def __init__(self):
        self.embedded: list[str] = []

    def embed_documents(self, texts):
        self.embedded += list(texts)
        return super().embed_documents(texts)


class RagHybridTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.docs = Path(self._tmp.name) / "docs"
        self.docs.mkdir()
        self.embeddings = CountingEmbeddings()
        self.rag = RAG(persist_directory=str(Path(self._tmp.name) / "db"), embeddings=self.embeddings)

    def write(self, name, text):
        (self.docs / name).write_text(text, encoding="utf-8")

    def sync(self):
        self.rag.add_document(str(self.docs))

    # --- incremental sync ---------------------------------------------------------

    def test_unchanged_chunks_are_not_embedded_again(self):
        self.write("a.txt", "alpha sentence about docker. " * 60)
        self.sync()
        first = len(self.embeddings.embedded)
        self.assertGreater(first, 1)

        self.sync()
        self.sync()
        self.assertEqual(len(self.embeddings.embedded), first)

    def test_only_new_chunks_are_embedded_after_an_edit(self):
        self.write("a.txt", "alpha " * 300)
        self.write("b.txt", "beta " * 300)
        self.sync()
        before = len(self.embeddings.embedded)

        self.write("b.txt", "gamma")
        self.sync()
        self.assertEqual(self.embeddings.embedded[before:], ["gamma"], "a.txt must keep its stored embeddings")
        self.assertNotIn("beta", " ".join(self.rag.vectorstores.get()["documents"]))

    def test_adding_a_document_does_not_re_embed_the_others(self):
        self.write("b.txt", "bravo sentence about docker. " * 40)
        self.write("c.txt", "charlie sentence about networks. " * 40)
        self.sync()
        before = len(self.embeddings.embedded)
        self.assertGreater(before, 2)

        # Whatever order the files are loaded in, the new file's chunks must not shift the others' ids.
        for name in ["a.txt", "0.txt", "zz.txt"]:
            self.write(name, f"fresh {name} content")
            self.sync()

        self.assertEqual(len(self.embeddings.embedded) - before, 3, "only the three new files are embedded")

    def test_removing_every_document_clears_the_index(self):
        self.write("a.txt", "alpha " * 100)
        self.sync()
        (self.docs / "a.txt").unlink()
        self.sync()
        self.assertEqual(self.rag.vectorstores.get()["ids"], [])
        self.assertEqual(self.rag.search_hybrid("alpha"), [])

    def test_more_document_types_are_indexed(self):
        self.write("notes.md", "# Heading\nmarkdown about kubernetes pods")
        self.write("guide.rst", "Title\n=====\nrestructured text about helm charts")
        self.write("ignored.bin", "binary-ish about terraform")
        self.sync()
        text = " ".join(self.rag.vectorstores.get()["documents"])
        self.assertIn("kubernetes", text)
        self.assertIn("helm charts", text)
        self.assertNotIn("terraform", text)

    # --- hybrid search ------------------------------------------------------------

    def fill(self):
        for i in range(6):
            self.write(f"filler{i}.txt", f"document number {i} talks about general topics and nothing special.")

    def fill_more(self):
        for i in range(10):
            self.write(f"other{i}.txt", f"unrelated note {i} about gardening and weather.")

    def test_an_exact_identifier_is_found_even_though_embeddings_cannot_tell(self):
        self.fill()
        self.write("invoice.txt", "The quarterly invoice number is INV-48213 for customer Zeta.")
        self.sync()
        hits = self.rag.search_hybrid("INV-48213", k=3)
        self.assertEqual(Path(hits[0].source).name, "invoice.txt")
        self.assertEqual(hits[0].score, 1.0)
        self.assertTrue(all(0 < h.score <= 1.0 for h in hits))

    def test_one_file_cannot_fill_the_results(self):
        self.fill()  # unrelated files, so that "docker" can tell passages apart
        self.fill_more()
        self.write("big.txt", "\n\n".join(f"docker paragraph {i}: " + "docker volumes and networks. " * 12 for i in range(12)))
        self.write("small.txt", "docker is mentioned once here.")
        self.sync()
        hits = self.rag.search_hybrid("docker", k=5)
        sources = [Path(h.source).name for h in hits]
        self.assertLessEqual(sources.count("big.txt"), 2)
        self.assertIn("small.txt", sources)

    def test_hits_carry_the_position_in_the_file(self):
        self.write("a.txt", "x" * 600 + " needle here " + "y" * 600)
        self.sync()
        [hit] = self.rag.search_hybrid("needle", k=1)
        self.assertIsInstance(hit.start, int)

    def test_new_documents_are_searchable_after_a_resync(self):
        self.write("a.txt", "first document about apples")
        self.sync()
        self.assertEqual(self.rag.search_hybrid("zebra"), [])
        self.write("b.txt", "second document about zebra crossings")
        self.sync()
        self.assertEqual(Path(self.rag.search_hybrid("zebra")[0].source).name, "b.txt")

    def test_empty_queries_and_empty_indexes_are_safe(self):
        self.assertEqual(self.rag.search_hybrid("anything"), [])
        self.write("a.txt", "apples")
        self.sync()
        self.assertEqual(self.rag.search_hybrid("   "), [])

    # --- the tool -----------------------------------------------------------------

    def test_the_tool_names_the_file_and_relevance(self):
        self.fill()
        self.write("invoice.txt", "The quarterly invoice number is INV-48213 for customer Zeta.")
        self.sync()
        output = self.rag.get_search_document_tools().invoke({"query": "INV-48213", "top_k": 2})
        self.assertTrue(output.startswith("[1] invoice.txt (relevance 1.00)"))
        self.assertIn("INV-48213", output)
        self.assertEqual(output.count("\n\n[") + 1, 2)

    def test_the_tool_says_when_nothing_matches(self):
        tool = self.rag.get_search_document_tools()
        self.assertEqual(tool.invoke({"query": "x"}), "No relevant passages found in the knowledge documents.")

    def test_the_tool_limits_how_many_passages_come_back(self):
        for i in range(15):
            self.write(f"d{i}.txt", f"shared keyword passage number {i}")

        self.fill()
        self.fill_more()
        for i in range(10):
            self.write(f"pad{i}.txt", f"padding passage {i} about something else entirely")

        self.sync()
        tool = self.rag.get_search_document_tools()
        self.assertEqual(tool.invoke({"query": "shared keyword", "top_k": 999}).count("(relevance"), 10)
        self.assertEqual(tool.invoke({"query": "shared keyword", "top_k": 0}).count("(relevance"), 1)


if __name__ == "__main__":
    unittest.main()
