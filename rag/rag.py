import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval import BM25, reciprocal_rank_fusion

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DOCUMENT_TYPES = ("py", "txt", "md", "markdown", "rst")
MAX_RESULTS = 10
# Vector distance above which a passage does not count as related. Calibrated with
# bge-small-en-v1.5 on the sample documents: related queries had their best match at
# 0.23 to 0.70 (even a Vietnamese one), unrelated queries at 1.06 or more. Re-check it
# if the embedding model changes.
MAX_VECTOR_DISTANCE = 0.9
KEYWORD_WEIGHT = 1.2


@dataclass
class Hit:
    text: str
    source: str
    start: int | None  # character offset of the passage in its file
    score: float  # relevance, 1.0 for the best hit of the query


class RAG:
    def __init__(self, persist_directory: str = "./chroma_db", embeddings=None):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, add_start_index=True
        )

        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        self.vectorstores = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            create_collection_if_not_exists=True,
        )

        self.retriever = self.get_retriever({"k": 5})
        self._records: list[tuple[str, str, dict]] | None = None  # (id, text, metadata)
        self._bm25: BM25 | None = None
        self._position: dict[tuple, int] = {}

    def add_document(self, dir):
        """Make the vector store mirror `dir`: split files into chunks, embed only the chunks
        that are new, and drop chunks of files that changed or disappeared.

        Safe to run on every startup: unchanged chunks keep their stored embeddings (the
        slow part), and nothing is duplicated."""
        documents = []

        for extension in DOCUMENT_TYPES:
            loader = DirectoryLoader(dir, glob=f"**/*.{extension}", loader_cls=TextLoader)
            documents += loader.load()

        if not documents:
            print("document not found: ", dir)

        chunks = self.splitter.split_documents(documents) if documents else []

        # The id depends on the file and the chunk's place *in that file*, never on other files:
        # adding a document must not change the ids (and force re-embedding) of the others.
        ids = []
        position_in_file: dict[str, int] = {}

        for chunk in chunks:
            source = str(chunk.metadata.get("source"))
            index = position_in_file.get(source, 0)
            position_in_file[source] = index + 1
            key = f"{source}|{index}|{chunk.page_content}"
            ids.append(hashlib.sha256(key.encode("utf-8")).hexdigest())

        existing = set(self.vectorstores.get()["ids"])
        stale = existing - set(ids)

        if stale:
            self.vectorstores.delete(ids=list(stale))

        new = [(chunk, id) for chunk, id in zip(chunks, ids) if id not in existing]

        if new:
            self.vectorstores.add_documents([c for c, _ in new], ids=[i for _, i in new])

        self._records = None  # the keyword index is rebuilt on the next search

    def search(self, query):
        return self.retriever.invoke(query)

    def get_retriever(self, parameter):
        return self.vectorstores.as_retriever(search_kwargs=parameter)

    def _index(self) -> BM25:
        if self._bm25 is None or self._records is None:
            data = self.vectorstores.get()
            self._records = list(zip(data["ids"], data["documents"], data["metadatas"]))
            self._bm25 = BM25([text for _, text, _ in self._records])
            self._position = {}

            for position, (_, text, metadata) in enumerate(self._records):
                self._position.setdefault((metadata.get("source"), text), position)

        return self._bm25

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        per_source: int = 2,
        candidates: int = 20,
        max_distance: float = MAX_VECTOR_DISTANCE,
    ) -> list[Hit]:
        """Best passages by meaning (vector) and by words (BM25), fused, with at most
        `per_source` passages from any one file so a single document cannot fill the answer.

        Nothing is returned for a query that matches neither by words nor closely enough by
        meaning, rather than the least-bad passages."""
        query = query.strip()

        if not query:
            return []

        bm25 = self._index()

        if not self._records:
            return []

        by_meaning = []

        for document, distance in self.vectorstores.similarity_search_with_score(query, k=candidates):
            if distance > max_distance:
                continue

            position = self._position.get((document.metadata.get("source"), document.page_content))

            if position is not None:
                by_meaning.append(position)

        by_words = bm25.top(query, candidates)
        # An exact word match (an identifier, an error message) is a precise signal; vector
        # similarity is a fuzzy one. The slightly higher weight lets the exact match win a tie.
        fused = reciprocal_rank_fusion([by_meaning, by_words], weights=[1.0, KEYWORD_WEIGHT])

        if not fused:
            return []

        best = fused[0][1]
        taken: dict[str, int] = {}
        hits = []

        for position, score in fused:
            _, text, metadata = self._records[position]
            source = str(metadata.get("source", ""))

            if taken.get(source, 0) >= per_source:
                continue

            taken[source] = taken.get(source, 0) + 1
            hits.append(Hit(text, source, metadata.get("start_index"), score / best))

            if len(hits) >= k:
                break

        return hits

    def get_search_document_tools(self) -> BaseTool:
        @tool
        def search_document(query: str, top_k: int = 5) -> str:
            """Search the user's indexed knowledge documents. Combines meaning and exact-word
            search and returns the best passages with their source file."""
            hits = self.search_hybrid(query, k=max(1, min(int(top_k), MAX_RESULTS)))

            if not hits:
                return "No relevant passages found in the knowledge documents."

            return "\n\n".join(
                f"[{number}] {Path(hit.source).name} (relevance {hit.score:.2f})\n{hit.text}"
                for number, hit in enumerate(hits, start=1)
            )

        return search_document
