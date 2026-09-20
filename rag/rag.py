from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core import document_loaders
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAG:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        self.embeddings = HuggingFaceEmbeddings(model_name="Baai/bge-small-en-v1.5")

        self.vectorstores = Chroma(
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
            create_collection_if_not_exists=True,
        )

        self.retriever = self.vectorstores.as_retriever(kwargs={"k": 5})

    def add_document(self, dir):
        py_loader = DirectoryLoader(dir, glob="**/*.py", loader_cls=TextLoader)
        txt_loader = DirectoryLoader(dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = py_loader.load() + txt_loader.load()

        if not documents:
            print("document not found: ", dir)
        else:
            self.vectorstores.add_documents(documents)

    def query(self, query):
        return self.retriever.invoke(query)

    def get_retriever(self, parameter):
        return self.vectorstores.as_retriever(kwargs=parameter)
