from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from data.output import CodeAnalysis, Person
from rag.rag import RAG

QUERY = "Who is Faker?"
rag = RAG()
retriever = rag.get_retriever({"k": 5, "filter": {}})


def format_docs(docs):
    return "\n\n".join(
        f"""
    Source: {d.metadata.get("source")}
    Page: {d.metadata.get("page")}

            {d.page_content}
    """
        for d in docs
    )


prompts = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful assistant"),
        (
            "user",
            """Using the following context to answer the question.

            Context: {context}

            Question: {question}
        """,
        ),
    ]
)


model = ChatOllama(model="qwen2.5-coder:7b")
structured_model = model.with_structured_output(CodeAnalysis)

parser = StrOutputParser()

chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompts
    | structured_model
)

print(chain.invoke(QUERY))
