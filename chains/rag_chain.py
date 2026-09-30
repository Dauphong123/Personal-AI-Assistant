from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from data import RagOutput
from dependency import dependency
from rag import RAG


def format_docs(docs):
    return "\n\n".join(
        f"""
        Source: {d.metadata.get("source")}
        Page: {d.metadata.get("page")}

                {d.page_content}
        """
        for d in docs
    )


rag = RAG()
retriever = rag.get_retriever({"k": 5, "filter": {}})


rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        (
            "user",
            """Using the following context to answer the question.

                Context: {context}

                Question: {question}
            """,
        ),
    ]
)

model = dependency["model"]
structured_model = model.with_structured_output(RagOutput)

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | structured_model
)
