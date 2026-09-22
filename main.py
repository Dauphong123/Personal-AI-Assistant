from langchain_core.runnables import RunnableBranch

from chains import code_loop, direct_chain, rag_chain, router_chain

QUERY = "From D:/app/Langchain, look in tools.py and explain to me if there is enough tools for project analysis"


def route_question(question):
    route = router_chain.invoke({"question": question})

    return {"question": question, "route": route}


chain = route_question | RunnableBranch(
    (lambda x: x["route"].destination == "rag", rag_chain),
    (lambda x: x["route"].destination == "code", code_loop),
    direct_chain,
)

response = chain.stream(QUERY)

for chunk in response:
    print(chunk.content, end="", flush=True)
