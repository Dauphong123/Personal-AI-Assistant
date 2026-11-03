from dependency import db_init, model_init, rag_init, tokenizer_init


class AgentApplication:
    def __init__(self):
        self.model = model_init()
        self.rag = rag_init()
        self.tokenizer = tokenizer_init()
        self.memory = db_init()

    def run(self, query: str, thread_id: str = "1"):
        config = {"configurable": {"thread_id": thread_id}}
