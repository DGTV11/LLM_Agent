from llm_os import memory_mod, host, ailang_executor

# Inference
def infer_without_stream(input: str) -> str:
    return host.OLLAMA_HOST.generate(
        model=host.LLM_NAME,
        prompt=input,
        stream=False
    )['response']

def infer_with_stream(input: str):
    stream = host.OLLAMA_HOST.generate(
        model=host.LLM_NAME,
        prompt=input,
        stream=True
    )

    for chunk in stream:
        yield chunk['response']

# Base event
def event(memory: memory.Memory, message: str) -> None:
    memory.main_context.fifo_queue.put(message)

    res = infer_without_stream(memory.main_context.main_context_str)

    memory.recall_storage.append(memory_mod.RecallStorageDatum(datetime.now().date(), message))
    memory.recall_storage.append(memory_mod.RecallStorageDatum(datetime.now().date(), res))
