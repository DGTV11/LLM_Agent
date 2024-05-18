from os import path

import ollama
from langchain.community.llms.ollama import Ollama as LCOllama

from config import CONFIG

HOST_URL = CONFIG['server_url']
OLLAMA_HOST = ollama.Client(HOST_URL)

LLM_NAME = CONFIG['llm_name']
EMBEDDING_NAME = CONFIG['embedding_name']

LLM_CTX_WINDOW = CONFIG['ctx_window']

LCOLLAMA_MODEL = LCOllama(base_url=HOST_URL, model=LLM_NAME)
