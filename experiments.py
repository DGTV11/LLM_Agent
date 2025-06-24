import hashlib
from datetime import datetime

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

NOMIC_EMBED_TEXT_TOKENIZER = Tokenizer.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
)

client = chromadb.EphemeralClient()
ef = OllamaEmbeddingFunction(
    url=f"http://127.0.0.1:11434", model_name="nomic-embed-text"
)
collection = client.get_or_create_collection(
    name="archival_storage", embedding_function=ef
)

user_id = 1
content = "anfriiuhrfuhfe"

splitter = TextSplitter.from_huggingface_tokenizer(NOMIC_EMBED_TEXT_TOKENIZER, 8192)
chunk_list = splitter.chunks(content)

hex_stringify = lambda chunk: hashlib.md5(chunk.encode("UTF-8")).hexdigest()
ids = [hex_stringify(chunk) for chunk in chunk_list]
metadatas = [
    {
        "user_id": user_id,
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d"),
    }
    for _ in range(len(chunk_list))
]
print("archival insert debugging", chunk_list, metadatas, ids)

collection.add(documents=chunk_list, metadatas=metadatas, ids=ids)

print("success")
