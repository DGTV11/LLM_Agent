from os import path
import hashlib

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

from host import HOST_URL
from llm_os.tokenisers import NOMIC_EMBED_TEXT_TOKENIZER

class ArchivalStorage:
    def __init__(self, top_k=100):
        self.top_k = top_k

        self.client = chromadb.PersistentClient(path=path.join(path.dirname(__file__), "persistent_storage")):
        self.ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url=f"{HOST_URL}/api/embeddings",
        )
        self.collection = self.client.get_or_create_collection(name="archival_storage", embedding_function=ef)

        self.cache = {}

    def __len__(self):
        return self.collection.count()

    def insert(content: str, return_ids: bool = False):
        try:
            splitter = TextSplitter.from_huggingface_tokenizer(NOMIC_EMBED_TEXT_TOKENIZER, 8192)
            chunks = splitter.chunks(content)

            hex_stringify = lambda chunk: hashlib.md5(chunk.encode("UTF-8")).hexdigest()
            ids = [hex_stringify(chunk) for chunk in chunks]
            self.collection.add(documents=[chunks], ids=ids)

            self.cache = {}
            
            if return_ids:
                return ids
            else:
                return True
        except Exception as e:
            print('Archival insert error', e)
            raise e

    def search(query: str, count: str, start: str):
        try:
            if query not in self.cache:
                self.cache[query] = self.collection.query(query_texts=[query], n_results=self.top_k)['documents'][0]

            start = int(start if start else 0)
            count = int(count if count else self.top_k)
            end = min(count+start, len(self.cache[query]))

            local_time = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z%z")}.strip()
            results = [{"timestamp": local_time(), "content": document} for document in self.cache[query][start:end]]

            return results, len(results)
        except Exception as e:
            print('Archival search error', e)
            raise e
