import hashlib
from datetime import datetime
from os import path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from host import HOST_URL
from llm_os.tokenisers import NOMIC_EMBED_TEXT_TOKENIZER
from semantic_text_splitter import TextSplitter


class ArchivalStorage:
    def __init__(self, conv_name, top_k=100):
        self.top_k = top_k

        self.client = chromadb.PersistentClient(
            path=path.join(
                path.dirname(path.dirname(path.dirname(__file__))),
                "persistent_storage",
                conv_name,
            )
        )
        self.ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url=f"{HOST_URL}/api/embed",
        )
        self.collection = self.client.get_or_create_collection(
            name="archival_storage", embedding_function=self.ef
        )

    def __len__(self):
        return self.collection.count()

    def insert(self, user_id: int, content: str, return_ids: bool = False):
        try:
            splitter = TextSplitter.from_huggingface_tokenizer(
                NOMIC_EMBED_TEXT_TOKENIZER, 8192
            )
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

            print(
                "lengths →",
                len(chunk_list),
                "chunks;",
                len(ids),
                "ids;",
                len(metadatas),
                "metadatas",
            )

            self.collection.add(
                documents=chunk_list, metadatas=metadatas, ids=ids
            )  # *BUG

            if return_ids:
                return ids
            else:
                return True
        except Exception as e:
            print("Archival insert error", e)
            raise e

    def search(self, query: str, user_id: int, count: str, start: str):
        try:
            query_res = self.collection.query(
                query_texts=[query], n_results=self.top_k, where={"user_id": user_id}
            )
            documents = query_res["documents"]
            metadatas = query_res["metadatas"]

            start = int(start) if start else 0
            count = int(count) if count else self.top_k
            end = min(count + start, len(documents))

            results = [
                {"timestamp": metadata["timestamp"], "content": document}
                for metadata, document in zip(
                    metadatas[start:end], documents[start:end]
                )
            ]

            return results, len(documents)
        except Exception as e:
            print("Archival search error", e)
            raise e
