from pathlib import Path
from os import path
from dataclasses import dataclass
from collections import deque
import json
import hashlib

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from semantic_text_splitter import TextSplitter

from host import HOST_URL
from llm_os.tokenisers import (
    NOMIC_EMBED_TEXT_TOKENIZER,
    get_tokeniser_and_context_window,
)
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.file_storage import FileStorage


class Memory:
    def __init__(
        self,
        model_name: str,
        conv_name: str,
        in_context_function_dats: dict,
        out_of_context_function_dats: dict,
        system_instructions: str,
        working_context: WorkingContext,
        archival_storage: ArchivalStorage,
        recall_storage: RecallStorage,
        file_storage: FileStorage,
        function_schema_search_top_k: int = 10,
    ):
        self.fq_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "persistent_storage",
            conv_name,
            "fifo_queue.json",
        )

        # Function data
        self.in_context_function_dats = in_context_function_dats
        self.out_of_context_function_dats = out_of_context_function_dats
        self.function_schema_search_top_k = function_schema_search_top_k

        # Function description embeddings
        self.client = chromadb.EphemeralClient()
        self.ef = OllamaEmbeddingFunction(
            model_name="nomic-embed-text",
            url=f"{HOST_URL}/api/embed",
        )
        self.collection = self.client.get_or_create_collection(
            name="out_of_context_functions", embedding_function=self.ef
        )

        # Main context
        self.system_instructions = system_instructions
        self.working_context = working_context
        if path.exists(self.fq_path):
            self.__save_fq_path_dat_to_fq()
        else:
            self.fifo_queue = deque()
            self.total_no_messages = 0
            self.no_messages_in_queue = 0
            self.write_fq_to_fq_path()

        # External context
        self.archival_storage = archival_storage
        self.recall_storage = recall_storage
        self.file_storage = file_storage

        self.tokenizer, self.ctx_window, self.num_token_func, self.ct_num_token_func = (
            get_tokeniser_and_context_window(model_name)
        )

    def populate_function_description_embeddings(self):
        splitter = TextSplitter.from_huggingface_tokenizer(
            NOMIC_EMBED_TEXT_TOKENIZER, 8192
        )

        for dat in self.out_of_context_function_dats.values():
            content = dat["json_schema"]["description"]
            chunk_list = splitter.chunks(content)

            hex_stringify = lambda chunk: hashlib.md5(chunk.encode("UTF-8")).hexdigest()
            ids = [hex_stringify(chunk) for chunk in chunk_list]
            metadatas = [
                {"function_schema": dat["json_schema"]} for _ in range(len(chunk_list))
            ]
            self.collection.add(documents=chunk_list, metadatas=metadatas, ids=ids)

    def search_function_description_embeddings(self, query, count, start):
        try:
            query_res = self.collection.query(
                query_texts=[query], n_results=self.function_schema_search_top_k
            )
            documents = query_res["documents"]
            metadatas = query_res["metadatas"]

            start = int(start) if start else 0
            count = int(count) if count else self.function_schema_search_top_k
            end = min(count + start, len(documents))

            results = list(
                dict.fromkeys([metadata["function_schema"] for metadata in metadatas])
            )[start:end]

            return results, len(results)
        except Exception as e:
            print("Archival search error", e)
            raise e

    def write_fq_to_fq_path(self):
        plf = Path(self.fq_path)
        plf.touch(exist_ok=True)
        with open(self.fq_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "fifo_queue": list(self.fifo_queue),
                        "total_no_messages": self.total_no_messages,
                        "no_messages_in_queue": self.no_messages_in_queue,
                    }
                )
            )

    def __save_fq_path_dat_to_fq(self):
        with open(self.fq_path, "r") as f:
            fq_info = json.loads(f.read())
            self.fifo_queue = deque(fq_info["fifo_queue"])
            self.total_no_messages = fq_info["total_no_messages"]
            self.no_messages_in_queue = fq_info["no_messages_in_queue"]

    def append_messaged_to_fq_and_rs(self, messaged):
        # note: messaged must be in the form {'type': type, 'user_id': user_id, 'message': {'role': role, 'content': content}}
        self.fifo_queue.append(messaged)
        self.recall_storage.insert(messaged)
        self.total_no_messages += 1
        self.no_messages_in_queue += 1
        self.write_fq_to_fq_path()

    @property
    def main_context_system_message(self):
        newline = "\n"
        return f"""# SYSTEM INSTRUCTIONS
        {self.system_instructions}
        # IN-CONTEXT FUNCTION JSON SCHEMAS
        {newline.join([str(dat["json_schema"]) for dat in self.in_context_function_dats.values()])}
        # EXTERNAL CONTEXT INFORMATION
        {len(self.recall_storage)} previous messages between you and the user are stored in recall storage (use functions to access them)
        {len(self.archival_storage)} total memories you created are stored in archival storage (use functions to access them)
        # CORE MEMORY (limited in size, additional information stored in archival/recall storage)
        {str(self.working_context)}"""

    @property
    def main_ctx_message_seq(self):
        # note: messaged must be in the form {'type': type, 'user_id': user_id, 'message': {'role': role, 'content': content}}
        translated_messages = []
        user_role_buf = []

        for messaged in self.fifo_queue:
            if messaged["type"] == "system":
                user_role_buf.append(
                    f"❮SYSTEM MESSAGE❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "tool":
                user_role_buf.append(
                    f"❮TOOL MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "user":
                user_role_buf.append(
                    f"❮USER MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                )
            else:
                translated_messages.append(
                    {"role": "user", "content": "\n\n".join(user_role_buf)}
                )
                translated_messages.append(messaged["message"])
                user_role_buf = []

        if user_role_buf:
            translated_messages.append(
                {"role": "user", "content": "\n\n".join(user_role_buf)}
            )

        return [
            {"role": "system", "content": self.main_context_system_message}
        ] + translated_messages

    @property
    def main_ctx_message_seq_no_tokens(self):
        return self.ct_num_token_func(self.main_ctx_message_seq)
