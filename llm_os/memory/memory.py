from pathlib import Path
from os import path
from dataclasses import dataclass
from collections import deque
import json

from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.working_context import WorkingContext


class Memory:
    def __init__(
        self,
        model_name: str,
        conv_name: str,
        function_dats: dict,
        system_instructions: str,
        working_context: WorkingContext,
        archival_storage: ArchivalStorage,
        recall_storage: RecallStorage,
    ):
        self.fq_path = path.join(
            path.dirname(path.dirname(path.dirname(__file__))),
            "persistent_storage",
            conv_name,
            "fifo_queue.json",
        )

        # Function data
        self.function_dats = function_dats

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

        self.tokenizer, self.ctx_window, self.num_token_func, self.ct_num_token_func = (
            get_tokeniser_and_context_window(model_name)
        )

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
        # FUNCTION JSON SCHEMAS
        {newline.join([str(dat["json_schema"]) for dat in self.function_dats.values()])}
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
