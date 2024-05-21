from dataclasses import dataclass
from queue import Queue

from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.archival_storage import ArchivalStorage
from llm_os.recall_storage import RecallStorage
from llm_os.working_context import WorkingContext

class MainContext:
    def __init__(
            self,       
            model_name: str, 
            system_instructions: str, working_context: WorkingContext, fifo_queue: Queue[str], 
            archival_storage: ArchivalStorage, recall_storage: RecallStorage
        ):
        self.system_instructions = system_instructions
        self.working_context = working_context
        self.fifo_queue = fifo_queue

        self.archival_storage = archival_storage
        self.recall_storage = recall_storage

        self.tokenizer, self.ctx_window, self.num_token_func, self.ct_num_token_func = get_tokeniser_and_context_window(model_name)
    
    @property
    def main_context_system_message(self):
        return f'''# SYSTEM INSTRUCTIONS
        {self.system_instructions}
        # EXTERNAL CONTEXT INFORMATION
        {len(self.recall_storage)} previous messages between you and the user are stored in recall storage (use functions to access them)
        {len(self.archival_storage)} total memories you created are stored in archival storage (use functions to access them)
        # WORKING CONTEXT (limited in size, additional information stored in archival/recall storage)>
        {str(self.working_context)}'''
    
    @property
    def main_ctx_message_seq(self):
        #note: messaged must be in the form {'type': type, {'role': role, 'content': content}}
        translated_messages = []
        for messaged in self.fifo_queue.queue:
            if messaged['type'] == 'system':
                translated_messages.append({"role": "user", "content": f"System: messaged['message']['content']"})
            elif messaged['type'] == 'function':
                translated_messages.append({"role": "user", "content": f"Function: messaged['message']['content']"})
            else:
                translated_messages.append(messaged['message'])

        return (
            [{"role": "system", "content": self.main_context_system_message}]
            + translated_messages
        )

    @property
    def main_ctx_message_seq_no_tokens(self):
        return self.ct_num_token_func(self.main_ctx_message_seq)
