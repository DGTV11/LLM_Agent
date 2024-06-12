from dataclasses import dataclass
from queue import Queue

from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage
from llm_os.memory.working_context import WorkingContext

class Memory:
    def __init__(
            self,       
            model_name: str, 
            function_dats: dict,
            system_instructions: str, working_context: WorkingContext, fifo_queue: Queue[str], 
            archival_storage: ArchivalStorage, recall_storage: RecallStorage
        ):
        # Function data      
        self.function_dats = function_dats

        # Main context
        self.system_instructions = system_instructions
        self.working_context = working_context
        self.fifo_queue = fifo_queue

        # External context
        self.archival_storage = archival_storage
        self.recall_storage = recall_storage

        self.tokenizer, self.ctx_window, self.num_token_func, self.ct_num_token_func = get_tokeniser_and_context_window(model_name)

    def append_messaged_to_fq_and_rs(self, messaged):
        #note: messaged must be in the form {'type': type, {'role': role, 'content': content}}
        self.fifo_queue.put(messaged)
        self.recall_storage.insert(messaged)
    
    @property
    def main_context_system_message(self):
        return f'''# SYSTEM INSTRUCTIONS
        {self.system_instructions}
        # FUNCTION JSON SCHEMAS
        {'\n'.join([dat[json_schema] for dat in self.function_dats.values()])}
        # EXTERNAL CONTEXT INFORMATION
        {len(self.recall_storage)} previous messages between you and the user are stored in recall storage (use functions to access them)
        {len(self.archival_storage)} total memories you created are stored in archival storage (use functions to access them)
        # CORE MEMORY (limited in size, additional information stored in archival/recall storage)>
        {str(self.working_context)}'''
    
    @property
    def main_ctx_message_seq(self):
        #note: messaged must be in the form {'type': type, 'message': {'role': role, 'content': content}}
        translated_messages = []
        user_role_buf = []

        for messaged in self.fifo_queue.queue:
            if messaged['type'] == 'system':
                user_role_buf.append(f"(SYSTEM MESSAGE) {messaged['message']['content']}"})
            elif messaged['type'] == 'tool':
                user_role_buf.append(f"(TOOL MESSAGE) {messaged['message']['content']}")
            elif messaged['type'] == 'user':
                user_role_buf.append(f"(USER MESSAGE) {messaged['message']['content']}")
            else:
                translated_messages.append({'role': 'user', 'content': '\n\n'.join(user_role_buf)})
                translated_messages.append(messaged['message'])
                user_role_buf = []

        if user_role_buf:
            translated_messages.append({'role': 'user', 'content': '\n\n'.join(user_role_buf)})

        return (
            [{"role": "system", "content": self.main_context_system_message}]
            + translated_messages
        )

    @property
    def main_ctx_message_seq_no_tokens(self):
        return self.ct_num_token_func(self.main_ctx_message_seq)
