from dataclasses import dataclass
from queue import Queue

from llm_os.tokenisers import get_tokeniser_and_context_window
from llm_os.archival_storage import ArchivalStorage

@dataclass
class WorkingContext:
    persona: str
    human: str

class MainContext:
    def __init__(self, model_name: str, system_instructions: str, working_context: WorkingContext, fifo_queue: Queue[str]):
        self.system_instructions = system_instructions
        self.working_context = working_context
        self.fifo_queue = fifo_queue

        self.tokenizer, self.ctx_window = get_tokeniser_and_context_window(model_name)
        self.num_token_func = lambda text: len(self.tokenizer.encode(text).ids)
    
    @property
    def main_context_str(self, recall_storage, archival_storage: ArchivalStorage):
        return f'''# SYSTEM INSTRUCTIONS
        {self.system_instructions}
        # EXTERNAL CONTEXT INFORMATION
        {len(recall_storage) if recall_storage else 0} previous messages between you and the user are stored in recall storage (use functions to access them)
        {len(archival_storage) if archival_storage else 0} total memories you created are stored in archival storage (use functions to access them)
        # WORKING CONTEXT (limited in size, additional information stored in archival/recall storage)>
        <persona>
        {self.working_context.persona}
        </persona>
        <human>
        {self.working_context.human}
        </human>
        # FIFO QUEUE
        {'\n'.join(self.fifo_queue.queue)'''

    @property
    def main_context_number_of_tokens(self):
        return self.num_token_func(self.main_context_str)
