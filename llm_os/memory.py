from dataclasses import dataclass
from queue import Queue

from HOST import HOST_URL, LLM_NAME, LCOLLAMA_MODEL

class MainContext:
    def __init__(self, system_instructions: str, working_context: list[str], fifo_queue: Queue[str]):
        self.system_instructions = system_instructions
        self.working_context = working_context
        self.fifo_queue = fifo_queue
    
    @property
    def main_context_str(self):
        return self.system_instructions+'\n'+'\n'.join(self.working_context)+'\n'+'\n'.join(self.fifo_queue.queue)

    @property
    def main_context_number_of_tokens(self):
        return LCOLLAMA_MODEL.get_num_tokens(self.main_context_str)

@dataclass
class RecallStorageDatum:
    datum_datetime: date
    datum: str

@dataclass
class Memory:
    main_context: MainContext
    recall_storage: list[RecallStorageDatum]
    archival_storage: list[str]
