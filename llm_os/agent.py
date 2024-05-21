from config import CONFIG
from host import HOST

from llm_os.interface import CLIInterface
from llm_os.memory.main_context import MainContext
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage

class Agent:
    def __init__(
            self, 
            interface: CLIInterface, 
            model_name: str, 
            system_instructions: str, working_context: WorkingContext, fifo_queue: Queue[str],
            archival_storage: ArchivalStorage,
            recall_storage: RecallStorage
        ):
        self.interface = interface
        self.model_name = model_name

        self.main_context = MainContext(
            self.model_name,
            system_instructions, working_context, fifo_queue, 
            archival_storage, recall_storage
        )

    
