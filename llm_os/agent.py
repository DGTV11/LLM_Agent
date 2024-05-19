from config import CONFIG
from host import HOST

fromllm_os.interface import CLIInterface

from llm_os.memory.main_context import WorkingContext, MainContext
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage

class Agent:
    def __init__(self, interface: CLIInterface):
        self.interface = interface
