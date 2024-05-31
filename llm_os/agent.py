from queue import Queue
import json

from config import CONFIG
from host import HOST

from llm_os.interface import CLIInterface
from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage

class Agent:
    def __init__(
            self, 
            interface: CLIInterface, 
            model_name: str, 
            function_dats: dict,
            system_instructions: str, working_context: WorkingContext, fifo_queue: Queue[str],
            archival_storage: ArchivalStorage,
            recall_storage: RecallStorage
        ):
        self.interface = interface
        self.model_name = model_name

        self.memory = Memory(
            self.model_name,
            function_dats,
            system_instructions, working_context, fifo_queue, 
            archival_storage, recall_storage
        )

        self.memory_pressure_warning_alr_given = False

    @staticmethod
    def package_tool_response(result, has_error):
        if has_error:
            status = f'Status: Failed.'
        else:
            status = f'Status: OK.'

        return {
            'type': 'tool',
            'message': {
                'role': 'user',
                'content': f'{status} Result: {result}'
            },
        }

    def __call_function(self, function_call):
        function_name = function_call['name']
        function_params = function_call['parameters']
        function = self.memory.function_dats[function_name]['python_function']


    def step(self, user_messaged) -> str: #TODO
        #note: messaged must be in the form {'type': type, {'role': role, 'content': content}}
        
        queued_messageds = user_messaged

        ## Step 1: Generate response
        result = HOST.chat(
            model=self.model_name, 
            messages=self.memory.main_ctx_message_seq(messaged),
            format="json",
            options={"num_ctx": self.ctx_window},
        )
        json_result = json.loads(result)
        self.interface.internal_monologue(json_result['thoughts'])

        ## Step 2: Check if LLM wanted to call a function
        if 'function_call' in json_result:
            res_messageds, heartbeat_request, function_failed = self.__call_function(json_result['function_call'])
        else:
            pass
