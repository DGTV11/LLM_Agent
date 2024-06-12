from queue import Queue
from functools import reduce
import json

from config import CONFIG
from host import HOST

from llm_os.interface import CLIInterface
from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage
from llm_os.constants import PY_TO_JSON_TYPE_MAP, JSON_TO_PY_TYPE_MAP, FUNCTION_PARAM_NAME_REQ_HEARTBEAT, WARNING_TOKEN_FRAC, FLUSH_TOKEN_FRAC

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

    def __call_function(self, function_call): #TODO
        # Returns: res_messageds, heartbeat_request, function_failed
        res_messageds = []  
        
        # Step 1: Parse function call
        try:
            called_function_name = function_call['name']
            called_function_arguments = function_call['arguments']
        except KeyError as e:
            res_messageds.append(Agent.package_tool_response(f'Failed to parse function call: {e}.', True))
            return res_messageds, True, True # Sends heartbeat request so LLM can retry

        # Step 2: Check if function exists
        called_function_dat = self.memory.function_dats.get(function_name, None)
        if not called_function_dat:
            res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" does not exist.', True))
            return res_messageds, True, True # Sends heartbeat request so LLM can retry

        # Step 3: Get python function and function schema
        called_function = function_dat['python_function']
        called_function_schema = json.loads(function_dat['json_schema'])
        called_function_parameters = called_function_schema['parameters']['properties']

        # Step 4: Valiate arguments
        ## Check if required arguments are present
        called_function_parameter_names = called_function_parameters.keys()
        for argument in called_function_arguments.keys():
            if argument not in called_function_parameter_names:
                res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" does not accept argument "{argument}".', True))
                return res_messageds, True, True # Sends heartbeat request so LLM can retry
        if len(called_function_arguments) != len(called_function_parameter_names):
            res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" requires {len(called_function_parameter_names)} arguments ({len(called_function_arguments)} given).', True))
            return res_messageds, True, True # Sends heartbeat request so LLM can retry
        ## Check if arguments are of the correct type
        for argument_name, argument_value in called_function_arguments.items():
            required_param_type = called_function_parameters[argument_name]['type']

            if type(argument_value) is list:
                if required_param_type != 'array':
                    res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" does not accept argument "{argument_name}" of type "array" (expected type "{required_param_type}").', True))
                    return res_messageds, True, True # Sends heartbeat request so LLM can retry
                param_array_field_type = JSON_TO_PY_TYPE_MAP['array'].__args__[0]
                all_arg_elem_correct_type = reduce(lambda x, y: x and y, map(lambda x: type(x) is param_array_field_type, argument_value), True)
                if not all_arg_elem_correct_type:
                    res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" does not accept argument "{argument_name}" of type "array" (some or all elements are not of type {PY_TO_JSON_TYPE_MAP[param_array_field_type]}).', True))
                    return res_messageds, True, True # Sends heartbeat request so LLM can retry
                continue

            argument_value_type = PY_TO_JSON_TYPE_MAP[type(argument_value)]
            if required_param_type == 'array':
                res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" does not accept argument "{argument_name}" of type "{argument_value_type}" (expected type "array").', True))
                return res_messageds, True, True # Sends heartbeat request so LLM can retry

            if argument_value_type != required_param_type:
                res_messageds.append(Agent.package_tool_response(f'Function "{function_name}" does not accept argument "{argument_name}" of type "{argument_value_type}" (expected type "{required_param_type}").', True))

        # Step 5: Call function
        called_heartbeat_request = called_function_arguments.get(FUNCTION_PARAM_NAME_REQ_HEARTBEAT, None)
        
        if called_heartbeat_request is not None:
            del called_function_arguments[FUNCTION_PARAM_NAME_REQ_HEARTBEAT]
        else:
            called_heartbeat_request = False

        called_function_arguments['self'] = self

        try:
            called_function_result = called_function(**called_function_arguments)
        except Exception as e:
            res_messageds.append(Agent.package_tool_response(e, True))
            return res_messageds, True, True # Sends heartbeat request so LLM can retry

        # Step 6: Package response
        res_messageds.append(Agent.package_tool_response(called_function_result, False))
        return res_messageds, called_heartbeat_request, False

    def step(self) -> str: #TODO
        #note: all messageds must be in the form {'type': type, 'message': {'role': role, 'content': content}}
        
        ## Step 1: Generate response
        result = HOST.chat(
            model=self.model_name, 
            messages=self.memory.main_ctx_message_seq,
            format="json",
            options={"num_ctx": self.ctx_window},
        )
        result_content = result['message']['content']
        res_messageds = [{'type': 'assistant', 'message': result['message']}]

        json_result = json.loads(result_content)
        self.interface.internal_monologue(json_result['thoughts'])

        ## Step 2: Check if LLM wanted to call a function
        if 'function_call' in json_result:
            d_res_messageds, heartbeat_request, function_failed = self.__call_function(json_result['function_call'])
            res_messageds += d_res_messageds
        else:
            heartbeat_request = function_failed = False
        
        ## Step 3: Check memory pressure
        if not self.memory_pressure_warning_alr_given and self.memory.main_ctx_message_seq_no_tokens > int(WARNING_TOKEN_FRAC * self.memory.ctx_window):
            res_messageds.append({'type': 'system', 'message': {'role': 'user', 'content': f"Warning: Memory pressure has exceeded {WARNING_TOKEN_FRAC*100}% of the context window. Consider storing important information from your recent conversation history into your core memory or archival storage."}})

        ## Step 4: Update memory
        for messaged in res_messageds:
            self.memory.append_messaged_to_fq_and_rs(queued_messageds)

        ## Step 5: Return response

