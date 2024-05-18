from dataclasses import dataclass
from queue import Queue
from string import digits 

from llm_os.memory import Memory, RecallStorageDatum, MainContext

# Executor
@dataclass
class InstructionResult:
    result: str|None
    error: str|None
    request_heartbeat: bool

class Instruction:
    def __init__(self, request_heartbeat: bool, instruction: str, arguments: list[str]):
        self.request_heartbeat = request_heartbeat
        self.instruction: str = instruction.lower()
        self.arguments: list[str] = [argument.lower() for argument in arguments]

    def call(self, memory: Memory) -> InstructionResult:
        instruction_caller = self.getattr(f'execute_{self.instruction}', self.execute_unknown_instruction)
        return instruction_caller(self.instruction, self.request_heartbeat, self.arguments)

    def execute_unknown_instruction(self, memory: Memory, request_heartbeat: bool, instruction: str, arguments: list[str]) -> InstructionResult:
        return InstructionResult(None, 'Unknown instruction', request_heartbeat)

    def execute_workingctx_search(self, memory: Memory, request_heartbeat: bool, instruction: str, arguments: list[str]) -> InstructionResult:
        if len(arguments) != 2:
            return InstructionResult(None, f'Invalid number of arguments (expected 2 but got {len(arguments)})', request_heartbeat)

        working_context = memory.main_context.working_context
        working_context_len = len(working_context)

        query = arguments[0]
        page_no = int(arguments[1])

        total_search_results = [f'"{item}"' for item for item in working_context if query in item]
        total_search_results_len = len(total_search_results)
        
        last_page_no = total_search_results_len//10
        
        lower_bound = (page_no-1)*10
        if lower_bound >= total_search_results_len:
            page_no = last_page_no
            lower_bound = (page_no-1)*10

        upper_bound = page_no*10

        if upper_bound > total_search_results_len:
            upper_bound = total_search_results_len

        search_results = total_search_results[lower_bound:upper_bound]
        if len(search_results) == 0:
            return InstructionResult(None, 'Not found', request_heartbeat)

        return InstructionResult(f'Showing {len(search_results)}/{total_search_results_len} results (page {page_no}/{last_page_no}):\n' + '\n'.join(search_results[]), None, request_heartbeat)

    def execute_recallstorage_search(self, memory: Memory, request_heartbeat: bool, instruction: str, arguments: list[str]) -> InstructionResult:
        if len(arguments) != 2:
            return InstructionResult(None, f'Invalid number of arguments (expected 2 but got {len(arguments)})', request_heartbeat)

        recall_storage = memory.main_context.working_context
        recall_storage_len = len(recall_storage)

        query = arguments[0]
        page_no = int(arguments[1])

        total_search_results = [f'[{item.datum_datetime.day}/{item.datum_datetime.month}/{item.datum_datetime.year}] "{item.datum}"' for item for item in recall_storage if query in item]
        total_search_results_len = len(total_search_results)
        
        last_page_no = total_search_results_len//10
        
        lower_bound = (page_no-1)*10
        if lower_bound >= total_search_results_len:
            page_no = last_page_no
            lower_bound = (page_no-1)*10

        upper_bound = page_no*10

        if upper_bound > total_search_results_len:
            upper_bound = total_search_results_len

        search_results = total_search_results[lower_bound:upper_bound]
        if len(search_results) == 0:
            return InstructionResult(None, 'Not found', request_heartbeat)

        return InstructionResult(f'Showing {len(search_results)}/{total_search_results_len} results (page {page_no}/{last_page_no}):\n' + '\n'.join(search_results[]), None, request_heartbeat)

    def execute_archivalstorage_search(self, memory: Memory, request_heartbeat: bool, instruction: str, arguments: list[str]) -> InstructionResult:
        if len(arguments) != 2:
            return InstructionResult(None, f'Invalid number of arguments (expected 2 but got {len(arguments)})', request_heartbeat)

        archival_storage = memory.archival_storage
        archival_storage_len = len(archival_storage)

        query = arguments[0]
        page_no = int(arguments[1])

        total_search_results = [f'"{item}"' for item for item in archival_storage if query in item]
        total_search_results_len = len(total_search_results)
        
        last_page_no = total_search_results_len//10
        
        lower_bound = (page_no-1)*10
        if lower_bound >= total_search_results_len:
            page_no = last_page_no
            lower_bound = (page_no-1)*10

        upper_bound = page_no*10

        if upper_bound > total_search_results_len:
            upper_bound = total_search_results_len

        search_results = total_search_results[lower_bound:upper_bound]
        if len(search_results) == 0:
            return InstructionResult(None, 'Not found', request_heartbeat)

        return InstructionResult(f'Showing {len(search_results)}/{total_search_results_len} results (page {page_no}/{last_page_no}):\n' + '\n'.join(search_results[]), None, request_heartbeat)

    
    def execute_workingctx_write(self, memory: Memory, request_heartbeat: bool, instruction: str, arguments: list[str]) -> InstructionResult:
        if len(arguments) != 1:
            return InstructionResult(None, f'Invalid number of arguments (expected 1 but got {len(arguments)})', request_heartbeat)

        memory.main_context.working_context.append(arguments[0])
        return InstructionResult(f'Successfully written "{arguments[0]}" to working context', None, request_heartbeat)

    def execute_archivalstorage_write(self, memory: Memory, request_heartbeat: bool, instruction: str, arguments: list[str]) -> InstructionResult:
        if len(arguments) != 1:
            return InstructionResult(None, f'Invalid number of arguments (expected 1 but got {len(arguments)})', request_heartbeat)

        memory.archival_storage.append(arguments[0])
        return InstructionResult(f'Successfully written "{arguments[0]}" to archival storage', None, request_heartbeat)

# Parser
@dataclass
class ParseResult:
    result: Instruction|None
    error: str|None

def parse(instruction_raw_txt: str) -> ParseResult:
    '''
    Ai Instruction Language (AILang) syntax:
    ENABLEHEARTBEAT -> %
    KEYWORD -> an AILang instruction keyword
    STRING -> a string (e.g. "Hello World!") (MUST BE ENCLOSED IN DOUBLE QUOTES)
    INT -> an integer

    instruction: KEYWORD(ENABLEHEARTBEAT)? (argument)*?
    argument: INT|STRING
    '''
    
    instruction_splitted = instruction_raw_txt.strip().lower().split(' ')
    if len(instruction_splitted) == 0:
        return ParseResult(None, 'Empty instruction')

    if instruction_splitted[0].endswith('%'):
        request_heartbeat = True
        instruction = instruction_splitted[0][:-1]
    else:
        request_heartbeat = False
        instruction = instruction_splitted[0]

    if instruction == '':
        return ParseResult(None, 'No KEYWORD present in instruction')

    arguments = instruction_splitted[1:]

    for argument in arguments:
        if argument.startswith('"') and argument.endswith('"'):
            argument = argument[1:-1]
        elif all(map(lambda c: c in digits, argument)):
            argument = int(argument)
        else:
            return ParseResult(None, 'Invalid argument')

    return ParseResult(Instruction(request_heartbeat, instruction, arguments), None)
