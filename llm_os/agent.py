from os import path
from collections import deque
from functools import reduce
import json, json5
import regex

from host import HOST

from llm_os.interface import CLIInterface
from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage
from llm_os.prompts.llm_os_summarize import get_summarise_system_prompt
from llm_os.constants import (
    SHOW_DEBUG_MESSAGES,
    SEND_MESSAGE_FUNCTION_NAME,
    FIRST_MESSAGE_COMPULSORY_FUNCTION_SET,
    PY_TO_JSON_TYPE_MAP,
    JSON_TO_PY_TYPE_MAP,
    FUNCTION_PARAM_NAME_REQ_HEARTBEAT,
    WARNING_TOKEN_FRAC,
    FLUSH_TOKEN_FRAC,
    TRUNCATION_TOKEN_FRAC,
    LAST_N_MESSAGES_TO_PRESERVE,
)


class Agent:
    def __init__(
        self,
        interface: CLIInterface,
        conv_name: str,
        model_name: str,
        function_dats: dict,
        system_instructions: str,
        working_context: WorkingContext,
        archival_storage: ArchivalStorage,
        recall_storage: RecallStorage,
    ):
        self.misc_info_path = path.join(
            path.dirname(path.dirname(__file__)),
            "persistent_storage",
            conv_name,
            "misc_info.json",
        )

        self.interface = interface
        self.model_name = model_name
        self.conv_name = conv_name

        self.memory = Memory(
            self.model_name,
            self.conv_name,
            function_dats,
            system_instructions,
            working_context,
            archival_storage,
            recall_storage,
        )

        self.__memory_pressure_warning_alr_given = False
        if path.exists(self.misc_info_path):
            self.__save_misc_info_path_dat_to_misc_info_vars()
        else:
            self.__write_misc_info_vars_to_misc_info_path_dat()

    def __save_misc_info_path_dat_to_misc_info_vars(self):
        with open(self.misc_info_path, "r") as f:
            misc_info = json.loads(f.read())
            self.__memory_pressure_warning_alr_given = misc_info[
                "memory_pressure_warning_alr_given"
            ]

    def __write_misc_info_vars_to_misc_info_path_dat(self):
        if not path.exists(self.misc_info_path):
            f = open(self.misc_info_path, "x")
            f.close()
        with open(self.misc_info_path, "w") as f:
            misc_info = {
                "memory_pressure_warning_alr_given": self.__memory_pressure_warning_alr_given
            }
            f.write(json.dumps(misc_info))

    @property
    def memory_pressure_warning_alr_given(self):
        return self.__memory_pressure_warning_alr_given

    @memory_pressure_warning_alr_given.setter
    def memory_pressure_warning_alr_given(self, value):
        self.__memory_pressure_warning_alr_given = value
        self.__write_misc_info_vars_to_misc_info_path()

    @staticmethod
    def package_tool_response(result, has_error):
        if has_error:
            status = f"Status: Failed."
        else:
            status = f"Status: OK."

        return {
            "type": "tool",
            "message": {"role": "user", "content": f"{status} Result: {result}"},
        }

    def __call_function(self, function_call, is_first_message=False):
        # Returns: res_messageds, heartbeat_request, function_failed
        res_messageds = []

        # Step 1: Parse function call
        try:
            called_function_name = function_call["name"]
            if is_first_message and called_function_name not in FIRST_MESSAGE_COMPULSORY_FUNCTION_SET:
                surround_with_single_quotes = lambda s: f"'{s}'"
                interface_message = f"Name of function called during starting message of conversation MUST be in {', '.join(map(surround_with_single_quotes, FIRST_MESSAGE_COMPULSORY_FUNCTION_SET))}."
                res_messageds.append(Agent.package_tool_response(interface_message, True))
                self.interface.function_res_message(interface_message, True)

                return res_messageds, True, True  # Sends heartbeat request so LLM can retry

            called_function_arguments = function_call["arguments"]
        except KeyError as e:
            interface_message = f"Failed to parse function call: Missing {e} field."
            if 'arguments' in e and 'parameters' in function_call:
                interface_message += " Please replace the 'parameters' field with the 'arguments' field."

            res_messageds.append(Agent.package_tool_response(interface_message, True))
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        # Step 2: Check if function exists
        called_function_dat = self.memory.function_dats.get(called_function_name, None)
        if not called_function_dat:
            interface_message = f'Function "{called_function_name}" does not exist.'
            res_messageds.append(Agent.package_tool_response(interface_message, True))
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        # Step 3: Get python function and function schema
        called_function = called_function_dat["python_function"]
        called_function_schema = called_function_dat["json_schema"]
        called_function_parameters = called_function_schema["parameters"]["properties"]
        called_function_required_parameter_names = called_function_schema["parameters"]["required"]

        # Step 4: Valiate arguments
        ## Check if required arguments are present
        called_function_parameter_names = called_function_parameters.keys()
        for argument in called_function_arguments.keys():
            if argument not in called_function_parameter_names:
                interface_message = f'Function "{called_function_name}" does not accept argument "{argument}".'
                res_messageds.append(
                    Agent.package_tool_response(interface_message, True)
                )
                self.interface.function_res_message(interface_message, True)

                return (
                    res_messageds,
                    True,
                    True,
                )  # Sends heartbeat request so LLM can retry

        if len(called_function_arguments) < len(called_function_required_parameter_names):
            interface_message = f'Function "{called_function_name}" requires at least {len(called_function_required_parameter_names)} arguments ({len(called_function_arguments)} given).'
            res_messageds.append(Agent.package_tool_response(interface_message, True))
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry
        if len(called_function_arguments) > len(called_function_parameter_names):
            interface_message = f'Function "{called_function_name}" can take at most {len(called_function_parameter_names)} arguments ({len(called_function_arguments)} given).'
            res_messageds.append(Agent.package_tool_response(interface_message, True))
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry
 
        if not set(called_function_required_parameter_names).issubset(set(called_function_arguments.keys())):
            required_arguments_str = ",".join(map(lambda arg_name: f"'{arg_name}'", called_function_required_parameter_names))
            given_arguments_str = ",".join(map(lambda arg_name: f"'{arg_name}'", called_function_arguments.keys())) 

            interface_message = f'Function "{called_function_name}" requires at least the arguments {required_arguments_str} ({given_arguments_str} given).'
            res_messageds.append(Agent.package_tool_response(interface_message, True))
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry


        ## Check if arguments are of the correct type
        for argument_name, argument_value in called_function_arguments.items():
            required_param_type = called_function_parameters[argument_name]["type"]

            if type(argument_value) is list:
                if required_param_type != "array":
                    interface_message = f'Function "{called_function_name}" does not accept argument "{argument_name}" of type "array" (expected type "{required_param_type}").'
                    res_messageds.append(
                        Agent.package_tool_response(interface_message, True)
                    )
                    self.interface.function_res_message(interface_message, True)

                    return (
                        res_messageds,
                        True,
                        True,
                    )  # Sends heartbeat request so LLM can retry
                param_array_field_type = JSON_TO_PY_TYPE_MAP["array"].__args__[0]
                all_arg_elem_correct_type = reduce(
                    lambda x, y: x and y,
                    map(lambda x: type(x) is param_array_field_type, argument_value),
                    True,
                )
                if not all_arg_elem_correct_type:
                    interface_message = f'Function "{called_function_name}" does not accept argument "{argument_name}" of type "array" (some or all elements are not of type {PY_TO_JSON_TYPE_MAP[param_array_field_type]}).'
                    res_messageds.append(
                        Agent.package_tool_response(interface_message, True)
                    )
                    self.interface.function_res_message(interface_message, True)

                    return (
                        res_messageds,
                        True,
                        True,
                    )  # Sends heartbeat request so LLM can retry
                continue

            argument_value_type = PY_TO_JSON_TYPE_MAP[type(argument_value)]
            if required_param_type == "array":
                interface_message = f'Function "{called_function_name}" does not accept argument "{argument_name}" of type "{argument_value_type}" (expected type "array").'
                res_messageds.append(
                    Agent.package_tool_response(interface_message, True)
                )
                self.interface.function_res_message(interface_message, True)

                return (
                    res_messageds,
                    True,
                    True,
                )  # Sends heartbeat request so LLM can retry

            if argument_value_type != required_param_type:
                interface_message = f'Function "{called_function_name}" does not accept argument "{argument_name}" of type "{argument_value_type}" (expected type "{required_param_type}").'
                res_messageds.append(
                    Agent.package_tool_response(interface_message, True)
                )
                self.interface.function_res_message(interface_message, True)

                return (
                    res_messageds,
                    True,
                    True,
                )  # Sends heartbeat request so LLM can retry

        # Step 5: Call function
        called_heartbeat_request = called_function_arguments.get(
            FUNCTION_PARAM_NAME_REQ_HEARTBEAT, None
        )

        if called_heartbeat_request is not None:
            if is_first_message and not called_heartbeat_request:
                interface_message = f"Function called during starting message of conversation MUST request a heartbeat through \"'heartbeat_request': true\" IF the function name is not '{SEND_MESSAGE_FUNCTION_NAME}'."
                res_messageds.append(Agent.package_tool_response(interface_message, True))
                self.interface.function_res_message(interface_message, True)

                return res_messageds, True, True  # Sends heartbeat request so LLM can retry

            del called_function_arguments[FUNCTION_PARAM_NAME_REQ_HEARTBEAT]
        else:
            called_heartbeat_request = False

        called_function_arguments["self"] = self

        try:
            self.interface.function_call_message(
                called_function_name, called_function_arguments
            )
            called_function_result = called_function(**called_function_arguments)
        except Exception as e:
            res_messageds.append(Agent.package_tool_response(e, True))
            self.interface.function_res_message(e, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        # Step 6: Package response
        res_messageds.append(Agent.package_tool_response(called_function_result, False))
        self.interface.function_res_message(called_function_result, False)

        return res_messageds, called_heartbeat_request, False

    def step(self, is_first_message=False) -> str:
        # note: all messageds must be in the form {'type': type, 'message': {'role': role, 'content': content}}

        ## Step 0: Check memory pressure
        if (
            not self.memory_pressure_warning_alr_given
            and self.memory.main_ctx_message_seq_no_tokens
            > int(WARNING_TOKEN_FRAC * self.memory.ctx_window)
        ):
            interface_message = f"Warning: Memory pressure has exceeded {WARNING_TOKEN_FRAC*100}% of the context window. Consider storing important information from your recent conversation history into your core memory or archival storage after responding to the user's query (if any)."
            res_messageds.append(
                {
                    "type": "system",
                    "message": {"role": "user", "content": interface_message},
                }
            )
            self.interface.system_message(interface_message)

            self.memory_pressure_warning_alr_given = True
            heartbeat_request = True
        elif (
            self.memory_pressure_warning_alr_given
            and self.memory.main_ctx_message_seq_no_tokens
            > int(FLUSH_TOKEN_FRAC * self.memory.ctx_window)
        ):
            self.summarise_messages_in_place()
            self.memory_pressure_warning_alr_given = False

        ## Step 1: Generate response
        result = HOST.chat(
            model=self.model_name,
            messages=self.memory.main_ctx_message_seq,
            options={"num_ctx": self.memory.ctx_window},
            stream=True
        )
        result_content = ""

        if SHOW_DEBUG_MESSAGES:
            self.interface.debug_message(f'Got result:')

            for chunk in result:
                chunk_content = chunk['message']['content']
                result_content += chunk_content
                self.interface.append_to_message(chunk_content)

            self.interface.append_to_message('\n')
        else:
            for chunk in result:
                result_content += chunk['message']['content']

        try:
            json_object_finder = regex.compile(r'\{(?:[^{}]|(?R))*\}')
            found_json_objects = json_object_finder.findall(result_content)
            if len(found_json_objects) != 1:
                raise RuntimeError

            result_content = found_json_objects[0]

            res_messageds = [{"type": "assistant", "message": {"role": "assistant", "content": result_content}}]

            json_result = json5.loads(result_content)
            if type(json_result) is not dict:
                raise RuntimeError
        except (RuntimeError, ValueError):
            res_messageds = [{"type": "assistant", "message": {"role": "assistant", "content": result_content}}]

            if is_first_message:
                interface_message = "Error: you MUST give a SINGLE WELL-FORMED JSON object AND ONLY THAT JSON OBJECT that includes the 'thoughts' field as your internal monologue and the 'function_call' field as a function call ('function_call' field is required during the starting message of a conversation and highly recommended otherwise)! You must NOT give ANY extra text other than the JSON object! Please try again without acknowledging this message."
            else:
                interface_message = "Error: you MUST give a SINGLE WELL-FORMED JSON object AND ONLY THAT OBJECT that at least includes the 'thoughts' field as your internal monologue! If you would like to call a function, do include the 'function_call' field. You must NOT give ANY extra text other than the JSON object! Please try again without acknowledging this message."
            res_messageds.append(
                {
                    "type": "system",
                    "message": {"role": "user", "content": interface_message},
                }
            )
            self.interface.system_message(interface_message)
            heartbeat_request = True
            function_failed = False
        else:
            unidentified_keys = []
            for key in json_result.keys():
                if key not in ["thoughts", "function_call"]:
                    unidentified_keys.append(key)

            if (not unidentified_keys) and json_result.get("thoughts", None):
                self.interface.internal_monologue(json_result["thoughts"])

                ## Step 2: Check if LLM wanted to call a function
                if "function_call" in json_result:
                    d_res_messageds, heartbeat_request, function_failed = self.__call_function(
                        json_result["function_call"], is_first_message
                    )
                    res_messageds += d_res_messageds
                else:
                    if is_first_message:
                        interface_message = "Error: you MUST include a function call inside the 'function_call' field as per the given schema during the starting message of a conversation. Please try again without acknowledging this message."
                        res_messageds.append(
                            {
                                "type": "system",
                                "message": {"role": "user", "content": interface_message},
                            }
                        )
                        self.interface.system_message(interface_message)
                        
                    heartbeat_request = is_first_message

                    function_failed = False
            elif unidentified_keys:
                surround_with_single_quotes = lambda s: f"'{s}'"
                interface_message = f"Error: fields {', '.join(map(surround_with_single_quotes, unidentified_keys))} should not be included in your generated JSON object (refer to the given JSON schema!). Please try again without acknowledging this message."
                res_messageds.append(
                    {
                        "type": "system",
                        "message": {"role": "user", "content": interface_message},
                    }
                )
                self.interface.system_message(interface_message)
                heartbeat_request = True
                function_failed = False
            else:
                interface_message = "Error: you MUST at least include the 'thoughts' field in the JSON object you respond with as your internal monologue! If you wanted to call a function, please try again while including your internal_monologue. Please try again without acknowledging this message."
                res_messageds.append(
                    {
                        "type": "system",
                        "message": {"role": "user", "content": interface_message},
                    }
                )
                self.interface.system_message(interface_message)
                heartbeat_request = True
                function_failed = False

        ## Step 3: Check memory pressure
        if (
            not self.memory_pressure_warning_alr_given
            and self.memory.main_ctx_message_seq_no_tokens
            > int(WARNING_TOKEN_FRAC * self.memory.ctx_window)
        ):
            interface_message = f"Warning: Memory pressure has exceeded {WARNING_TOKEN_FRAC*100}% of the context window. Consider storing important information from your recent conversation history into your core memory or archival storage."
            res_messageds.append(
                {
                    "type": "system",
                    "message": {"role": "user", "content": interface_message},
                }
            )
            self.interface.system_message(interface_message)

            self.memory_pressure_warning_alr_given = True
            heartbeat_request = True
        elif (
            self.memory_pressure_warning_alr_given
            and self.memory.main_ctx_message_seq_no_tokens
            > int(FLUSH_TOKEN_FRAC * self.memory.ctx_window)
        ):
            self.summarise_messages_in_place()
            self.memory_pressure_warning_alr_given = False

        ## Step 4: Update memory
        for messaged in res_messageds:
            self.memory.append_messaged_to_fq_and_rs(messaged)

        ## Step 5: Return response
        return res_messageds, heartbeat_request, function_failed

    @staticmethod
    def summary_message_seq(messaged_seq):
        # note: messaged must be in the form {'type': type, 'message': {'role': role, 'content': content}}
        translated_messages = []
        user_role_buf = []

        for messaged in messaged_seq:
            if messaged["type"] == "system":
                user_role_buf.append(
                    f"❮SYSTEM MESSAGE❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "tool":
                user_role_buf.append(f"❮TOOL MESSAGE❯ {messaged['message']['content']}")
            elif messaged["type"] == "user":
                user_role_buf.append(f"❮USER MESSAGE❯ {messaged['message']['content']}")
            else:
                translated_messages.append(
                    {"role": "user", "content": "\n\n".join(user_role_buf)}
                )
                message_content_dict = json5.loads(messaged["message"]["content"])
                assistant_message_content = (
                    "❮ASSISTANT MESSAGE❯" + message_content_dict["thoughts"]
                )
                if "function_call" in message_content_dict:
                    assistant_message_content += (
                        "\n\n❮TOOL CALL❯" + message_content_dict["function_call"]
                    )
                translated_messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )
                user_role_buf = []

        if user_role_buf:
            translated_messages.append(
                {"role": "user", "content": "\n\n".join(user_role_buf)}
            )

        return [
            {"role": "system", "content": get_summarise_system_prompt}
        ] + translated_messages

    def summarise_messages_in_place(self):
        assert self.memory.main_ctx_message_seq[0]["role"] == "system"

        messages_to_be_summarised = deque()

        while (
            self.memory.main_ctx_message_seq_no_tokens
            > int(TRUNCATION_TOKEN_FRAC * self.memory.ctx_window)
            and len(self.memory.fifo_queue) > LAST_N_MESSAGES_TO_PRESERVE
        ):
            if (
                len(self.memory.fifo_queue) + 1 > LAST_N_MESSAGES_TO_PRESERVE
                and self.memory.fifo_queue[0]["message"]["role"] == "assistant"
            ):
                break
            self.memory.no_messages_in_queue -= 1
            messages_to_be_summarised.appendleft(self.memory.popleft())
        self.memory.__write_fq_to_fq_path()

        summary_message_seq = Agent.summary_message_seq(messages_to_be_summarised)

        result = HOST.chat(
            model=self.model_name,
            messages=summary_message_seq,
            options={"num_ctx": self.memory.ctx_window},
        )
        result_content = result["message"]["content"]

        self.memory.fifo_queue.appendleft(
            {
                "type": "system",
                "message": {
                    "role": "user",
                    "content": (
                        f"Note: prior messages ({self.memory.total_no_messages-self.memory.no_messages_in_queue} of {self.memory.total_no_messages}) have been hidden from view due to conversation memory constraints.\n"
                        + f"The following is a summary of the previous {len(summary_message_seq)} messages:\n{result_content}"
                    ),
                },
            }
        )
        self.memory.no_messages_in_queue += 1
