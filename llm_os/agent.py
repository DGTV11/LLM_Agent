from random import choice
from os import path
from collections import deque
from functools import reduce
import json, json5
import regex

from host import HOST

from llm_os.interface import CLIInterface
from llm_os.web_interface import WebInterface
from llm_os.memory.memory import Memory
from llm_os.memory.working_context import WorkingContext
from llm_os.memory.archival_storage import ArchivalStorage
from llm_os.memory.recall_storage import RecallStorage
from llm_os.prompts.llm_os_summarize import get_summarise_system_prompt
from llm_os.constants import (
    USE_JSON_MODE,
    USE_SET_STARTING_MESSAGE,
    SET_STARTING_THOUGHTS_LIST,
    SET_STARTING_GREETING_LIST,
    SET_STARTING_AUX_MESSAGE_LIST,
    SHOW_DEBUG_MESSAGES,
    SEND_MESSAGE_FUNCTION_NAME,
    MEMORY_EDITING_FUNCTIONS,
    WARNING__MESSAGE_SINCE_LAST_CONSCIOUS_MEMORY_EDIT__COUNT,
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
        web_interface: WebInterface,
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
        self.web_interface = web_interface

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
        self.__messages_since_last_conscious_memory_write = 0
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
            self.__messages_since_last_conscious_memory_write = misc_info[
                "messages_since_last_conscious_memory_write"
            ]

    def __write_misc_info_vars_to_misc_info_path_dat(self):
        if not path.exists(self.misc_info_path):
            f = open(self.misc_info_path, "x")
            f.close()
        with open(self.misc_info_path, "w") as f:
            misc_info = {
                "memory_pressure_warning_alr_given": self.__memory_pressure_warning_alr_given,
                "messages_since_last_conscious_memory_write": self.__messages_since_last_conscious_memory_write
            }
            f.write(json.dumps(misc_info))

    @property
    def memory_pressure_warning_alr_given(self):
        return self.__memory_pressure_warning_alr_given

    @memory_pressure_warning_alr_given.setter
    def memory_pressure_warning_alr_given(self, value):
        self.__memory_pressure_warning_alr_given = value
        self.__write_misc_info_vars_to_misc_info_path_dat()

    @property
    def messages_since_last_conscious_memory_write(self):
        return self.__messages_since_last_conscious_memory_write

    @messages_since_last_conscious_memory_write.setter
    def messages_since_last_conscious_memory_write(self, value):
        self.__messages_since_last_conscious_memory_write = value
        self.__write_misc_info_vars_to_misc_info_path_dat()

    @staticmethod
    def package_tool_response(user_id, result, has_error):
        if has_error:
            status = f"Status: Failed."
        else:
            status = f"Status: OK."

        trailing_message = (
            " Please try again without acknowledging this message." if has_error else ""
        )
        return {
            "type": "tool",
            "user_id": user_id,
            "message": {
                "role": "user",
                "content": f"{status} Result: {result}" + trailing_message,
            },
        }

    def __call_function(self, user_id, function_call, is_first_message=False):
        # Returns: res_messageds, heartbeat_request, function_failed
        res_messageds = []

        # Step 1: Parse function call
        if type(function_call) is not dict:
            interface_message = f"Failed to parse function call: 'function_call' field's value is not an object."

            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        try:
            called_function_name = function_call.get("name", (None, 0))
            if called_function_name == (None, 0):
                raise KeyError("name")
            if type(called_function_name) is not str:
                raise TypeError("'name' field's value is not a string.")

            if (
                is_first_message
                and called_function_name not in FIRST_MESSAGE_COMPULSORY_FUNCTION_SET
            ):
                surround_with_single_quotes = lambda s: f"'{s}'"
                interface_message = f"Name of function called during starting message of conversation MUST be in {', '.join(map(surround_with_single_quotes, FIRST_MESSAGE_COMPULSORY_FUNCTION_SET))}. Name of function called during starting message of conversation MUST NOT be {surround_with_single_quotes(called_function_name)}"
                res_messageds.append(
                    Agent.package_tool_response(user_id, interface_message, True)
                )
                self.interface.function_res_message(interface_message, True)

                return (
                    res_messageds,
                    True,
                    True,
                )  # Sends heartbeat request so LLM can retry

            called_function_arguments = function_call.get("arguments", (None, 0))
            if called_function_arguments == (None, 0):
                raise KeyError("arguments")
            if type(called_function_arguments) is not dict:
                raise TypeError("'arguments' field's value is not an object.")

        except KeyError as e:
            interface_message = f"Failed to parse function call: Missing {e} field of 'function_call' field. You need to add this field for the conversation to proceed!"
            if "arguments" in str(e) and "parameters" in function_call:
                interface_message += (
                    " You MUST replace the 'parameters' field with the 'arguments' field in your NEXT message!!!"
                )

            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry
        except TypeError as e:
            interface_message = f"Failed to parse function call: {e}"
            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        # Step 2: Check if function exists
        called_function_dat = self.memory.function_dats.get(called_function_name, None)
        if not called_function_dat:
            interface_message = f'Function "{called_function_name}" does not exist.'
            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        # Step 3: Get python function and function schema
        called_function = called_function_dat["python_function"]
        called_function_schema = called_function_dat["json_schema"]
        called_function_parameters = called_function_schema["parameters"]["properties"]
        called_function_required_parameter_names = called_function_schema["parameters"][
            "required"
        ]

        # Step 4: Valiate arguments
        ## Check if required arguments are present
        called_function_parameter_names = called_function_parameters.keys()
        for argument in called_function_arguments.keys():
            if argument not in called_function_parameter_names:
                interface_message = f'Function "{called_function_name}" does not accept argument "{argument}".'
                res_messageds.append(
                    Agent.package_tool_response(user_id, interface_message, True)
                )
                self.interface.function_res_message(interface_message, True)

                return (
                    res_messageds,
                    True,
                    True,
                )  # Sends heartbeat request so LLM can retry

        if len(called_function_arguments) < len(
            called_function_required_parameter_names
        ):
            interface_message = f'Function "{called_function_name}" requires at least {len(called_function_required_parameter_names)} arguments ({len(called_function_arguments)} given, missing arguments are {list(set(called_function_required_parameter_names)-set(called_function_arguments))}).'
            if (
                function_call.get("request_heartbeat", (None, 0)) != (None, 0)
                and "request_heartbeat" in called_function_required_parameter_names
            ):
                interface_message += ' Please move the "request_heartbeat" argument into the "arguments" field.'

            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry
        if len(called_function_arguments) > len(called_function_parameter_names):
            interface_message = f'Function "{called_function_name}" can take at most {len(called_function_parameter_names)} arguments ({len(called_function_arguments)} given).'
            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        if not set(called_function_required_parameter_names).issubset(
            set(called_function_arguments.keys())
        ):
            required_arguments_str = ",".join(
                map(
                    lambda arg_name: f"'{arg_name}'",
                    called_function_required_parameter_names,
                )
            )
            given_arguments_str = ",".join(
                map(lambda arg_name: f"'{arg_name}'", called_function_arguments.keys())
            )

            interface_message = f'Function "{called_function_name}" requires at least the arguments {required_arguments_str} ({given_arguments_str} given).'
            res_messageds.append(
                Agent.package_tool_response(user_id, interface_message, True)
            )
            self.interface.function_res_message(interface_message, True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        ## Check if arguments are of the correct type
        for argument_name, argument_value in called_function_arguments.items():
            required_param_type = called_function_parameters[argument_name]["type"]

            if type(argument_value) is list:
                if required_param_type != "array":
                    interface_message = f'Function "{called_function_name}" does not accept argument "{argument_name}" of type "array" (expected type "{required_param_type}").'
                    res_messageds.append(
                        Agent.package_tool_response(user_id, interface_message, True)
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
                        Agent.package_tool_response(user_id, interface_message, True)
                    )
                    self.interface.function_res_message(interface_message, True)

                    return (
                        res_messageds,
                        True,
                        True,
                    )  # Sends heartbeat request so LLM can retry
                continue

            argument_value_type = PY_TO_JSON_TYPE_MAP.get(type(argument_value), None)
            if required_param_type == "array":
                interface_message = f'Function "{called_function_name}" does not accept argument "{argument_name}" of type "{argument_value_type}" (expected type "array").'
                res_messageds.append(
                    Agent.package_tool_response(user_id, interface_message, True)
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
                    Agent.package_tool_response(user_id, interface_message, True)
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
                res_messageds.append(
                    Agent.package_tool_response(user_id, interface_message, True)
                )
                self.interface.function_res_message(interface_message, True)

                return (
                    res_messageds,
                    True,
                    True,
                )  # Sends heartbeat request so LLM can retry

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
            res_messageds.append(Agent.package_tool_response(user_id, str(e), True))
            self.interface.function_res_message(str(e), True)

            return res_messageds, True, True  # Sends heartbeat request so LLM can retry

        # Step 6: Package response
        res_messageds.append(
            Agent.package_tool_response(user_id, called_function_result, False)
        )
        self.interface.function_res_message(called_function_result, False)

        # Step 7: Check if called function is in MEMORY_EDITING_FUNCTIONS
        if called_function_name in MEMORY_EDITING_FUNCTIONS:
            self.messages_since_last_conscious_memory_write = -1

        return res_messageds, called_heartbeat_request, False

    def step(self, user_id, is_first_message=False) -> str:
        # note: all messageds must be in the form {'type': type, 'user_id': user_id, 'message': {'role': role, 'content': content}}
        ## Step -1: Update working context if needed
        self.memory.working_context.submit_used_human_id(user_id)

        ## Step 0: Check memory pressure
        if (
            self.memory.main_ctx_message_seq_no_tokens
            > int(FLUSH_TOKEN_FRAC * self.memory.ctx_window)
        ):
            self.summarise_messages_in_place()
            self.memory_pressure_warning_alr_given = False

        ## Step 1: Generate response
        if USE_SET_STARTING_MESSAGE and self.memory.total_no_messages == 1:
            HOST.generate(
                model=self.model_name,
                options={"num_ctx": self.memory.ctx_window}
            ) # load model into memory

            result_content = '''{
    "thoughts": "<ST>",
    "function_call": {
        "name": "send_message",
        "arguments": {
            "message": "<SM>"
        }
    }
}'''.replace("<ST>", choice(SET_STARTING_THOUGHTS_LIST)).replace("<SM>", ' '.join((choice(SET_STARTING_GREETING_LIST), choice(SET_STARTING_AUX_MESSAGE_LIST))))
            pass
        else: # Regular LLM inference
            if USE_JSON_MODE:
                response = HOST.chat(
                    model=self.model_name,
                    messages=self.memory.main_ctx_message_seq,
                    format="json",
                    options={"num_ctx": self.memory.ctx_window},
                )
            else:
                response = HOST.chat(
                    model=self.model_name,
                    messages=self.memory.main_ctx_message_seq,
                    options={"num_ctx": self.memory.ctx_window},
                )

            result_content = response["message"]["content"]

        if SHOW_DEBUG_MESSAGES:
            self.interface.debug_message(f"Got result:\n{result_content}")

        res_messageds = [
            {
                "type": "assistant",
                "user_id": user_id,
                "message": {"role": "assistant", "content": result_content},
            }
        ]

        try:
            json_result = json5.loads(result_content)
            if type(json_result) is not dict:
                raise RuntimeError
        except (RuntimeError, ValueError):
            if is_first_message:
                interface_message = "Error: you MUST give a SINGLE WELL-FORMED JSON object AND ONLY THAT JSON OBJECT that includes the 'thoughts' field as your internal monologue and the 'function_call' field as a function call ('function_call' field is required during the starting message of a conversation and highly recommended otherwise)! You must NOT give ANY extra text other than the JSON object! You must NOT just give a single piece of regular natural language! Please try again without acknowledging this message."
            else:
                interface_message = "Error: you MUST give a SINGLE WELL-FORMED JSON object AND ONLY THAT OBJECT that at least includes the 'thoughts' field as your internal monologue! If you would like to call a function, do include the 'function_call' field. You must NOT give ANY extra text other than the JSON object! You must NOT just give a single piece of regular natural language! Please try again without acknowledging this message."
            res_messageds.append(
                {
                    "type": "system",
                    "user_id": user_id,
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
                    d_res_messageds, heartbeat_request, function_failed = (
                        self.__call_function(
                            user_id, json_result["function_call"], is_first_message
                        )
                    )
                    res_messageds += d_res_messageds
                else:
                    if is_first_message:
                        interface_message = "Error: you MUST include a function call inside the 'function_call' field as per the given schema during the starting message of a conversation. Please try again without acknowledging this message."
                        res_messageds.append(
                            {
                                "type": "system",
                                "user_id": user_id,
                                "message": {
                                    "role": "user",
                                    "content": interface_message,
                                },
                            }
                        )
                        self.interface.system_message(interface_message)

                    heartbeat_request = is_first_message

                    function_failed = False
            elif unidentified_keys:
                surround_with_single_quotes = lambda s: f"'{s}'"
                interface_message = f"Error: fields {', '.join(map(surround_with_single_quotes, unidentified_keys))} should not be included in your generated JSON object's top level (refer to the given JSON schema!). Please try again without acknowledging this message."
                res_messageds.append(
                    {
                        "type": "system",
                        "user_id": user_id,
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
                        "user_id": user_id,
                        "message": {"role": "user", "content": interface_message},
                    }
                )
                self.interface.system_message(interface_message)
                heartbeat_request = True
                function_failed = False

        self.messages_since_last_conscious_memory_write += 1

        ## Step 3: Check memory pressure
        if not is_first_message: # Because first message only accepts a limited range of functions
            had_just_sent_mpw = False
            if (
                not self.memory_pressure_warning_alr_given
                and self.memory.main_ctx_message_seq_no_tokens
                > int(WARNING_TOKEN_FRAC * self.memory.ctx_window)
            ):
                interface_message = f"Warning: Memory pressure has exceeded {WARNING_TOKEN_FRAC*100}% of the context window. Please store important information from your recent conversation history into your core memory or archival storage by calling functions. You should not speak to the user before you finish updating your memory."
                if heartbeat_request:
                    interface_message += " After writing important information into your long-term memory, you should call the necessary functions based on the user's query before responding to the user."
                else:
                    interface_message += " A heartbeat request will be automatically triggered."
                    
                res_messageds.append(
                    {
                        "type": "system",
                        "user_id": user_id,
                        "message": {"role": "user", "content": interface_message},
                    }
                )
                self.interface.memory_message(interface_message)

                self.memory_pressure_warning_alr_given = True
                heartbeat_request = True
                had_just_sent_mpw = True
            elif (
                self.memory.main_ctx_message_seq_no_tokens
                > int(FLUSH_TOKEN_FRAC * self.memory.ctx_window)
            ):
                self.summarise_messages_in_place()
                self.memory_pressure_warning_alr_given = False

            ## Step 4: Check if it has been too long since the agent consciously updated its memory
            if not is_first_message and not had_just_sent_mpw and self.messages_since_last_conscious_memory_write >= WARNING__MESSAGE_SINCE_LAST_CONSCIOUS_MEMORY_EDIT__COUNT:
                interface_message = f"Warning: It has been {self.messages_since_last_conscious_memory_write} messages since you last SUCCESSFULLY edited your memory. Please store important information from your recent conversation history into your core memory or archival storage by calling functions. You should not speak to the user before you finish updating your memory."
                if heartbeat_request:
                    interface_message += " After writing important information into your long-term memory, you should call the necessary functions based on the user's query before responding to the user."
                else:
                    interface_message += " Heartbeat requests will be automatically triggered until you successfully edit your memory."

                res_messageds.append(
                    {
                        "type": "system",
                        "user_id": user_id,
                        "message": {"role": "user", "content": interface_message},
                    }
                )
                self.interface.memory_message(interface_message)

                heartbeat_request = True

            had_just_sent_mpw = False

        ## Step 4: Update memory
        for messaged in res_messageds:
            self.memory.append_messaged_to_fq_and_rs(messaged)

        ## Step 5: Return response
        return res_messageds, heartbeat_request, function_failed

    """
    @staticmethod
    def summary_message_seq(messaged_seq):
        # note: messaged must be in the form {'type': type, 'user_id': user_id, 'message': {'role': role, 'content': content}}
        translated_messages = []
        user_role_buf = []

        for messaged in messaged_seq:
            if messaged["type"] == "system":
                user_role_buf.append(
                    f"❮SYSTEM MESSAGE❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "tool":
                user_role_buf.append(
                    f"❮TOOL MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "user":
                user_role_buf.append(
                    f"❮USER MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                )
            else:
                translated_messages.append(
                    {"role": "user", "content": "\n\n".join(user_role_buf)}
                )
                try:
                    message_content_dict = json5.loads(messaged["message"]["content"])
                except ValueError:
                    assistant_message_content = f"❮ERRONEOUS ASSISTANT MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                else:
                    assistant_message_content = f"❮ASSISTANT MONOLOGUE for conversation with user with id '{messaged['user_id']}'❯ {message_content_dict['thoughts']}"
                    if "function_call" in message_content_dict:
                        assistant_message_content += f"\n\n❮TOOL CALL for conversation with user with id '{messaged['user_id']}'❯ {str(message_content_dict['function_call'])}"

                translated_messages.append(
                    {"role": "assistant", "content": assistant_message_content}
                )
                user_role_buf = []

        if user_role_buf:
            translated_messages.append(
                {"role": "user", "content": "\n\n".join(user_role_buf)}
            )

        return [
            {"role": "system", "content": get_summarise_system_prompt()}
        ] + translated_messages
    """

    @staticmethod
    def summary_message_seq(messaged_seq):
        # note: messaged must be in the form {'type': type, 'user_id': user_id, 'message': {'role': role, 'content': content}}
        translated_messages = []

        for messaged in messaged_seq:
            if messaged["type"] == "system":
                translated_messages.append(
                    f"❮SYSTEM MESSAGE❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "tool":
                translated_messages.append(
                    f"❮TOOL MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                )
            elif messaged["type"] == "user":
                translated_messages.append(
                    f"❮USER MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                )
            else:
                try:
                    message_content_dict = json5.loads(messaged["message"]["content"])
                except ValueError:
                    translated_messages.append(
                        f"❮ERRONEOUS ASSISTANT MESSAGE for conversation with user with id '{messaged['user_id']}'❯ {messaged['message']['content']}"
                    )
                else:
                    translated_messages.append(
                        f"❮ASSISTANT MONOLOGUE for conversation with user with id '{messaged['user_id']}'❯ {message_content_dict['thoughts']}"
                    )
                    if "function_call" in message_content_dict:
                        translated_messages.append(
                            f"❮TOOL CALL for conversation with user with id '{messaged['user_id']}'❯ {str(message_content_dict['function_call'])}"
                        )

        return [
            {"role": "system", "content": get_summarise_system_prompt()},
            {"role": "user", "content": '\n\n'.join(translated_messages)}
        ]

    def summarise_messages_in_place(self):
        if SHOW_DEBUG_MESSAGES:
            print(f"Memory pressure has exceeded {FLUSH_TOKEN_FRAC*100}% of the context window. Flushing message queue...")
            
        self.interface.memory_message(
            f"Memory pressure has exceeded {FLUSH_TOKEN_FRAC*100}% of the context window. Flushing message queue..."
        )
        assert self.memory.main_ctx_message_seq[0]["role"] == "system"

        messages_to_be_summarised = deque()

        while (
            self.memory.main_ctx_message_seq_no_tokens
            > int(TRUNCATION_TOKEN_FRAC * self.memory.ctx_window)
            and len(self.memory.fifo_queue) > LAST_N_MESSAGES_TO_PRESERVE
        ):
            self.memory.no_messages_in_queue -= 1
            messages_to_be_summarised.appendleft(self.memory.fifo_queue.popleft())

        while (
            self.memory.main_ctx_message_seq_no_tokens 
            < int(WARNING_TOKEN_FRAC * self.memory.ctx_window)
            and self.memory.fifo_queue[0]["type"] != "user"
        ):
            self.memory.no_messages_in_queue += 1
            self.memory.fifo_queue.appendleft(messages_to_be_summarised.popleft())

        summary_message_seq = Agent.summary_message_seq(messages_to_be_summarised)
        
        if SHOW_DEBUG_MESSAGES:
            print('Got summary message sequence')

        result = HOST.chat(
            model=self.model_name,
            messages=summary_message_seq,
            options={"num_ctx": self.memory.ctx_window},
        )
        result_content = result["message"]["content"]

        if SHOW_DEBUG_MESSAGES:
            print('Got new recursive summary')

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

        self.memory.write_fq_to_fq_path()
        
        if SHOW_DEBUG_MESSAGES:
            print('summarise_messages_in_place function success!')
