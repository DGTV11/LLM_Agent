import os, subprocess
from time import time

from emoji import emojize

from llm_os.constants import SHOW_DEBUG_MESSAGES


class CLIInterface:
    @staticmethod
    def warning_message(msg: str, end="\n"):
        print(emojize(f":warning: {msg}"), end=end, flush=True)

    @staticmethod
    def debug_message(msg: str, end="\n"):
        print(emojize(f":lady_beetle: {msg}"), end=end, flush=True)

    @staticmethod
    def internal_monologue(msg: str, internal_monologue_part: str, end="\n"):
        match internal_monologue_part:  # ["user_emotion_analysis", "inner_emotions", "long_term_planning", "conversation_planning", "auxiliary_reasoning", "function_call_planning"]
            case "user_emotion_analysis":
                emoji_text = "heart"
            case "inner_emotions":
                emoji_text = "grey_heart"
            case "long_term_planning":
                emoji_text = "tear-off_calendar"
            case "conversation_planning":
                emoji_text = "clipboard"
            case "auxiliary_reasoning":
                emoji_text = "thought_balloon"
            case "function_call_planning":
                emoji_text = "wrench"
            case _:
                emoji_text = "thought_balloon"

        print(emojize(f":{emoji_text}: {msg}"), end=end, flush=True)

    @staticmethod
    def assistant_message(msg: str, end="\n"):
        print(emojize(f":robot: {msg}"), end=end, flush=True)

    @staticmethod
    def memory_message(msg: str, end="\n"):
        print(emojize(f":brain: {msg}"), end=end, flush=True)

    @staticmethod
    def system_message(msg: str, end="\n"):
        print(emojize(f":desktop_computer: {msg}"), end=end, flush=True)

    @staticmethod
    def user_message(msg: str, end="\n"):
        print(emojize(f":person: {msg}"), end=end, flush=True)

    @staticmethod
    def function_call_message(func_name: str, func_args: dict, end="\n"):
        if SHOW_DEBUG_MESSAGES:
            func_args = func_args.copy()
            if func_args.get("self", None):
                del func_args["self"]

            print(
                emojize(
                    f":high_voltage: Called function '{func_name}' with arguments {func_args}"
                ),
                end=end,
                flush=True,
            )
        else:
            print(
                emojize(f":high_voltage: Called function '{func_name}'"),
                end=end,
                flush=True,
            )

    @staticmethod
    def function_res_message(msg: str, has_error: bool, end="\n"):
        print(
            emojize(f':{"red_circle" if has_error else "green_circle"}: {msg}'),
            end=end,
            flush=True,
        )


class ServerInterface:
    def __init__(self):
        self.server_message_stack = []

    def warning_message(self, msg: str, end="\n"):
        self.server_message_stack.append(
            {"type": "warning_message", "arguments": {"msg": msg, "end": end}}
        )

    def debug_message(self, msg: str, end="\n"):
        self.server_message_stack.append(
            {"type": "debug_message", "arguments": {"msg": msg, "end": end}}
        )

    def internal_monologue(self, msg: str, internal_monologue_part: str, end="\n"):
        self.server_message_stack.append(
            {
                "type": "internal_monologue",
                "arguments": {
                    "msg": msg,
                    "internal_monologue_part": internal_monologue_part,
                    "end": end,
                },
            }
        )

    def assistant_message(self, msg: str, end="\n"):
        self.server_message_stack.append(
            {"type": "assistant_message", "arguments": {"msg": msg, "end": end}}
        )

    def memory_message(self, msg: str, end="\n"):
        self.server_message_stack.append(
            {"type": "memory_message", "arguments": {"msg": msg, "end": end}}
        )

    def system_message(self, msg: str, end="\n"):
        self.server_message_stack.append(
            {"type": "system_message", "arguments": {"msg": msg, "end": end}}
        )

    def user_message(self, msg: str, end="\n"):
        self.server_message_stack.append(
            {"type": "user_message", "arguments": {"msg": msg, "end": end}}
        )

    def function_call_message(self, func_name: str, func_args: dict, end="\n"):
        func_args = func_args.copy()
        if func_args.get("self", None):
            del func_args["self"]

        self.server_message_stack.append(
            {
                "type": "function_call_message",
                "arguments": {
                    "func_name": func_name,
                    "func_args": func_args,
                    "end": end,
                },
            }
        )

    def function_res_message(self, msg: str, has_error: bool, end="\n"):
        self.server_message_stack.append(
            {
                "type": "function_res_message",
                "arguments": {"msg": msg, "has_error": has_error, "end": end},
            }
        )
