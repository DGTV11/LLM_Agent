from emoji import emojize

from llm_os.constants import SHOW_DEBUG_MESSAGES

class CLIInterface:
    @staticmethod
    def warning_message(msg: str, end="\n"):
        print(emojize(f":warning: {msg}"), end=end, flush=True)

    @staticmethod
    def debug_message(msg: str, end="\n"):
        if SHOW_DEBUG_MESSAGES:
            print(emojize(f":lady_beetle: {msg}"), end=end, flush=True)

    @staticmethod
    def internal_monologue(msg: str, end="\n"):
        print(emojize(f":thought_balloon: {msg}"), end=end, flush=True)

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
        print(
            emojize(
                f':high_voltage: Called function {func_name} with arguments {func_args}))'
            ),
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

    @staticmethod
    def append_to_message(msg: str):
        print(msg, end="", flush=True)
