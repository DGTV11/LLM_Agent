import os, subprocess
from time import time

from emoji import emojize
from gtts import gTTS
from playsound import playsound

from llm_os.constants import SHOW_DEBUG_MESSAGES, READ_SENT_MESSAGES

GTTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "tmp.mp3")
GTTS_SPED_UP_PATH = os.path.join(os.path.dirname(__file__), "tmp2.mp3")

GTTS_SPEED_UP_ATEMPO = 1.2
GTTS_SPEED_UP_COMMAND = [
    "ffmpeg",
    "-y",
    "-i",
    GTTS_OUTPUT_PATH,
    "-filter:a",
    f"atempo={GTTS_SPEED_UP_ATEMPO}",
    GTTS_SPED_UP_PATH,
]

def read_response(response):
    try:
        tts = gTTS(response, lang="en")
        tts.save(GTTS_OUTPUT_PATH)
        subprocess.run(
            GTTS_SPEED_UP_COMMAND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        playsound(GTTS_SPED_UP_PATH)

        os.remove(GTTS_OUTPUT_PATH)
        os.remove(GTTS_SPED_UP_PATH)

        return True
    except KeyboardInterrupt:
        if os.path.exists(GTTS_OUTPUT_PATH):
            os.remove(GTTS_OUTPUT_PATH)
        if os.path.exists(GTTS_SPED_UP_PATH):
            os.remove(GTTS_SPED_UP_PATH)
        return False

class CLIInterface:
    @staticmethod
    def warning_message(msg: str, end="\n"):
        print(emojize(f":warning: {msg}"), end=end, flush=True)

    @staticmethod
    def debug_message(msg: str, end="\n"):
        print(emojize(f":lady_beetle: {msg}"), end=end, flush=True)

    @staticmethod
    def internal_monologue(msg: str, end="\n"):
        print(emojize(f":thought_balloon: {msg}"), end=end, flush=True)

    @staticmethod
    def assistant_message(msg: str, end="\n"):
        print(emojize(f":robot: {msg}"), end=end, flush=True)

        CLIInterface.system_message("Reading response... (Use Ctrl-C to skip)")

        start_time = time()
        successfully_read = read_response(msg)
        end_time = time()

        if successfully_read:
            CLIInterface.system_message(f"Successfully read response in {round(end_time - start_time, 2)}s")

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

    @staticmethod
    def append_to_message(msg: str):
        print(msg, end="", flush=True)
