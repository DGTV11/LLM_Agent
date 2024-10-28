import json, os, subprocess, urllib.parse
from time import time

from emoji import emojize
from gtts import gTTS
from playsound import playsound
import requests

# Interface
from constants import SERVER_URL_AND_PORT, SHOW_DEBUG_MESSAGES, READ_SENT_MESSAGES

TTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "tmp.mp3")

"""
TTS_SPED_UP_PATH = os.path.join(os.path.dirname(__file__), "tmp2.mp3")

TTS_SPEED_UP_ATEMPO = 1.2
TTS_SPEED_UP_COMMAND = [
    "ffmpeg",
    "-y",
    "-i",
    GTTS_OUTPUT_PATH,
    "-filter:a",
    f"atempo={GTTS_SPEED_UP_ATEMPO}",
    GTTS_SPED_UP_PATH,
]
"""

"""
def read_response(response):
    try:
        tts = gTTS(response, lang="en")
        tts.save(TTS_OUTPUT_PATH)
        subprocess.run(
            TTS_SPEED_UP_COMMAND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        playsound(TTS_SPED_UP_PATH)

        os.remove(TTS_OUTPUT_PATH)
        os.remove(TTS_SPED_UP_PATH)

        return True
    except KeyboardInterrupt:
        if os.path.exists(TTS_OUTPUT_PATH):
            os.remove(TTS_OUTPUT_PATH)
        if os.path.exists(TTS_SPED_UP_PATH):
            os.remove(TTS_SPED_UP_PATH)
        return False
"""


def read_response(response):
    try:
        os.system(
            f"echo '{response.encode('unicode_escape')}' | ../piper/piper --model piper-voice/en_GB-northern_english_male-medium.onnx --output_file {TTS_OUTPUT_PATH}"
        )

        playsound(TTS_OUTPUT_PATH)

        os.remove(TTS_OUTPUT_PATH)

        return True
    except KeyboardInterrupt:
        if os.path.exists(TTS_OUTPUT_PATH):
            os.remove(TTS_OUTPUT_PATH)
        return False


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

        if READ_SENT_MESSAGES:
            if not msg.strip():
                CLIInterface.system_message("Nothing to read!")
                return

            CLIInterface.system_message("Reading response... (Use Ctrl-C to skip)")

            start_time = time()
            successfully_read = read_response(msg)
            end_time = time()

            if successfully_read:
                CLIInterface.system_message(
                    f"Successfully read response in {round(end_time - start_time, 2)}s"
                )

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


# Main loop
if __name__ == "__main__":
    conv_name = None

    has_prev_conv = False
    if ps_folders := requests.get(
        urllib.parse.urljoin(SERVER_URL_AND_PORT, "/conversation-ids")
    ).json()["conv_ids"]:
        use_existing_conv = (
            True if input("Use existing conv? (y/n) ").strip().lower() == "y" else False
        )
        if use_existing_conv:
            existing_conv_enum = list(enumerate(ps_folders, start=1))
            existing_conv_enum_dict = {idx: name for idx, name in existing_conv_enum}

            print(
                "Please choose the number corresponding to your chosen existing conversation"
            )
            for folder_idx, folder_name in existing_conv_enum:
                print(f"{folder_idx} -> {folder_name}")

            while True:
                try:
                    chosen_folder_num = int(input("> "))
                    if chosen_folder_num not in existing_conv_enum_dict:
                        print(
                            "Input is not in the given conversation numbers! Please try again!"
                        )
                        continue
                    break
                except ValueError:
                    print("Input is not an integer! Please try again!")

            conv_name = existing_conv_enum_dict[chosen_folder_num]

            has_prev_conv = True

    if not conv_name:
        agent_persona_folder = requests.get(
            urllib.parse.urljoin(SERVER_URL_AND_PORT, "/personas/agents")
        ).json()["persona_names"]
        human_persona_folder = requests.get(
            urllib.parse.urljoin(SERVER_URL_AND_PORT, "/personas/humans")
        ).json()["persona_names"]

        # Choose agent persona
        agent_persona_folder_enum = list(enumerate(agent_persona_folder, start=1))
        agent_persona_folder_enum_dict = {
            idx: name for idx, name in agent_persona_folder_enum
        }

        print("Please choose the number corresponding to your chosen agent persona")
        for persona_idx, persona_name in agent_persona_folder_enum:
            print(f"{persona_idx} -> {persona_name.split('.')[0]}")

        while True:
            try:
                chosen_agent_persona_num = int(input("> "))
                if chosen_agent_persona_num not in agent_persona_folder_enum_dict:
                    print(
                        "Input is not in the given persona numbers! Please try again!"
                    )
                    continue
                break
            except ValueError:
                print("Input is not an integer! Please try again!")

        chosen_agent_persona = agent_persona_folder_enum_dict[chosen_agent_persona_num]

        # Choose human persona
        human_persona_folder_enum = list(enumerate(human_persona_folder, start=1))
        human_persona_folder_enum_dict = {
            idx: name for idx, name in human_persona_folder_enum
        }

        print("Please choose the number corresponding to your chosen human persona")
        for persona_idx, persona_name in human_persona_folder_enum:
            print(f"{persona_idx} -> {persona_name.split('.')[0]}")

        while True:
            try:
                chosen_human_persona_num = int(input("> "))
                if chosen_human_persona_num not in agent_persona_folder_enum_dict:
                    print(
                        "Input is not in the given persona numbers! Please try again!"
                    )
                    continue
                break
            except ValueError:
                print("Input is not an integer! Please try again!")

        chosen_human_persona = human_persona_folder_enum_dict[chosen_agent_persona_num]

        # Create new agent
        conv_name = requests.post(
            urllib.parse.urljoin(SERVER_URL_AND_PORT, "/agent"),
            json={
                "agent_persona_name": chosen_agent_persona,
                "human_persona_name": chosen_human_persona,
            },
        ).json()["conv_name"]

    # Main conversation loop
    try:
        s = requests.Session()
        with s.post(
            urllib.parse.urljoin(SERVER_URL_AND_PORT, "/messages/send/first-message"),
            json={
                "conv_name": conv_name,
                "user_id": 1,
                "message": f'User with id \'{1}\' entered the conversation. You should greet the user and start the conversation based on your persona\'s specifications{" and your previous conversation" if has_prev_conv else ""}.',
            },
            stream=True,
        ) as resp:
            for line in resp.iter_lines():
                json_obj = json.loads(line)
                if td := json_obj.get("total_duration", None):
                    print(f"Time taken for agent response: {td}s")
                    print("\n\n", end="")
                else:
                    server_message_stack = json_obj["server_message_stack"]
                    for message in server_message_stack:
                        getattr(CLIInterface, message["type"])(**message["arguments"])

                    current_ctx_token_count = json_obj["ctx_info"][
                        "current_ctx_token_count"
                    ]
                    ctx_window = json_obj["ctx_info"]["ctx_window"]
                    print(
                        f"Context info: {current_ctx_token_count}/{ctx_window} tokens ({round((current_ctx_token_count/ctx_window)*100, 2)}%)"
                    )

                    print(f"Time taken for agent step: {json_obj['duration']}s")
                    print("\n\n", end="")

        print("/help for commands, /exit to exit")
        while True:
            input_message = input(
                f"{current_ctx_token_count}/{ctx_window} tokens ({round((current_ctx_token_count/ctx_window)*100, 2)}%) > "
            )

            match input_message.strip():
                case "/help":
                    print("HELP")
                    print("/exit -> exit conversation")
                case "/exit":
                    break
                case _:
                    s = requests.Session()
                    with s.post(
                        urllib.parse.urljoin(SERVER_URL_AND_PORT, "/messages/send"),
                        json={
                            "conv_name": conv_name,
                            "user_id": 1,
                            "message": input_message,
                        },
                        stream=True,
                    ) as resp:
                        for line in resp.iter_lines():
                            json_obj = json.loads(line)
                            if td := json_obj.get("total_duration", None):
                                print(f"Time taken for agent response: {td}s")
                                print("\n\n", end="")
                            else:
                                server_message_stack = json_obj["server_message_stack"]
                                for message in server_message_stack:
                                    getattr(CLIInterface, message["type"])(
                                        **message["arguments"]
                                    )

                                current_ctx_token_count = json_obj["ctx_info"][
                                    "current_ctx_token_count"
                                ]
                                ctx_window = json_obj["ctx_info"]["ctx_window"]
                                print(
                                    f"Context info: {current_ctx_token_count}/{ctx_window} tokens ({round((current_ctx_token_count/ctx_window)*100, 2)}%)"
                                )

                                print(
                                    f"Time taken for agent step: {json_obj['duration']}s"
                                )
                                print("\n\n", end="")
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Exiting...")
    except Exception as e:
        print("Exiting due to error:", e)
    finally:
        json_obj = requests.post(
            urllib.parse.urljoin(SERVER_URL_AND_PORT, "/messages/send/no-heartbeat"),
            json={
                "conv_name": conv_name,
                "user_id": 1,
                "message": f"User with id '{1}' exited the conversation",
            },
            stream=True,
        ).json()
        server_message_stack = json_obj["server_message_stack"]
        for message in server_message_stack:
            getattr(CLIInterface, message["type"])(**message["arguments"])

        current_ctx_token_count = json_obj["ctx_info"]["current_ctx_token_count"]
        ctx_window = json_obj["ctx_info"]["ctx_window"]
        print(
            f"Context info: {current_ctx_token_count}/{ctx_window} tokens ({round((current_ctx_token_count/ctx_window)*100, 2)}%)"
        )

        print("\n\n", end="")
