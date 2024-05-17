import os, subprocess, argparse
from time import time

from emoji import emojize
from gtts import gTTS
from playsound import playsound

import model_regen as mrh
from STT_Whisper.lib import whisper_microphone_transcribe
from latent_space_activation.technique01_dialog import lsa_query
from config import CONFIG
from host import HOST

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

wrap_message = lambda role, content: {"role": role, "content": content}


def query(model_name, input_messages, query):
    generated_result = False
    try:
        print(emojize(f":person: User: {query}"))

        messages = input_messages.copy()

        stream_not_started = True
        lsa_context = ""
        for i, message, is_streaming in lsa_query(
            query, model=model_name, chatbot=HOST.chat
        ):
            if i % 2 == 0:
                lsa_context += f"Interrogation thought: {message['content']}\n"
                print(
                    emojize(
                        f":thought_balloon: Interrogation thought: {message['content']}\n"
                    ),
                    end="",
                )
                start_time = time()
            else:
                if not is_streaming:
                    lsa_context += f"Response thought:\n{message['content']}\n\n"
                    end_time = time()
                    stream_not_started = True
                    print(
                        f"\n\nGenerated response thought in {round(end_time-start_time, 2)}s"
                    )
                    continue
                if stream_not_started:
                    print(emojize(":thought_balloon: Response Thought:"))
                    stream_not_started = False

                print(message, end="", flush=True)

        messages.append(
            wrap_message(
                "user",
                f"""
                Query: {query}
                Respond to the query DIRECTLY based on ONLY THE MOST RELEVANT PARTS of the following context as if you are SPEAKING to me (do not acknowledge that you have discussed the query in the following context):
                {lsa_context}""",
            )
        )

        print("Generating response...")
        print(emojize(":robot: Assistant: "), end="")

        start_time = time()
        result = HOST.chat(
            model=model_name, messages=messages, stream=True, keep_alive=30
        )
        res_stream = ""
        for chunk in result:
            res_stream += chunk["message"]["content"]
            print(chunk["message"]["content"], end="", flush=True)
        end_time = time()

        print(f"\n\nGenerated response in {round(end_time-start_time, 2)}s")

        generated_result = True

        print("Summarising Latent Space Activation context...")
        print(emojize(":memo: Summarised context:\n"), end="")

        start_time = time()
        result = HOST.chat(
            model=model_name.replace("assistant", "spr"),
            messages=[wrap_message("user", lsa_context)],
            stream=True,
            keep_alive=30,
        )
        spr_lsa_context = ""
        for chunk in result:
            spr_lsa_context += chunk["message"]["content"]
            print(chunk["message"]["content"], end="", flush=True)
        end_time = time()

        print(
            f"\n\nSummarised Latent Space Activation context in {round(end_time-start_time, 2)}s"
        )

        messages.pop()
        messages.append(
            wrap_message(
                "user",
                f"""
                Query: {query}
                Respond to the query DIRECTLY based on ONLY THE MOST RELEVANT PARTS of the following context as if you are SPEAKING to me (do not acknowledge that you have discussed the query in the following context) (context has been summarized for brevity):
                {spr_lsa_context}""",
            )
        )
        messages.append(wrap_message("assistant", res_stream))

        return messages, res_stream
    except KeyboardInterrupt:
        if generated_result:
            messages.append(wrap_message("assistant", res_stream))
            return messages, res_stream
        return input_messages, None


def read_query(query):
    try:
        tts = gTTS(query, lang="en")
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


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Speech-to-Speech Chatbot")
    # parser.add_argument(
    #     "--local-ollama", action="store_true", help="Use local ollama server at http://127.0.0.1:11434", dest="use_local_ollama"
    # )
    # args = parser.parse_args()

    for file in os.listdir(os.path.join(os.path.dirname(__file__), "modelfiles")):
        if file[0] in [".", "_"]:
            continue

        print(f"Checking if the '{file}' model exists...")

        if f"{file}:latest" not in [model["name"] for model in HOST.list()["models"]]:
            mrh.gen_model_from_modelfile(
                f"{file}",
                __file__,
                lambda: print(f"Creating the '{file}' model..."),
                lambda: print(f"'{file}' model created!"),
            )

    model_name = CONFIG["model_name"]

    print(f"Using the '{model_name}' model...")

    whisper_model_name = CONFIG["whisper_model_name"]

    print(f"Using the '{whisper_model_name}' Whisper model...")

    messages = []

    start_time = time()
    messages, res_stream = query(
        model_name, messages, "Hello! Could you introduce yourself?"
    )
    end_time = time()
    print("\n\n")

    if res_stream is not None:
        print(f"Responded in {round(end_time - start_time, 2)}s")

        start_time = time()
        success = read_query(res_stream)
        end_time = time()

        if success:
            print(f"Read in {round(end_time - start_time, 2)}s")

    while True:
        # Record user input
        result = whisper_microphone_transcribe(model_name=whisper_model_name)

        start_time = time()
        messages, res_stream = query(model_name, messages, result)
        end_time = time()
        print("\n\n")

        if res_stream is not None:
            print(f"Responded in {round(end_time - start_time, 2)}s")

            start_time = time()
            success = read_query(res_stream)
            end_time = time()

            if success:
                print(f"Read in {round(end_time - start_time, 2)}s")
