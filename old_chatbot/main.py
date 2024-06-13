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

LSA_SYSTEM = """# MISSION
You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation of Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human. Use complete sentences. ONLY write the statements - nothing more."""

wrap_message = lambda role, content: {"role": role, "content": content}

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

def query(model_name, input_messages, query):
    generated_result = False
    try:
        if "mistral" in model_name:
            model_display_name = "Mistral"
            model_architecture = "Mistral"
        elif "phi3" in model_name:
            model_display_name = "Phi-3"
            model_architecture = "LLaMA"
        elif "openchat" in model_name:
            model_display_name = "OpenChat"
            model_architecture = "LLaMA"
        elif "llama3" in model_name:
            model_display_name = "Llama-3"
            model_architecture = "LLaMA"
        else:
            raise NotImplementedError("Model not supported")

        print(emojize(f":person: User: {query}"))

        messages = input_messages.copy()

        #*Generate LSA context
        stream_not_started = True
        lsa_context = ""
        for i, message, is_streaming in lsa_query(
            query, 
            model=model_name, 
            chatbot=HOST.chat, 
            model_name=model_display_name, 
            model_architecture=model_architecture
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
                Respond to the query DIRECTLY based on ONLY THE MOST RELEVANT PARTS of the following context as if you are SPEAKING to me (do not acknowledge that you have discussed the query in the following context){' (you may reference the above messages as needed to aid in your response)' if len(messages) >= 2 else ''}:
                {lsa_context}""",
            )
        )

        #*Generate response
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

        #*Read response
        print("Reading response... (Use Ctrl-C to skip)")

        start_time = time()
        successfully_read = read_response(res_stream)
        end_time = time()

        if successfully_read:
            print(f"Successfully read response in {round(end_time - start_time, 2)}s")

        #*Summarise LSA context into an SPR
        print("Summarising Latent Space Activation context... (use Ctrl-C to skip)")
        print(emojize(":memo: Summarised context:"))

        start_time = time()
        result = HOST.chat(
            model="phi3" if "phi3" in model_name else 'mistral',
            messages=[wrap_message("system", LSA_SYSTEM), wrap_message("user", lsa_context)],
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
                Respond to the query DIRECTLY based on ONLY THE MOST RELEVANT PARTS of the following context as if you are SPEAKING to me (do not acknowledge that you have discussed the query in the following context){' (you may reference the above messages as needed to aid in your response)' if len(messages) >= 2 else ''} (context has been summarized for brevity):
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
        model_name, messages, "Hi there! Could you introduce yourself?"
    )
    end_time = time()
    print("\n\n")

    if res_stream is not None:
        print(f"Responded in {round(end_time - start_time, 2)}s")

    while True:
        # Record user input
        result = whisper_microphone_transcribe(model_name=whisper_model_name)

        start_time = time()
        messages, res_stream = query(model_name, messages, result)
        end_time = time()
        print("\n\n")

        if res_stream is not None:
            print(f"Responded in {round(end_time - start_time, 2)}s")
