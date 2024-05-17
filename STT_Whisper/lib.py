from os import path, remove
import argparse

import STT_Whisper.record as record
import STT_Whisper.transcribe as transcribe

WAVE_OUTPUT_PATH = path.join(path.dirname(__file__), "tmp.wav")


def whisper_microphone_transcribe(model_name="base"):
    # record audio from the microphone
    text_input = record.record_audio(WAVE_OUTPUT_PATH)

    if text_input is not None:
        return text_input

    # delete the temporary audio file
    remove(WAVE_OUTPUT_PATH)

    # transcribe the audio
    result = transcribe.transcribe_audio(WAVE_OUTPUT_PATH, model_name=model_name)

    # return the result
    return result.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper")
    parser.add_argument(
        "--model", type=str, default="base", help="The name of the Whisper model to use"
    )
    args = parser.parse_args()

    result = whisper_microphone_transcribe(model_name=args.model)

    print(result)
