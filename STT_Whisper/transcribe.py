import whisper


def transcribe_audio(audio_file, model_name="base"):
    model = whisper.load_model(model_name)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language and initalise decoding options
    if ".en" not in model_name:
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        options = whisper.DecodingOptions()
    else:
        options = whisper.DecodingOptions(language="en")

    # decode the audio
    result = whisper.decode(model, mel, options)

    # return the result (text at result.text)
    return result
