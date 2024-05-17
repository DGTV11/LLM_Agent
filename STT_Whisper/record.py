import wave

import pyaudio
from pynput import keyboard

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5


def record_audio(out_filename):
    audio = pyaudio.PyAudio()

    # Check if there are any mics availiable
    try:
        audio.get_default_input_device_info()
    except:
        print("No mics availiable, falling back on text input")
        return input("Input: ")

    # start Recording
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    print("Recording...")

    frames = []

    def on_press(key):
        if key == keyboard.Key.space:
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while listener.is_alive():
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(out_filename, "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b"".join(frames))
    waveFile.close()

    return None
