import os
from google.cloud import speech
import pyaudio
import wave

# Function to record audio from the microphone
def record_audio(filename="output.wav", duration=5, rate=16000):
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16-bit audio format
    channels = 1  # Mono audio
    rate = rate  # Sampling rate

    p = pyaudio.PyAudio()

    print("Recording... Speak Now!")
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    return filename

# Function to transcribe audio using Google Cloud Speech-to-Text API
def transcribe_audio(filename):
    client = speech.SpeechClient()

    # Load the audio file
    with open(filename, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    # Perform speech-to-text
    print("Transcribing audio...")
    response = client.recognize(config=config, audio=audio)

    # Extract and return the transcription
    for result in response.results:
        print("Transcription: {}".format(result.alternatives[0].transcript))
        return result.alternatives[0].transcript

    return "No transcription available."

if __name__ == "__main__":
    # Record audio and save it to a file
    audio_file = record_audio()

    # Transcribe the recorded audio
    transcription = transcribe_audio(audio_file)
    print("\nFinal Transcription:")
    print(transcription)