import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import io

# Configure NVIDIA API
API_KEY = "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"
NVIDIA_API_URL = "https://api.nvidia.com/asr/canary-1b"  # Replace with actual endpoint

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    print("Recording... Speak Now!")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return samplerate, audio_data

# Function to save audio as WAV format
def save_as_wav(samplerate, audio_data):
    with io.BytesIO() as wav_buffer:
        wav.write(wav_buffer, samplerate, audio_data)
        return wav_buffer.getvalue()

# Function to transcribe audio using NVIDIA Canary-1B-ASR
def transcribe_audio(audio_bytes):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "audio/wav"
    }
    
    response = requests.post(NVIDIA_API_URL, headers=headers, data=audio_bytes)
    
    if response.status_code == 200:
        return response.json().get("transcription", "No transcription found.")
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    samplerate, audio_data = record_audio()
    audio_wav = save_as_wav(samplerate, audio_data)
    transcription = transcribe_audio(audio_wav)
    
    print("\nTranscribed Text:")
    print(transcription)
