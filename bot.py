import os
import sounddevice as sd
import numpy as np
import wave
import whisper
import requests
import json


SAMPLERATE = 44100  
DURATION = 5  
FILENAME = "recorded_audio.wav"  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"  

class SpeechToTextBot:
    def __init__(self):
        """Initialize Whisper model and API settings."""
        # print("Loading Whisper model...")
        self.model = whisper.load_model("base")  
        # print("Model loaded.")

    def record_audio(self):
        """Records audio from microphone and saves it as a WAV file."""
        print("Recording... Speak Now!")
        audio = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype=np.int16)
        sd.wait()  # Wait for recording to finish

        with wave.open(FILENAME, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLERATE)
            wf.writeframes(audio.tobytes())

        print("Recording complete.")

    def transcribe_audio(self):
        """Transcribes recorded audio using Whisper."""
        # print("Transcribing...")
        result = self.model.transcribe(FILENAME)
        return result["text"]

    def process_with_groq(self, text):
        """Sends transcribed text to Groq LLM for processing."""
        # print(f"Processing with Groq: {text}")
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.5
        }

        response = requests.post(GROQ_ENDPOINT, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    bot = SpeechToTextBot()
    bot.record_audio()
    text = bot.transcribe_audio()
    print("\n Text:", text)

    response = bot.process_with_groq(text)
