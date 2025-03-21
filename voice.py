import os
import sounddevice as sd
import numpy as np
import wave
import whisper
import requests
import json
import streamlit as st

# Constants
SAMPLERATE = 44100  
DURATION = 5  
FILENAME = "recorded_audio.wav"  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"  

class SpeechToTextBot:
    def __init__(self):
        """Initialize Whisper model and API settings."""
        self.model = whisper.load_model("base")  # Load Whisper model

    def record_audio(self):
        """Records audio from microphone and saves it as a WAV file."""
        with st.spinner("Recording... Speak Now!"):
            audio = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype=np.int16)
            sd.wait()  # Wait for recording to finish

            with wave.open(FILENAME, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLERATE)
                wf.writeframes(audio.tobytes())

    def transcribe_audio(self):
        """Transcribes recorded audio using Whisper."""
        result = self.model.transcribe(FILENAME)
        return result["text"]

    def process_with_groq(self, text):
        """Sends transcribed text to Groq LLM for processing."""
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

# Streamlit UI
st.title("🎙️ Voice to Text Bot with AI Processing")
bot = SpeechToTextBot()

if st.button("🎤 Record Audio"):
    bot.record_audio()
    st.success("Recording complete!")

if os.path.exists(FILENAME):
    if st.button("📝 Transcribe Audio"):
        text = bot.transcribe_audio()
        st.text_area("Transcribed Text:", text, height=100)
        
        if st.button("🤖 record again"):
            response = bot.process_with_groq(text)
            st.text_area("Groq AI Response:", response, height=100)
