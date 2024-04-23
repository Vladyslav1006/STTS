# main.py
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import os
from groq_translation import groq_translate
from gtts import gTTS

# Set page config
st.set_page_config(page_title='BABEL 24 - RT Speech Translator', page_icon='🎤')

# Set page title
st.title('BABEL 24 - RT Speech Translator')

# Load whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))
# model = WhisperModel("base", device="cuda")


# Speech to text
def speech_to_text(audio_chunk):
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    speech_text = " ".join([segment.text for segment in segments])
    return speech_text

# Text to speech
def text_to_speech(translated_text, language):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text, lang=language)
    my_obj.save(file_name)
    return file_name

languages = {
    "English": "en", 
   "German": "de",
   "Italian": "it",
   "Japanese": "ja",
   "Chinese": "zh",
}

# Language selection for input
input_language = st.selectbox(
   "Language of the input speech:",
   languages,
   placeholder="Select input language...",
)

# Language selection for translation
target_language = st.selectbox(
   "Language to translate to:",
   languages,
   placeholder="Select target language...",
)



# Record audio
audio_bytes = audio_recorder()
if audio_bytes and target_language and input_language:
    # Display audio player
    st.audio(audio_bytes, format="audio/wav")

    # Save audio to file
    with open('audio.wav', mode='wb') as f:
        f.write(audio_bytes)

    # Speech to text
    st.divider()
    with st.spinner('Transcribing...'):
        text = speech_to_text('audio.wav')
    st.subheader('Transcribed Text')
    st.write(text)

    # Groq translation
    st.divider()
    with st.spinner('Translating...'):
        translation = groq_translate(text, languages[input_language], target_language)
    st.subheader(f'Translated Text to {target_language}')
    st.write(translation.text)

    # Text to speech
    audio_file = text_to_speech(translation.text, languages[target_language])
    st.audio(audio_file, format="audio/mp3")
