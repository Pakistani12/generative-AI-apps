import streamlit as st
import torch
from TTS.api import TTS 
from io import BytesIO
import base64
import scipy.io.wavfile as wavfile
import numpy as np


@st.cache_resource
def load_model():
    model_name = "tts_models/en/ljspeech/tacotron2-DDC" 
    tts = TTS(model_name).to("cpu") 
    return tts

def synthesize_audio(text, tts):
    try:
        
        wav_output = tts.tts(text)

        
        audio_data = np.array(wav_output, dtype=np.float32)
        return audio_data
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def audio_download_link(audio_data, filename="output.wav"):
    buffer = BytesIO()
    
   
    sample_rate = 22050 
    audio_data = np.int16(audio_data * 32767)  
    
    
    wavfile.write(buffer, sample_rate, audio_data)
    buffer.seek(0)

    
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download Audio</a>'
    return href, buffer


def main():
    st.title("Text-to-Audio Generation ")
    st.markdown("Enter your text, and the app will convert it into lifelike speech!")

    tts = load_model()

 
    text = st.text_area("Enter text for audio generation:")

    if st.button("Generate Audio"):
        if text.strip():
            with st.spinner("Generating audio..."):
                audio_data = synthesize_audio(text, tts)

                if audio_data is not None:
                    
                    _, buffer = audio_download_link(audio_data)
                    st.audio(buffer, format="audio/wav")

                 
                    st.markdown(audio_download_link(audio_data)[0], unsafe_allow_html=True)
        else:
            st.warning("Please enter some text before generating audio.")

if __name__ == "__main__":
    main()
