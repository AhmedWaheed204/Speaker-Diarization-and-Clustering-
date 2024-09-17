import os
import streamlit as st
from utils import load_models, load_embedding, process_audio, plot_audio_segs
import librosa

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
auth_token = "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"
model_size = "tiny"

# Load models and embeddings
embedding_model = load_embedding(auth_token)
pipe, whisper = load_models(auth_token, model_size)

# Streamlit app title and description
st.title("Transcription and Speaker Diarization")

with st.expander("Record"):
    uploaded_file = st.file_uploader("Upload file (.wav):")
    if uploaded_file:
        # Load audio file
        audio, sr = librosa.load(uploaded_file, sr=None)

        # Process audio
        segs, speakers = process_audio(audio, sr, pipe, whisper, embedding_model)

        # Display results
        st.write("### Speaker Diarization and Transcription Results")
        for seg in segs:
            st.write(f"{seg.speaker}: {seg.start_sec:.2f}s - {seg.end_sec:.2f}s")
            st.write(seg.text)
            st.write("---")

        # Plot the audio segments with speaker labels
        fig = plot_audio_segs(audio, sr, speakers, segs)
        st.pyplot(fig)
