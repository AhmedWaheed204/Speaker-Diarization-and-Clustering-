import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dataclasses import dataclass
import torch
import streamlit as st
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import numpy as np
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
import librosa

@dataclass
class AudioSeg:
    start_sec: float
    end_sec: float
    speaker: str
    text: str = ""
    language: str = "en"
    sr: int = 16000

    @property
    def start(self):
        return int(self.sr * self.start_sec)

    @property
    def end(self):
        return int(self.sr * self.end_sec)

    def __repr__(self):
        return f"Segment from:{self.start_sec:.3f} to {self.end_sec:.3f}, speaker:{self.speaker}, text:{self.text}"    

@st.cache_resource()
def load_models(auth_token, model_size="tiny"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "float32"
    whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token).to(torch_device)
    return pipe, whisper_model

def load_embedding(auth_token):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        embedding_model = Inference("pyannote/embedding", window="whole", use_auth_token=auth_token).to(torch_device)
        if embedding_model is None:
            raise ValueError("Failed to load the embedding model. The model is None.")
        print(f"Embedding model loaded successfully on {torch_device}.")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

def plot_audio_segs(audio, sr, speakers, segs, figsize=(25, 7), fontsize=20):
    colors = list(matplotlib.colors.CSS4_COLORS.keys())
    speaker_color_map = dict(zip(speakers, np.random.choice(colors, len(speakers), replace=False)))

    fig, ax = plt.subplots(figsize=figsize)
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio)
    for seg in segs:
        color = speaker_color_map[seg.speaker]
        ax.axvspan(seg.start_sec, seg.end_sec, color=color, alpha=0.5)
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=sp) for sp, color in speaker_color_map.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=fontsize)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.grid(True)
    return fig

def process_audio(audio, sr, pipe, whisper_model, embedding_model):
    duration = len(audio) / sr
    
    if duration <= 5:  # Short audio (5 seconds or less)
        return process_short_audio(audio, sr, pipe, whisper_model)
    else:  # Long audio (more than 5 seconds)
        return process_long_audio(audio, sr, pipe, whisper_model, embedding_model)

def process_short_audio(audio, sr, pipe, whisper_model):
    diarization = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr})
    segments, info = whisper_model.transcribe(audio, language="en")
    text = "".join([s.text for s in segments])
    
    segs = []
    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segs.append(AudioSeg(turn.start, turn.end, speaker, text, sr=sr))
        speakers.add(speaker)
    
    return segs, speakers

def process_long_audio(audio, sr, pipe, whisper_model, embedding_model):
    # Segment audio into 2-5 second chunks
    chunk_duration = 3  # You can adjust this between 2-5 seconds
    chunk_samples = int(chunk_duration * sr)
    chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]
    
    # Extract embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        embedding = extract_embedding(chunk, sr, embedding_model)
        embeddings.append(embedding)
    
    # Cluster embeddings
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5).fit(embeddings)
    
    # Process chunks based on clustering
    segs = []
    speakers = set()
    for i, (chunk, label) in enumerate(zip(chunks, clustering.labels_)):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, len(audio) / sr)
        speaker = f"Speaker_{label}"
        segments, info = whisper_model.transcribe(chunk, language="en")
        text = "".join([s.text for s in segments])
        segs.append(AudioSeg(start_time, end_time, speaker, text, sr=sr))
        speakers.add(speaker)
    
    return segs, speakers

def extract_embedding(audio_chunk, sr, embedding_model):
    if embedding_model is None:
        raise ValueError("Embedding model is not loaded properly. Please check your setup.")

    if len(audio_chunk.shape) > 1:
        audio_chunk = np.mean(audio_chunk, axis=1)

    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)
    embedding = embedding_model({"waveform": audio_tensor, "sample_rate": sr})

    if not isinstance(embedding, np.ndarray):
        embedding = embedding.cpu().detach().numpy()

    embedding = np.squeeze(embedding)
    return embedding
