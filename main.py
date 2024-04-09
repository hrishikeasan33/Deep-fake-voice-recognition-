import streamlit as st
import sys
import os
import librosa
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from tensorflow.keras.models import load_model

class DeepFakeVoiceDetector:
    def __init__(self):
        self.model = load_model('gru_final.h5')
        self.audio_path = None

    def preprocess_audio(self, audio_file_path, max_length=500):
        try:
            audio, _ = librosa.load(audio_file_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)

            if mfccs.shape[1] < max_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[:, :max_length]

            mfccs_normalized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
            return mfccs_normalized
        except Exception as e:
            st.error(f"Error encountered while processing file: {audio_file_path}")
            return None
    
    def detect_voice(self):
        if self.audio_path:
            preprocessed_audio = self.preprocess_audio(self.audio_path)
            preprocessed_audio = np.expand_dims(preprocessed_audio, axis=0)

            prediction = self.model.predict(preprocessed_audio)

            if prediction > 0.5:
                return 'Fake voice detected'
            else:
                return 'Real voice detected'
        else:
            return 'Please select an audio file first'

def main():
    st.title("Deep Fake Voice Detector")

    detector = DeepFakeVoiceDetector()

    audio_file = st.file_uploader("Upload Audio", type=["wav"])
    if audio_file:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())
        detector.audio_path = "uploaded_audio.wav"

    if st.button("Detect Voice"):
        result = detector.detect_voice()
        st.write(result)

if __name__ == "__main__":
    main()
