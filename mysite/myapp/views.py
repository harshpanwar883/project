from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from pathlib import Path

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import requests as r
import json

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

from pydub import AudioSegment


BASE_DIR = Path(__file__).resolve().parent.parent

def index(request):
    def extract_feature(file_name, mfcc, chroma, mel):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate=sound_file.samplerate
            if chroma:
                stft=np.abs(librosa.stft(X))
            result=np.array([])
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))
            if chroma:
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))
            if mel:
                mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
        return result

    if request.method == "POST":

        freq = 44100
        duration = 5
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        

        name = "test"
        number = "1"
        file_name = "{}{}.wav".format(os.path.join(BASE_DIR, 'myapp/static/'),name+"_"+number)
        print(file_name)

        wv.write(file_name, recording, freq, sampwidth=2)

        audio_a = AudioSegment.from_file(file_name, format="wav")

        sample_rate_b = 16000
        channels_b = 1
        bits_per_sample_b = 16

        audio_a_resampled = audio_a.set_frame_rate(sample_rate_b)

        # Convert stereo to mono if necessary
        if audio_a_resampled.channels > 1 and channels_b == 1:
            audio_a_resampled = audio_a_resampled.set_channels(1)

        # Convert the bit depth if necessary
        if audio_a_resampled.sample_width != bits_per_sample_b // 8:
            audio_a_resampled = audio_a_resampled.set_sample_width(bits_per_sample_b // 8)

        # Export the converted audio to format b
        audio_a_resampled.export(file_name, format="wav")

        modelname = 'trained_model1.sav'

        path = os.path.join(BASE_DIR, 'myapp/static/')
        print(path)
        
        loaded_model = pickle.load(open(modelname, 'rb')) # loading the model file from the storage

        feature=extract_feature(file_name, mfcc=True, chroma=True, mel=True)

        feature=feature.reshape(1,-1)

        prediction=loaded_model.predict(feature)

        print(prediction)

        return render(request, "index.html",context = {"mymessage":"Emotion Detected : "+str(prediction[0]),"Flag":"True"})

    return render(request, "index.html")
