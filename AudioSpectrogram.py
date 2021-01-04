from scipy.io import wavfile
from scipy import fft
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import tqdm
import json

def print_spectrogram(song_number):
    # fs, sample = wavfile.read("301-project-train-noisy/sample3.wav")
    sample, fs = librosa.load("301-project-train-noisy/sample"+ str(song_number) +".wav")
    X = librosa.stft(sample)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=fs, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def make_spectrogram_matrix(output_address):
    matrix = []
    for i in tqdm.tqdm(range(0, 7)):
        sample, fs = librosa.load("301-project-train-noisy/sample" + str(i) + ".wav")
        matrix.append(sample)

    npmatrix = np.array(matrix)


    np.save(output_address, npmatrix, allow_pickle= True)


make_spectrogram_matrix("test")

print(np.load("test.npy"))