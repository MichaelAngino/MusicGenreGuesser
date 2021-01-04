from scipy.io import wavfile
from scipy import fft
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display



X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()