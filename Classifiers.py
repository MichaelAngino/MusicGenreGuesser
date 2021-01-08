import MakeDataVectors
from scipy.io import wavfile
from scipy import fft
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import tqdm
import json

def save_and_make_data_vector():
    data = MakeDataVectors.import_np_data(training_data=True)
    data_vector = MakeDataVectors.form_data_matrix(data)
    # np.save("Data_vector", data_vector)



def ssvm_classification():


    data_vector = np.load("Data_vector.npy")
    print(data_vector)


save_and_make_data_vector()