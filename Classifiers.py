import MakeDataVectors
from scipy.io import wavfile
from scipy import fft
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import tqdm
import json
import ComputeFeatures
from sklearn import svm

def save_and_make_training_data_vector():
    data = MakeDataVectors.import_np_data(training_data=True)
    np.save("Data_vector", data)



def svm_classification():


    data_vector = np.load("Data_vector.npy")
    print(data_vector)

    labels = ComputeFeatures.import_training_labels()

    clf = svm.SVC()
    clf.fit(data_vector,labels)


# save_and_make_training_data_vector()
svm_classification()
