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
    np.save("Training_Data_vector", data)


def save_and_make_test_data_vector():
    data = MakeDataVectors.import_np_data(testing_data=True)
    np.save("Test_Data_vector", data)

def svm_classification():


    training_data_vector = np.load("Training_Data_vector.npy")
    test_data_vector = np.load("Test_Data_vector.npy")


    labels = ComputeFeatures.import_training_labels()

    clf = svm.SVC()
    clf.fit(training_data_vector.transpose(),labels)
    results = clf.predict(test_data_vector.transpose())
    print(results)



# save_and_make_training_data_vector()
# save_and_make_test_data_vector()
svm_classification()
