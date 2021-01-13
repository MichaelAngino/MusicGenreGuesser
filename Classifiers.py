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
import csv


def save_and_make_training_data_vector():
    data = MakeDataVectors.import_np_data(training_data=True)
    np.save("Training_Data_vector", data)


def save_and_make_test_data_vector():
    data = MakeDataVectors.import_np_data(testing_data=True)
    np.save("Test_Data_vector", data)


def save_and_make_data2():
    train_data = MakeDataVectors.import_np_data2(training_data=True)
    np.save("Training_Data2_vector", train_data)
    test_data = MakeDataVectors.import_np_data2(testing_data=True)
    np.save("Test_Data2_vector", test_data)



def svm_classification():
    # training_data_vector = np.load("Training_Data_vector.npy")
    # test_data_vector = np.load("Test_Data_vector.npy")
    training_data_vector = np.load("Training_Data2_vector.npy")
    test_data_vector = np.load("Training_Data2_vector.npy")

    labels = ComputeFeatures.import_training_labels()

    clf = svm.SVC()
    clf.fit(training_data_vector.transpose(),labels)
    results = clf.predict(test_data_vector.transpose())
    print(results)
    # print(results.shape)
    # np.save("svm_predict", results)
    export_results_csv(results, "svm_predictions2b")


def export_results_csv(results, file_name):
    with open(f'{file_name}.csv', 'w', newline='') as csvfile:
        wrtr = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        wrtr.writerow(["filename", "label"])
        output_lists = [[f"sample{i+700}.wav", results[i]] for i in range(300)]
        wrtr.writerows(output_lists)
    print(f"{file_name} exported successfully")

# save_and_make_training_data_vector()
# save_and_make_test_data_vector()
save_and_make_data2()
svm_classification()

