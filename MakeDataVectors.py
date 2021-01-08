import ComputeFeatures
import numpy as np
import librosa


def import_np_data(training_data=False, testing_data=False):
    if training_data:
        arrs = []
        for fil_num in range(700):
            arrs.append(librosa.load(f"301-project-train-clean/clean{fil_num}.wav")[0])
        return arrs
    if testing_data:
        arrs = []
        for fil_num in range(300):
            arrs.append(librosa.load(f"301-project-test-clean/clean{fil_num+700}.wav")[0])
        return arrs


def form_data_matrix(np_data, sample_rate = 22050):
    data_matrix = np.zeros(shape=(24, len(np_data)))
    for entry_num in range(len(np_data)):
        data_matrix[0, entry_num] = np.mean(ComputeFeatures.compute_chroma(np_data[entry_num]))
        # Should we expand chroma into the 12 subcategories?
        data_matrix[1, entry_num] = np.mean(ComputeFeatures.compute_spec_centroid(np_data[entry_num]))
        data_matrix[2, entry_num] = np.mean(ComputeFeatures.compute_spec_rolloff(np_data[entry_num], sample_rate))
        data_matrix[3, entry_num] = np.mean(ComputeFeatures.compute_zero_crossing_rate(np_data[entry_num]))
        mfccs = np.mean(ComputeFeatures.compute_MFCC(np_data[entry_num], sample_rate), axis=1)
        data_matrix[4:24, entry_num] = mfccs
    return data_matrix