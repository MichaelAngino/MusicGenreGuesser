import ComputeFeatures
import numpy as np
import librosa
import tqdm


def import_np_data(training_data=False, testing_data=False):
    if training_data:
        data_matrix = np.zeros(shape=(24, 700))
        for fil_num in tqdm.tqdm(range(700)):
            current_audio, fs = librosa.load(f"301-project-train-clean/clean{fil_num}.wav")
            data_matrix[0, fil_num] = np.mean(ComputeFeatures.compute_chroma(current_audio))
            # Should we expand chroma into the 12 subcategories?
            data_matrix[1, fil_num] = np.mean(ComputeFeatures.compute_spec_centroid(current_audio))
            data_matrix[2, fil_num] = np.mean(ComputeFeatures.compute_spec_rolloff(current_audio, fs))
            data_matrix[3, fil_num] = np.mean(ComputeFeatures.compute_zero_crossing_rate(current_audio))
            mfccs = np.mean(ComputeFeatures.compute_MFCC(current_audio, fs), axis=1)
            data_matrix[4:24, fil_num] = mfccs
        return data_matrix
    if testing_data:
        data_matrix = np.zeros(shape=(24, 300))
        for fil_num in tqdm.tqdm(range(300)):
            current_audio, fs = librosa.load(f"301-project-test-clean/clean{fil_num}.wav")
            data_matrix[0, fil_num] = np.mean(ComputeFeatures.compute_chroma(current_audio))
            # Should we expand chroma into the 12 subcategories?
            data_matrix[1, fil_num] = np.mean(ComputeFeatures.compute_spec_centroid(current_audio))
            data_matrix[2, fil_num] = np.mean(ComputeFeatures.compute_spec_rolloff(current_audio, fs))
            data_matrix[3, fil_num] = np.mean(ComputeFeatures.compute_zero_crossing_rate(current_audio))
            mfccs = np.mean(ComputeFeatures.compute_MFCC(current_audio, fs), axis=1)
            data_matrix[4:24, fil_num] = mfccs
        return data_matrix
