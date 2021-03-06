from librosa import feature
import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv


def compute_zero_crossing_rate(data):
    """
    :param data: audio file stored as a numpy array
    :return: zero_crossing rates for each time
    NOTE: could adjust to return avg rate
    """
    zero_crossings = feature.zero_crossing_rate(data)
    return zero_crossings


def compute_spec_centroid(data):
    """

    :param data:
    :return:
    """
    cents = feature.spectral_centroid(data)[0]
    return cents


def compute_chroma(data):
    """
    :param data:
    :return:
    """
    chr = feature.chroma_stft(data)
    return chr


def compute_spec_rolloff(data, sample_rate):
    """
    Computes frequency where 85% of the energy is lower than
    :param data: audio np array
    :param sample_rate: sample rate
    :return: np array of spec rolloff data
    """
    spec_r = feature.spectral_rolloff(data, sr=sample_rate)
    return spec_r


def compute_MFCC(data, sample_rate, num_coefs=20):
    """
    small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope
    :param data: np array audiodata
    :param sample_rate:
    :param num_coefs: number of features to generate, default 20
    :return: np array (2D) that contains the key features of an audio file
    """
    return feature.mfcc(data, sr=sample_rate, n_mfcc=num_coefs)


def compute_tempo(data, sample_rate):
    onset_env = librosa.onset.onset_strength(data, sr=sample_rate)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)
    return tempo[0]


def import_training_labels():
    labels = []
    with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels.append(row[1])

    labels.pop(0)

    return labels

