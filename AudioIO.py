from scipy.io import wavfile

fs, sample = wavfile.read("301-project-train-noisy/sample0.wav")

wavfile.write("test_outputs/sample0.wav", fs, sample)