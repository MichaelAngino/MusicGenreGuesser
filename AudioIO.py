from scipy.io import wavfile
from scipy import fft
import numpy as np

fs, sample = wavfile.read("301-project-train-noisy/sample3.wav")

print(sample.shape)
print(type(sample[0]))
# output = sample
output = np.real(fft.ifft(fft.fft(sample))).astype(np.int16)
print(output.shape)

# print(sample[0:50])
# print(output[0:50])
wavfile.write("test_outputs/there_and_back3.wav", fs, output)
wavfile.write("test_outputs/sample3.wav", fs, sample)