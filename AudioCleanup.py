import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from scipy import fft
from scipy.io import wavfile

fs, sample = wavfile.read("301-project-train-noisy/sample1.wav")

spectrum = fft.fftshift(fft.fft(sample))
abs_spec = np.abs(spectrum)
print("partway")
plt.stem(abs_spec[80000:130000], use_line_collection=True)
plt.show()
fftlength = len(spectrum)
print(len(abs_spec))
# Notice, band from 90000 to 122000 is unusually increased. To clean, we must remove that bump

zeros = np.zeros((122000 - 90000))

spectrum[90000:122000] = zeros
spectrum[fftlength-122000: fftlength-90000] = zeros

output = np.real(fft.ifft(fft.ifftshift(spectrum))).astype(np.int16).T

wavfile.write("test_outputs/clean1.wav", fs, output)

print("done")
