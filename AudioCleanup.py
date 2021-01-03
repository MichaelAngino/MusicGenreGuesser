import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from scipy import fft
from scipy.io import wavfile

fs, sample = wavfile.read("301-project-train-noisy/sample1.wav")

spectrum = fft.fft(sample)
abs_spec = np.abs(spectrum)
print("partway")
plt.stem(fft.fftshift(abs_spec), use_line_collection=True)
plt.show()
# Notice, band from ~9000 to 11000 is unusually increased. To clean, we must remove that bump

# wavfile.write("test_outputs/clean0.wav", fs, sample)

print("done")
