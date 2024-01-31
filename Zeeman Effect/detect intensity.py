from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks

im = Image.open('Zeeman Effect\TEST.png')
im = im.convert('LA')
x, y = im.size
# im.show()
xPixels = np.arange(0, x)

px = im.load()
intensity = np.zeros(x)
for i in range(x):
    intensity[i] = px[i,y/2][0]

max = argrelextrema(intensity, np.greater)
max_2, _ = find_peaks(intensity, height = np.max(intensity)/2)

plt.plot(xPixels, intensity, linewidth = 1)
# plt.vlines(max, 0, np.max(intensity), color = 'red')
plt.vlines(max_2, 0, np.max(intensity), color = 'green')
plt.show()