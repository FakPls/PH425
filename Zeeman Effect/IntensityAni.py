from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks
import time

ti  = time.time()
#Open image, convert to grayscale

im = Image.open('Zeeman Effect\Lab Images\Yellow 4.png')
im = im.convert('LA')

cropped_im = im.crop((704, 143, 5656, 3727))
cropped_im = cropped_im.rotate(-90)
x, y = cropped_im.size
# cropped_im.show()
xPixels = np.arange(0, x)

#Record Intensity/intensity of pixels along x

plots = []

px = cropped_im.load()
slices = 200
slices_to_pixels = int(y/slices)

for j in range(0, y, slices_to_pixels):
    intensity = np.zeros(x)
    for i in range(x):
        intensity[i] = px[i,j][0]
    plots.append(intensity)

def animate(i):
    ln1.set_data(xPixels, plots[i])
    ax.set_title(i)

fig, ax = plt.subplots(1, 1, figsize = (8,8))
ax.set_xlim(0, x)
ax.set_ylim(0, 255)
ln1, = plt.plot([], [], lw=1)
ani = animation.FuncAnimation(fig, animate, frames=len(plots), interval=10)
ani.save('pen.gif',writer='pillow',fps=24)

tf = time.time()
print('Computation Time:', int(tf - ti), 's')