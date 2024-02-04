from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks

#Open image, convert to grayscale

im = Image.open('Zeeman Effect\Test Images\TEST3.jpg')
im = im.convert('LA')
x, y = im.size
# im.show()
xPixels = np.arange(0, x)

#Record Intensity/intensity of pixels along x

px = im.load()
intensity = np.zeros(x)
for i in range(x):
    intensity[i] = px[i,y/2][0]

#Locate peaks in Intensity vs x plot and calculate distances between peaks

max, _ = find_peaks(intensity, height = np.max(intensity)/2)

diff_pixels = np.ediff1d(max)
print('Distance between peaks (pixels):', diff_pixels)

#Zoom in on largest peak and fit gaussian, calculate errors on fit

zoom_peak = intensity[max].argmax()
# print(zoom_peak)

zoom_factor = 7
xZoom = xPixels[(xPixels > max[zoom_peak] - zoom_factor) & (xPixels < max[zoom_peak] + zoom_factor)]
yZoom = intensity[(xPixels > max[zoom_peak] - zoom_factor) & (xPixels < max[zoom_peak] + zoom_factor)]

def gauss_f(x, A, mu, sig):
    return A*np.exp(-(x-mu)**2/sig**2)

#p0 = [amplitude guess, center guess, STD guess]
popt, pcov = curve_fit(gauss_f, xZoom, yZoom, p0 = [245, 152, 5])
A_opt, mu_opt, sig_opt = popt
xModel = np.linspace(np.min(xZoom), np.max(xZoom), 1000)
yModel = gauss_f(xModel, A_opt, mu_opt, sig_opt)

print('Gaussian Values:', popt)
print('Errors on Gaussian Values:', np.sqrt(np.diag(pcov)))

#Plot

fig, axes = plt.subplots(1, 3, figsize = (14, 6))
ax = axes[0]
ax.set_title('Intensity vs. x')
ax.set_xlabel('x [Pixels]')
ax.set_ylabel('Intensity [0-255]')
ax.plot(xPixels, intensity, linewidth = 1, label = 'Intensity')
ax.scatter(max, intensity[max], color = 'red', s = 15, label = 'Maxima')
ax.legend(loc = 'lower right')
zoomRect = patches.Rectangle((min(xZoom), min(yZoom)), (np.max(xZoom)-min(xZoom)), (np.max(yZoom)-min(yZoom)), linewidth = 1, edgecolor = 'green', facecolor = 'none')
ax.add_patch(zoomRect)

bx = axes[1]
bx.set_title('Largest Peak Gaussian Fit')
bx.set_xlabel('x [Pixels]')
bx.set_ylabel('Intensity [0-255]')
bx.scatter(xZoom, yZoom, s = 15, label = 'Intensity')
bx.plot(xModel, yModel, linewidth = 1, label = 'Gaussian fit', color = 'green')
bx.legend(loc = 'lower right')

cx = axes[2]
cx.set_title('Covariance Matrix (Gaussian Fit)')
cx.imshow(np.log(np.abs(pcov)))
fig.colorbar(cx.imshow(np.log(np.abs(pcov))), ax = cx)

plt.show()