import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import deconvolve, convolve
from matplotlib import pyplot as plt

def f_sin(x):
    return np.sin(x)

def f_gauss(x, A, mu, sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

x = np.linspace(0, 10, 1000)
y_gauss_1 = f_gauss(x, 2, 2, 1)
y_gauss_2 = f_gauss(x, 2, 8, 1)

y_conv = convolve(y_gauss_1, y_gauss_2, mode = 'same')


plt.plot(x, y_gauss_1)
plt.plot(x, y_gauss_2)
plt.plot(x, y_conv)
plt.show()
