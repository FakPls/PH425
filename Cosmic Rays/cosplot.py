import numpy as np
from matplotlib import pyplot as plt

sigma = 20
mu = 0

x = np.linspace(-90, 90, 1000)
y = np.cos(np.deg2rad(x))**2
gauss = np.exp( - (x - mu)**2 / (2 * sigma**2))

conv = np.convolve(y, gauss, mode = 'same')
conv /= np.max(conv)

plt.plot(x, y, label = r'$\cos^{2}(\theta)$')
plt.plot(x, gauss, label = 'Guassian')
plt.plot(x, conv, label = 'Convolved Signal')
plt.grid()
plt.xlabel('Angle [Degrees]')
plt.ylabel('Counts [#]')
plt.title('Counts vs Angle')
plt.legend()

plt.show()