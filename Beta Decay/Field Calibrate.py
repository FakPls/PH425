import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

voltage = np.arange(0, 5, 0.2)
field = np.array([7.80, 7.80, 7.80, 7.80, 7.80, 7.80, 9.19, 14.12, 
                  20.54, 26.55, 33.60, 41.09, 48.58, 55.88, 63.19, 
                  70.08, 76.98, 83.75, 90.15, 96.68, 102.95, 109.25, 
                  115.48, 121.68, 122.68])

plt.plot(voltage, field)
plt.scatter(voltage, field, color = 'red', s = 15)
plt.show()