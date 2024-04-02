import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def lin_fit(x, a, b):
    return a*x + b

voltage = np.arange(0, 5, 0.2)
field = np.array([7.80, 7.80, 7.80, 7.80, 7.80, 7.80, 9.19, 14.12, 
                  20.54, 26.55, 33.60, 41.09, 48.58, 55.88, 63.19, 
                  70.08, 76.98, 83.75, 90.15, 96.68, 102.95, 109.25, 
                  115.48, 121.68, 122.68])

voltage = voltage[6:-1]
field = field[6:-1]

popt, pcov = curve_fit(lin_fit, voltage, field)
a, b = popt

xmodel = np.linspace(min(voltage), max(voltage), 1000)
ymodel = lin_fit(xmodel, a, b)

print('A:', a, "B:", b)

# plt.plot(voltage, field)
plt.xlabel('Voltage [V]')
plt.ylabel('Field [mT]')
plt.plot(xmodel, ymodel, color = 'red', label = 'Fit')
plt.scatter(voltage, field, color = 'black', s = 15, label = 'Data')
plt.legend()
plt.grid()
plt.show()