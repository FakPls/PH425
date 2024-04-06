import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

file = open('Beta Decay\Data\Calibration_Parameters.txt', 'w')

def lin_fit(x, a, b):
    return a*x + b

voltage = np.arange(0, 5.2, 0.2)
field = np.array([7.93, 7.93, 7.93, 7.93, 7.95, 7.92, 7.93, 9.24, 14.44, 
                  20.835, 26.995, 34.0, 41.3, 48.7, 56.0, 63.2, 70.3, 77.2, 
                  83.9, 90.3, 96.8, 103.0, 109.4, 115.6, 121.8, 122.8])


voltage = voltage[7:-1]
field = field[7:-1]

popt, pcov = curve_fit(lin_fit, field, voltage)
a, b = popt

xmodel = np.linspace(min(field), max(field), 1000)
ymodel = lin_fit(xmodel, a, b)

print('A:', a, "B:", b)

file.write(str(a) + ',' + str(b) + ',' + str(min(field)) + ',' + str(max(field)))
file.close()

# plt.plot(voltage, field)
plt.ylabel('Voltage [V]')
plt.xlabel('Field [mT]')
plt.plot(xmodel, ymodel, color = 'red', label = 'Fit')
plt.scatter(field, voltage, color = 'black', s = 15, label = 'Data')
plt.legend()
plt.grid()
plt.show()