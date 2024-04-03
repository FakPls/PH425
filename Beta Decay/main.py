import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def f_lin_inverse(x, a, b):
    return (x - b)/a

files = [
    'Beta Decay\Data\Manual Run 1.csv'
    # 'Beta Decay\Data\Manual Run 1.txt'
]

A, B = np.genfromtxt('Beta Decay\Data\Calibration_Parameters.txt', delimiter = ',')

count = np.array([])
run = np.array([])

for file in files:
    r, v, c, t, d = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    count = np.append(count, c)
    run = np.append(run, r)

voltage = run * 0.1
field = f_lin_inverse(voltage, A, B)


plt.xlabel('Field [mT]')
plt.ylabel('Counts [N]')
plt.plot(field, count)
plt.show()
