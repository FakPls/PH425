import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

def f_lin_inverse(x, a, b):
    return (x - b)/a

def f_gauss(x, A, mu, sig):
    return A*np.exp(-(x-mu)**2/sig**2)

files = [
    'Beta Decay\Data\Manual Run 1.csv'
    # 'Beta Decay\Data\Manual Run 1.txt'
]

incriment = 0.1

A, B = np.genfromtxt('Beta Decay\Data\Calibration_Parameters.txt', delimiter = ',')

count = np.array([])
run = np.array([])

for file in files:
    r, v, c, t, d = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    count = np.append(count, c)
    run = np.append(run, r)

voltage = run * incriment
field = f_lin_inverse(voltage, A, B)



max, _ = find_peaks(count, height = np.max(count)/2)

print(max)


fig, axes = plt.subplots(2, 1, figsize = (6, 6), constrained_layout = True)

ax = axes[0]

ax.set_title('Field vs Counts')
ax.set_xlabel('Field [mT]')
ax.set_ylabel('Counts [#]')
ax.grid()
ax.plot(field, count, label = 'Counts')
ax.legend()


plt.show()
