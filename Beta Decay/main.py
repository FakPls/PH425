import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

files = [
    'Beta Decay\Data\Manual Run 1.csv'
    # 'Beta Decay\Data\Manual Run 1.txt'
]

count = np.array([])
run = np.array([])

for file in files:
    r, v, c, t, d = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    count = np.append(count, c)
    run = np.append(run, r)

plt.plot(run, count)
plt.show()
