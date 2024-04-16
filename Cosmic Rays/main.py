import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

files = [
    ''
]

counts = np.array([])
angle = np.array([])

for file in files:
    counts_temp, angle_temp = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    counts = counts + counts_temp
    angle = angle + angle_temp

fig, axes = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)

main_plot = axes

main_plot.scatter(angle, counts)

plt.show()
