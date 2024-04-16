import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

files = [
    'Cosmic Rays\Data\Run 1, delta_angle = 10, time = 15 min - Sheet1.csv'
]

counts = np.array([])
angle = np.array([])

for file in files:
    counts_temp, angle_temp = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    counts = np.append(counts, counts_temp)
    angle = np.append(angle, angle_temp)

fig, axes = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)

main_plot = axes

main_plot.grid()
main_plot.set_title('Counts vs Angle')
main_plot.set_xlabel('Angle [Degrees]')
main_plot.set_ylabel('Counts [#]')
main_plot.scatter(angle, counts)

plt.show()
