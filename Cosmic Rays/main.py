import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def f_cos_2(x, A, B, h, k):
    return A * np.cos(B * (x - h))**2 + k

files = [
    'Cosmic Rays\Data\Run 1, delta_angle = 10, time = 1 min - Sheet1.csv'
]

counts = np.array([])
angle = np.arange(180, 0, -10)

for file in files:
    angle_temp, counts_temp = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    counts = np.append(counts, counts_temp)

popt, pcov = curve_fit(f_cos_2, angle, counts)
A_opt, B_opt, h_opt, k_opt = popt

x_model = np.linspace(np.min(angle), np.max(angle), 1000)
y_model = f_cos_2(x_model, A_opt, B_opt, h_opt, k_opt)

fig, axes = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)

main_plot = axes

main_plot.grid()
main_plot.set_title('Counts vs Angle')
main_plot.set_xlabel('Angle [Degrees]')
main_plot.set_ylabel('Counts [#]')
main_plot.scatter(angle, counts)

plt.show()
