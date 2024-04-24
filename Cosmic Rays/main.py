import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def f_cos_2(x, A, B, h, k):
    return A * np.cos(B * (x - h))**2 + k

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

files = [
    # 'Cosmic Rays\Data\Run 1, delta_angle = 20, time = 2 min.csv',
    'Cosmic Rays\Data\Run 2, delta_angle = 20, time = 30s.csv'
]

file = files[0]
raw_csv = np.genfromtxt(file, delimiter = ',', skip_header = 1)

angle = np.arange(90, -91, -20)
angle_err = np.zeros(len(angle)) + 10
counts = []
count_err = []

for item in raw_csv:
    counts.append(np.sum(item))
    count_err.append(np.std(item))
    



popt, pcov = curve_fit(f_cos_2, angle, counts, p0 = [0, 0, 0, 0])
A_opt, B_opt, h_opt, k_opt = popt

x_model = np.linspace(np.min(angle), np.max(angle), 1000)
y_model = f_cos_2(x_model, A_opt, B_opt, h_opt, k_opt)

fig, axes = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)

main_plot = axes

main_plot.grid()
main_plot.set_title('Counts vs Angle')
main_plot.set_xlabel('Angle [Degrees]')
main_plot.set_ylabel('Counts [#]')
main_plot.errorbar(angle, counts, xerr = angle_err, yerr = count_err)
main_plot.plot(x_model, y_model)

plt.show()
