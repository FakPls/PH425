import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def f_exp(x, A, k, B):
    return A * np.exp(k * x) + B

def f_cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

file = 'Cosmic Rays\Data\Counts vs Threshold voltage - Sheet1.csv'

CPM, voltage = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
CPM_err = np.sqrt(CPM)

popt, pcov = curve_fit(f_exp, voltage, CPM)
A_opt, k_opt, B_opt = popt
A_opt_err, k_opt_err, B_opt_err = totuple(np.sqrt(np.diag(pcov)))

x_model = np.linspace(np.min(voltage), np.max(voltage), 1000)
y_model = f_exp(x_model, A_opt, k_opt, B_opt)


fig, axes = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)
ax = axes

ax.errorbar(voltage, CPM, yerr = CPM_err, marker = 'o', ls = 'none', color = 'black', label = 'Raw Data')
ax.axvspan(-1.65, -1.45, alpha = 0.5, color = 'green', label = 'Area of Interest')
# ax.plot(x_model, y_model, color = 'red', label = 'Fit')
ax.set_xlabel('Threshold Voltage [V]')
ax.set_ylabel('Counts [#]')
ax.set_title('Counts vs Threshold Voltage')
ax.grid()
ax.legend()

plt.show()