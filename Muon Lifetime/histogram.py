import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

files = [
    R'Muon Lifetime\Data\Run 1\24-02-20-14-33.data',
    R'Muon Lifetime\Data\Run 1\24-02-21-09-59.data'
]

lifetime = np.array([])
time = np.array([])

for file in files:
    lt, t = np.genfromtxt(file, delimiter = ' ').T
    lifetime = np.append(lifetime, lt)
    time = np.append(time, t)


muon_decay_time = lifetime[lifetime < 20000]/1000
hist, bins = np.histogram(muon_decay_time, bins = 20)
hist_error = np.sqrt(hist)

def exp_fit(x, A, tau, B):
    return A * np.exp(-x/tau) + B

popt, pcov = curve_fit(exp_fit, bins[1:], hist, sigma = hist_error)
A, tau, B = popt
A_err, tau_err, B_err = np.sqrt(np.diag(pcov)).reshape(1, -1)[0]
x_model = np.linspace(min(bins), max(bins), 1000)
y_model = exp_fit(x_model, A, tau, B)

print("Coefficients: A: %f, tau: %f, B: %f" % (A, tau, B))
print("Standard Deviation: A: %f, tau: %f, B: %f" % (A_err, tau_err, B_err))

fig, axes = plt.subplots(1, 1, figsize = (6, 6))


ax = axes
# ax.stairs(hist, bins, color = 'black', label = 'Histogram')
ax.errorbar(bins[1:], hist, hist_error, color = 'blue', ls = 'none', label = 'Error Bars')
ax.plot(x_model, y_model, color = 'red', label = 'Fit')
ax.set_xlabel('Decay Time [$\mu s$]')
ax.set_ylabel('Events')
textbox_str = r"$N(\Delta t) = Ae^{\frac{\Delta t}{\tau}} + B$" + "\n" + r"$A = %.2f \pm %.2f$" %(A, A_err) + "\n" + r"$B = %.2f \pm %.2f$" %(B, B_err) + "\n" + r"$\tau = %.3f \pm %.3f$" %(tau, tau_err)
ax.text(0.55, 0.55, textbox_str, transform=ax.transAxes, fontsize = 8, verticalalignment='top')
ax.legend()

# bx = axes[1]
# bx.imshow(np.log(np.abs(pcov)))
# fig.colorbar(bx.imshow(np.log(np.abs(pcov))), ax = bx)

plt.show()



