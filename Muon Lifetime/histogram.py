import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from tau_vs_bins import params_vs_bins

files = [
    R'Muon Lifetime\Data\Run 1\24-02-20-14-33.data',
    R'Muon Lifetime\Data\Run 1\24-02-21-09-59.data',
    R'Muon Lifetime\Data\Run 2\24-02-27-15-38.data',
    R'Muon Lifetime\Data\Run 2\24-02-28-09-49.data'
]

lifetime = np.array([])
time = np.array([])

for file in files:
    lt, t = np.genfromtxt(file, delimiter = ' ').T
    lifetime = np.append(lifetime, lt)
    time = np.append(time, t)

bin_number = 30
muon_decay_time = lifetime[lifetime < 20000]/1000
hist, bins = np.histogram(muon_decay_time, bins = bin_number)
hist_error = np.sqrt(hist)

def exp_fit(x, A, tau, B):
    return A * np.exp(-x/tau) + B

popt, pcov = curve_fit(exp_fit, bins[1:], hist, sigma = hist_error)
A, tau, B = popt
A_err, tau_err, B_err = np.sqrt(np.diag(pcov)).reshape(1, -1)[0]
x_model = np.linspace(min(bins), max(bins), 1000)
y_model = exp_fit(x_model, A, tau, B)

# print("Coefficients: A: %f, tau: %f, B: %f" % (A, tau, B))
# print("Standard Deviation: A: %f, tau: %f, B: %f" % (A_err, tau_err, B_err))

bins_changing, Tau_v_bins, A_v_bins, B_v_bins, bins_changing_err, Tau_v_bins_err, A_v_bins_err, B_v_bins_err = params_vs_bins(muon_decay_time, 4, 70)

fig, axes = plt.subplots(2, 2, figsize = (7, 6), constrained_layout = True)

ax = axes[0][0]
ax.stairs(hist, bins, color = 'black', label = 'Histogram')
# ax.errorbar(bins[1:], hist, hist_error, color = 'blue', ls = 'none', label = 'Error Bars')
ax.plot(x_model, y_model, color = 'red', label = 'Fit')
ax.set_xlabel('Decay Time [$\mu s$]')
ax.set_ylabel('Events')
textbox_str = r"$N(\Delta t) = Ae^{\frac{\Delta t}{\tau}} + B$" + "\n" + r"$A = %.2f \pm %.2f$" %(A, A_err) + "\n" + r"$B = %.2f \pm %.2f$" %(B, B_err) + "\n" + r"$\tau = %.3f \pm %.3f$" %(tau, tau_err)
ax.text(0.45, 0.55, textbox_str, transform=ax.transAxes, fontsize = 8, verticalalignment='top')
ax.legend()

bx = axes[0][1]
bx.set_title(r"$\tau$ Vs. Bin Number")
bx.set_xlabel("Bin Number")
bx.set_ylabel(r"$\tau$")
bx.plot(bins_changing, Tau_v_bins, color = 'Blue')
bx.axvline(bin_number, color = 'Red', linestyle = 'dashed')
bx.axhline(Tau_v_bins[bin_number - 4], color = 'Red', linestyle = 'dashed')

cx = axes[1][0]
cx.set_title("A Vs. Bin Number")
cx.set_xlabel("Bin Number")
cx.set_ylabel("A")
cx.plot(bins_changing, A_v_bins, color = 'Blue')
cx.axvline(bin_number, color = 'Red', linestyle = 'dashed')
cx.axhline(A_v_bins[bin_number - 4], color = 'Red', linestyle = 'dashed')


dx = axes[1][1]
dx.set_title("B Vs. Bin Number")
dx.set_xlabel("Bin Number")
dx.set_ylabel("B")
dx.plot(bins_changing, B_v_bins, color = 'Blue')
dx.axvline(bin_number, color = 'Red', linestyle = 'dashed')
dx.axhline(B_v_bins[bin_number - 4], color = 'Red', linestyle = 'dashed')


# bx = axes[1]
# bx.imshow(np.log(np.abs(pcov)))
# fig.colorbar(bx.imshow(np.log(np.abs(pcov))), ax = bx)

plt.show()



