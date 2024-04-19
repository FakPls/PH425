import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import deconvolve, convolve
from matplotlib import pyplot as plt


###################################################################################################
#FITTING  FUNCTIONS / EXTRA FUNCTIONS

def f_lin_inverse(x, a, b):
    return (x - b)/a

def f_gauss(x, A, mu, sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

###################################################################################################
#READ CSV

files = [
    # 'Beta Decay\Data\Manual Run 1.csv',               #0-5V, incriment = 0.10
    # 'Beta Decay\Data\Manual Run 2.csv',               #0-5V, incriment = 0.05
    'Beta Decay\Data\Manual Run 3 1-5V.csv',          #1-5V, incriment = 0.04
    'Beta Decay\Data\Manual Run 4 1-5V.csv'           #1-5V, incriment = 0.04
]

incriment = 0.04            #VOLTS

e_const = 1.60217663E-19    #C
c_const = 299792458         #m/s
m_0_c2 = 8.18712225E-14     #j
r_const = 0.25               #m

A, B, min_field, max_field = np.genfromtxt('Beta Decay\Data\Calibration_Parameters.txt', delimiter = ',')

count = np.zeros(int(5/incriment))
run = np.array([])

for file in files:
    r, v, c, t, d = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    count = count + c
    run = r

voltage = run * incriment
field = f_lin_inverse(voltage, A, B)


mask = (field > min_field) & (field < max_field)
count = count[mask]
count_err = np.sqrt(count)
voltage = voltage[mask]
field = field[mask]

energy = (np.sqrt((m_0_c2)**2 + (e_const*c_const*r_const*field/1000)**2) - m_0_c2)
energy = energy * 6.242E+18

background = count[field > 100]
avg_background = np.average(background)

count = count - avg_background


###################################################################################################
#CURVE FITTING

max, _ = find_peaks(count, height = np.max(count)/2)
poi = max[-1]

peak_offset = 5
x_interest = field[poi - peak_offset:poi + peak_offset]
y_interest = count[poi - peak_offset:poi + peak_offset]
err_interest = count_err[poi - peak_offset:poi + peak_offset]

#p0 = [amplitude guess, center guess, STD guess]
popt, pcov = curve_fit(f_gauss, x_interest, y_interest, sigma = err_interest, maxfev = 1500, p0=[100, 90, 2])
a_opt, mu_opt, sig_opt = popt

x_gauss = np.linspace(np.min(x_interest), np.max(x_interest), 1000)
y_gauss = f_gauss(x_gauss, a_opt, mu_opt, sig_opt)

print('Gaussian Parameters: [A: %3f, Mu: %3f, Sigma: %3f]' % (a_opt, mu_opt, sig_opt))
print('Errors on Gaussian Parameters: [A: %3f, Mu: %3f, Sigma: %3f]' % totuple(np.sqrt(np.diag(pcov))))
print('Average Background: %3f Counts' % avg_background)

recovred, remainder = deconvolve(count, y_gauss)
print(recovred, remainder)

###################################################################################################
#PLOTTING

fig, axes = plt.subplots(3, 1, figsize = (8, 6), constrained_layout = True)

ax = axes[0]
bx = axes[1]
cx = axes[2]

ax.set_title('Counts vs Field')
ax.set_xlabel('Field [mT]')
ax.set_ylabel('Counts [#]')
ax.grid()
ax.errorbar(field, count, yerr = count_err, capsize = 3, label = 'Raw Data')
# ax.plot(field, recovered, label = 'Deconvolution', color = 'black')
ax.plot(x_gauss, y_gauss, label = 'Fit', color = 'red')
ax.scatter(field[max], count[max], marker = 'x', label = 'Maxima', color = 'green')
ax.legend()

bx.set_title('Zoomed Peak')
bx.set_xlabel('Field [mT]')
bx.set_ylabel('Counts [#]')
bx.grid()
bx.errorbar(x_interest, y_interest, yerr = err_interest, capsize = 3, label = 'Raw Data')
bx.plot(x_gauss, y_gauss, label = 'Fit', color = 'red')
bx.legend()

cx.set_title('Counts vs Energy')
cx.set_xlabel('Energy [eV]')
cx.set_ylabel('Counts [#]')
cx.grid()
cx.errorbar(energy, count, yerr = count_err, capsize = 3, label = 'Raw Data')
cx.legend()


plt.show()
