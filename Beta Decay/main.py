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

def f_quad(x, a, h, k):
    return a*(x - h)**2 + k

def f_lin(x, a, b):
    return a*x + b

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

incriment = 0.04          #VOLTS

e_const = 1.60217663E-19    #C
c_const = 299792458         #m/s
m_0_c2 = 8.18712225E-14     #j
r_const = 0.0381             #m

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

maximum, _ = find_peaks(count, height = np.max(count)/2)
poi = maximum[-1]

peak_offset = 10
x_interest = field[poi - peak_offset:poi + peak_offset]
y_interest = count[poi - peak_offset:poi + peak_offset]
err_interest = count_err[poi - peak_offset:poi + peak_offset]

#p0 = [amplitude guess, center guess, STD guess]
popt, pcov = curve_fit(f_gauss, x_interest, y_interest, sigma = err_interest, maxfev = 1500, p0=[100, 90, 2])
a_opt, mu_opt, sig_opt = popt

x_gauss = np.linspace(np.min(x_interest), np.max(x_interest), 1000)
y_gauss = f_gauss(x_gauss, a_opt, mu_opt, sig_opt)

x_gauss_energy = (np.sqrt((m_0_c2)**2 + (e_const*c_const*r_const*x_gauss/1000)**2) - m_0_c2)
x_gauss_energy = x_gauss_energy * 6.242E+18

print('Gaussian Parameters: [A: %3f, Mu: %3f, Sigma: %3f]' % (a_opt, mu_opt, sig_opt))
print('Errors on Gaussian Parameters: [A: %3f, Mu: %3f, Sigma: %3f]' % totuple(np.sqrt(np.diag(pcov))))
print('Average Background: %3f Counts' % avg_background)


###################################################################################################
#POST CALIBRATION

k_shell_energy = .6242E6

offset = (x_gauss_energy[np.argmax(y_gauss)]) - k_shell_energy
energy -= offset
x_gauss_energy -= offset


print('Peak Energy: %3f ev' % x_gauss_energy[np.argmax(y_gauss)])

###################################################################################################
#QUAD FITTING

quad_poi = maximum[4]
quad_offset = 25
quad_interest_x = energy[quad_poi - quad_offset:quad_poi + quad_offset]
quad_interest_y = count[quad_poi - quad_offset:quad_poi + quad_offset]
quad_err_interest = count_err[quad_poi - quad_offset:quad_poi + quad_offset]

popt_quad, pcov_quad = curve_fit(f_quad, quad_interest_x, quad_interest_y, sigma = quad_err_interest, maxfev = 1500, p0 = [-1, 2E5, 120])
a_quad_opt, h_opt, k_opt = popt_quad

x_quad = np.linspace(np.min(quad_interest_x), np.max(quad_interest_x), 1000)
y_quad = f_quad(x_quad, a_quad_opt, h_opt, k_opt)

print('Quadratic Parameters: [a: %3f, h: %3f, k: %3f]' % (a_quad_opt, h_opt, k_opt))
print('Errors on Quadratic Parameters: [a: %3f, h: %3f, k: %3f]' % totuple(np.sqrt(np.diag(pcov_quad))))


###################################################################################################
#LIN FITTING

lin_poi = maximum[8] + 4
lin_offset = 14
lin_interest_x = energy[lin_poi - lin_offset:lin_poi + lin_offset]
lin_interest_y = count[lin_poi - lin_offset:lin_poi + lin_offset]
lin_err_interest = count_err[lin_poi - lin_offset:lin_poi + lin_offset]

popt_lin, pcov_lin = curve_fit(f_lin, lin_interest_x, lin_interest_y, sigma = lin_err_interest)
a_lin_opt, b_lin_opt = popt_lin

x_lin = np.linspace(np.min(lin_interest_x), np.max(lin_interest_x), 1000)
y_lin = f_lin(x_lin, a_lin_opt, b_lin_opt)

print('Linear Parameters: [a: %3f, b: %3f]' % (a_lin_opt, b_lin_opt))
print('Errors on Linear Parameters: [a: %3f, b: %3f]' % totuple(np.sqrt(np.diag(pcov_lin))))

###################################################################################################
#INTERSECTIONS

idx_quad = np.argwhere(np.diff(np.sign(y_quad - y_gauss))).flatten()
idx_lin = np.argwhere(np.diff(np.sign(y_lin - y_gauss))).flatten()

# print(idx_lin, idx_quad)
print('Intersections: [Quadratic: %3f, Linear: %3f]' % (x_quad[idx_quad[-1]], x_lin[idx_lin[-1]]))



###################################################################################################
#PLOTTING

fig, axes = plt.subplots(1, 1, figsize = (8, 4), constrained_layout = True)

# ax = axes[0]
# bx = axes[1]
cx = axes

# ax.set_title('Counts vs Field')
# ax.set_xlabel('Field [mT]')
# ax.set_ylabel('Counts [#]')
# ax.grid()
# ax.errorbar(field, count, yerr = count_err, label = 'Raw Data', color = 'black')
# ax.plot(x_gauss, y_gauss, label = 'Fit', color = 'red')
# ax.scatter(field[maximum], count[maximum], marker = 'x', label = 'Maxima', color = 'green')
# ax.legend()

# bx.set_title('Zoomed Peak')
# bx.set_xlabel('Field [mT]')
# bx.set_ylabel('Counts [#]')
# bx.grid()
# bx.errorbar(x_interest, y_interest, yerr = err_interest, capsize = 3, label = 'Raw Data')
# bx.plot(x_gauss, y_gauss, label = 'Fit', color = 'red')
# bx.legend()

cx.set_title('Counts vs Energy')
cx.set_xlabel('Energy [eV]')
cx.set_ylabel('Counts [#]')
cx.grid()
cx.ticklabel_format(axis = 'x', style = 'sci', scilimits = (1E5, 1E6))
# cx.axvline(k_shell_energy, color = 'blue', label = 'K Shell Energy', linestyle = '--')
cx.errorbar(energy, count, yerr = count_err, label = 'Raw Data', color = 'black', zorder = 1)
cx.plot(x_gauss_energy, y_gauss, label = 'Gaussian Fit', color = 'red', zorder = 2)
cx.plot(x_quad, y_quad, label = 'Quadratic Fit', color = 'green', zorder = 3)
cx.plot(x_lin, y_lin, label = 'Linear Fit', color = 'magenta', zorder = 4)
cx.legend()


plt.show()
