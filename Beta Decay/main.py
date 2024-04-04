import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


###################################################################################################
#FITTING  FUNCTIONS

def f_lin_inverse(x, a, b):
    return (x - b)/a

def f_gauss(x, A, mu, sig):
    return A*np.exp(-(x-mu)**2/sig**2)


###################################################################################################
#READ CSV

files = [
    'Beta Decay\Data\Manual Run 1.csv'
    # 'Beta Decay\Data\Manual Run 1.txt'
]

incriment = 0.1 #VOLTS

A, B = np.genfromtxt('Beta Decay\Data\Calibration_Parameters.txt', delimiter = ',')

count = np.array([])
run = np.array([])

for file in files:
    r, v, c, t, d = np.genfromtxt(file, delimiter = ',', skip_header = 1).T
    count = np.append(count, c)
    run = np.append(run, r)

voltage = run * incriment
field = f_lin_inverse(voltage, A, B)


###################################################################################################
#CURVE FITTING

max, _ = find_peaks(count, height = np.max(count)/2)
poi = np.max(max)

peak_offset = 5
x_interest = field[poi-peak_offset:poi+peak_offset]
y_interest = count[poi-peak_offset:poi+peak_offset]

#p0 = [amplitude guess, center guess, STD guess]
popt, pcov = curve_fit(f_gauss, x_interest, y_interest, p0 = [40, 92, 5])
a_opt, mu_opt, sig_opt = popt

x_gauss = np.linspace(np.min(x_interest), np.max(x_interest), 1000)
y_gauss = f_gauss(x_gauss, a_opt, mu_opt, sig_opt)

print('Gaussian Parameters: [A: %3f, Mu: %3f, Sigma: %3f]' % (a_opt, mu_opt, sig_opt))
print('Errors on Gaussian Parameters:', np.sqrt(np.diag(pcov)))

###################################################################################################
#PLOTTING

fig, axes = plt.subplots(2, 1, figsize = (6, 6), constrained_layout = True)

ax = axes[0]
bx = axes[1]

ax.set_title('Field vs Counts')
ax.set_xlabel('Field [mT]')
ax.set_ylabel('Counts [#]')
ax.grid()
ax.plot(field, count, label = 'Raw Data')
ax.plot(x_gauss, y_gauss, label = 'Fit', color = 'red')
ax.scatter(field[max], count[max], marker = 'x', label = 'Maxima', color = 'green')
ax.legend()

bx.set_title('Zoomed Peak')
bx.set_xlabel('Field [mT]')
bx.set_ylabel('Counts [#]')
bx.grid()
bx.plot(x_interest, y_interest, label = 'Raw Data')
bx.plot(x_gauss, y_gauss, label = 'Fit', color = 'red')
bx.legend()


plt.show()
