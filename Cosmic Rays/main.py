import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



def f_cos_2(x, A, B):
    return A * np.cos(np.deg2rad(x))**2 + B

def f_cos_2_3d(r, A):
    return A * np.cos(np.deg2rad(r))**2

def reject_outliers(data, m = 1):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
    
def percent_error(param, param_err):
    return 100 * (param_err/param)
    


files = [
    # 'Cosmic Rays\Data\Run 1, delta_angle = 20, time = 2 min.csv',
    'Cosmic Rays\Data\Run 2, delta_angle = 20, time = 30s.csv'
]

file = files[0]
raw_csv = np.genfromtxt(file, delimiter = ',', skip_header = 1)

area = 30.5 * 7.5
area_err = 0.3E2
runtime = 30
angle = np.arange(90, -91, -20)
angle_err = np.zeros(len(angle)) + 10
counts = []
counts_raw = []
num_runs = 0

for item in raw_csv:
    temp = np.sort(item)[3:-3]
    temp_raw = np.sort(item)
    counts.append(np.sum(temp))
    counts_raw.append(np.sum(temp_raw))
    num_runs = len(temp)


count_err = np.sqrt(counts)
background = np.min(counts)

# counts -= background

popt, pcov = curve_fit(f_cos_2, angle, counts, sigma = count_err, p0 = [0, 0])
A_opt, B_opt = popt
A_opt_err, B_opt_err = totuple(np.sqrt(np.diag(pcov)))

x_model = np.linspace(np.min(angle), np.max(angle), 1000)
y_model = f_cos_2(x_model, A_opt, B_opt)

mu = 0
sigma = 40
gauss = np.exp( - (x_model - mu)**2 / (2 * sigma**2))

conv = np.convolve(y_model, gauss, mode = 'same')
conv /= np.max(conv)
conv *= A_opt


num_points = len(counts)
r = np.linspace(-90, 90, num_points)
phi = np.linspace(0, 2 * np.pi, num_points)
R, P = np.meshgrid(r, phi)
Z = f_cos_2_3d(R, A_opt)

X, Y = R * np.cos(P), R * np.sin(P)

I_vert = A_opt / (30 / num_runs) / area
# I_vert_err = A_opt_err[0] / (30 / num_runs) / area_err
I_vert_err = 0.01

integral = np.sum(Z) / np.diff(r)[0] * I_vert 
integral_err = I_vert * 2 * np.pi



print('COS Optimal Parameters:          [A: %.4f, B: %.4f]' % (A_opt, background))

print('COS Optimal Parameters Error:    [A: %.4f, B: %.4f]' % (A_opt_err, np.min(count_err)))

print('Percent Error:                   [A: %.4f%%, B: %.4f%%]' % (np.abs(percent_error(A_opt, A_opt_err)), np.abs(percent_error(background, count_err[count_err.argmin()]))))

print("Values:                          [I_vert: %.4f, Integral: %.4f]" % (I_vert, integral))

print("Values Error:                    [I_vert: %.4f, Integral: %.4f]" % (I_vert_err, integral_err))

print('Percent Error:                   [I_vert: %.4f%%, Integral: %.4f%%]' % (np.abs(percent_error(I_vert, I_vert_err)), np.abs(percent_error(integral, integral_err))))

fig, axes = plt.subplots(1, 2, figsize = (12, 6), constrained_layout = True)
main_plot = axes[1]
raw_plot = axes[0]


raw_plot.grid()
raw_plot.set_axisbelow(True)
raw_plot.set_title('Counts vs Angle\n(RAW)')
raw_plot.set_xlabel(r'$\theta$ [Degrees]')
raw_plot.set_ylabel('Counts [#]')
raw_plot.errorbar(angle, counts_raw, xerr = angle_err, yerr = count_err, ls='none', color = 'black', label = 'Raw Data', zorder = 1)
raw_plot.legend()

# count_err = np.sqrt(counts)

main_plot.grid()
main_plot.set_axisbelow(True)
main_plot.set_title('Counts vs Angle\n(OUTLIERS REMOVED)')
main_plot.set_xlabel(r'$\theta$ [Degrees]')
main_plot.set_ylabel('Counts [#]')
main_plot.errorbar(angle, counts, xerr = angle_err, yerr = count_err, ls='none', color = 'black', label = 'Raw Data', zorder = 1)
main_plot.plot(x_model, y_model, color = 'red', label = 'Fit', zorder = 2)
# main_plot.plot(x_model, conv, color = 'green', label = "Convolved Fit", zorder = 3)
main_plot.legend()

# fig_3d = plt.figure()
# ax = fig_3d.add_subplot(111, projection = '3d')

# ax.plot_surface(X, Y, Z, color = 'red', alpha = 0.5)
# ax.set_title('Counts vs Angle')
# ax.set_xlabel(r'$\phi$ [Degrees]')
# ax.set_ylabel(r'$\phi$ [Degrees]')
# ax.set_zlabel(r'Counts per Second [$\frac{\#}{s}$]')
# ax.grid()

plt.show()
