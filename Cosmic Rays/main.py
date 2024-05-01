import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt



def f_cos_2(x, A, n, B):
    return A * np.cos(np.deg2rad(n * x))**2 + B

def f_cos_2_3d(x, y, A, n, B):
    return ((A * np.cos(np.deg2rad(n * x))**2 + B) + (A * np.cos(np.deg2rad(n * y))**2 + B)) / 2

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

angle = np.arange(90, -91, -20)
angle_err = np.zeros(len(angle)) + 10
counts = []

for item in raw_csv:
    temp = np.sort(item)[3:-2]
    counts.append(np.sum(temp))
    
count_err = np.sqrt(counts)

popt, pcov = curve_fit(f_cos_2, angle, counts, sigma = count_err, p0 = [0, 0, 0])
A_opt, n_opt, B_opt = popt
A_opt_err, n_opt_err, B_opt_err = totuple(np.sqrt(np.diag(pcov)))

x_model = np.linspace(np.min(angle), np.max(angle), 1000)
y_model = f_cos_2(x_model, A_opt, n_opt, B_opt)

num_points = 1000
x = np.linspace(-90, 90, num_points)
y = np.linspace(-90, 90, num_points)
X, Y = np.meshgrid(x, y)
Z = f_cos_2_3d(X, Y, A_opt, n_opt, B_opt)

print('COS Optimal Parameters:          [A: %.4f, n: %.4f, B: %.4f]' % (A_opt, 
                                                                        n_opt,  
                                                                        B_opt))

print('COS Optimal Parameters Error:    [A: %.4f, n: %.4f, B: %.4f]' % (A_opt_err, 
                                                                                 n_opt_err, 
                                                                                 B_opt_err))

print('Percent Error:                   [A: %.4f%%, n: %.4f%%, B: %.4f%%]' % (np.abs(percent_error(A_opt, A_opt_err)), 
                                                                            np.abs(percent_error(n_opt, n_opt_err)),  
                                                                            np.abs(percent_error(B_opt, B_opt_err))))

fig, axes = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)
main_plot = axes

main_plot.grid()
main_plot.set_title('Counts vs Angle')
main_plot.set_xlabel('Angle [Degrees]')
main_plot.set_ylabel('Counts [#]')
main_plot.errorbar(angle, counts, xerr = angle_err, yerr = count_err, color = 'black', label = 'Raw Data', zorder = 0)
main_plot.plot(x_model, y_model, color = 'red', label = 'Fit', zorder = 1)
main_plot.legend()

fig_3d = plt.figure()
ax = fig_3d.add_subplot(111, projection = '3d')

ax.plot_surface(X, Y, Z, color = 'red', alpha = 0.5)
ax.set_title('Counts vs Angle')
ax.set_xlabel('Angle [Degrees]')
ax.set_ylabel('Angle [Degrees]')
ax.set_zlabel('Counts [#]')
ax.grid()

plt.show()
