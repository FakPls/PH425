import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

voltage = np.arange(0, 3, 0.1)
field = np.array([29.9, 55.1, 97.8, 114.8, 195.3, 246, 299, 347, 397, 444, 489, 534, 577, 619, 661, 701, 739, 775, 809, 842, 870, 898, 920, 943, 961, 979, 993, 1009, 1023, 1035]) / 1000 #T

def f_tri(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x+ d

popt, pcov = curve_fit(f_tri, field, voltage,p0 = [0, 0, .5, 0.02])
a_opt, b_opt, c_opt, d_opt = popt
xModel = np.linspace(np.min(field), np.max(field), 1000)
yModel = f_tri(xModel, a_opt, b_opt, c_opt, d_opt)

co = np.polyfit(voltage, field, 3)

print('Polyfit: ', co)
print('Curve_fit:', popt)
print('Curve_fit error:', np.sqrt(np.diag(pcov)))

fig, axes= plt.subplots(1, 2, figsize = (8, 6))
ax = axes[0]
bx = axes[1]

ax.scatter(field, voltage, s = 15, color = 'blue', label = 'Data')
ax.plot(xModel, yModel, color = 'red', label = 'Fit')
ax.set_title('Field vs Voltage')
ax.set_ylabel('Voltage [V]')
ax.set_xlabel('B Field [T]')
ax.legend()
ax.grid()

bx.imshow(np.log(np.abs(pcov)))
fig.colorbar(bx.imshow(np.log(np.abs(pcov))), ax = bx)

plt.show()