import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

voltage = np.arange(0, 3, 0.1)
field = np.array([33.0, 60.4, 107.1, 158.3, 213, 269, 325, 379, 433, 483, 535, 584, 631, 677, 724, 767, 810, 852, 891, 926, 960, 990, 1017, 1042, 1066, 1086, 1105, 1123, 1137, 1152]) / 1000 #T

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