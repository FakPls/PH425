import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

voltage = np.array([0, 0.3, 0.6, 0.9, 1.20, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00])
field = np.array([36, 165.9, 342.0, 509.0, 660.0, 803, 931.0, 1033.0, 1114.0, 1173.0, 1220.0]) / 1000 #T

def f_tri(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x+ d

popt, pcov = curve_fit(f_tri, voltage, field, p0 = [0, 0, .5, 0.02])
a_opt, b_opt, c_opt, d_opt = popt
xModel = np.linspace(np.min(voltage), np.max(voltage), 1000)
yModel = f_tri(xModel, a_opt, b_opt, c_opt, d_opt)

co = np.polyfit(voltage, field, 3)

print('Polyfit: ', co)
print('Curve_fit:', popt)
print('Curve_fit error:', np.sqrt(np.diag(pcov)))

fig, axes= plt.subplots(1, 2, figsize = (8, 6))
ax = axes[0]
bx = axes[1]

ax.scatter(voltage, field, s = 15, color = 'blue', label = 'Data')
ax.plot(xModel, yModel, color = 'red', label = 'Fit')
ax.set_title('Voltage vs Field')
ax.set_xlabel('Voltage [V]')
ax.set_ylabel('B Field [T]')
ax.legend()
ax.grid()

bx.imshow(np.log(np.abs(pcov)))
fig.colorbar(bx.imshow(np.log(np.abs(pcov))), ax = bx)

plt.show()