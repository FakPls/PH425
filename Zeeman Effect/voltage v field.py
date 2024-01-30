import numpy as np 
import matplotlib.pyplot as plt 

voltage = np.array([0, 0.3, 0.6, 0.9, 1.20, 1.50, 1.80, 2.10, 2.40, 2.70, 3.00])
field = np.array([36, 165.9, 342.0, 509.0, 660.0, 803, 931.0, 1033.0, 1114.0, 1173.0, 1220.0]) / 1000 #T

co = np.polyfit(voltage, field, 3)
print(co)

# plt.plot(field, voltage)
# plt.show()