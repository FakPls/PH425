import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

count, bin = np.genfromtxt('Muon Lifetime\Data\dummydata.csv', delimiter = ',', skip_header = 21).T


