import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def exp_fit(x, A, tau, B):
    return A * np.exp(-x/tau) + B


def params_vs_bins(muon_decay_time, min_bins, max_bins):
    bins_changing = np.arange(min_bins, max_bins, 1)
    Tau_v_bins = []
    A_v_bins = []
    B_v_bins = []
    bins_changing_err = np.sqrt(bins_changing)


    for i in bins_changing:
        hist, bins = np.histogram(muon_decay_time, bins = i)
        hist_error = np.sqrt(hist)

        popt, pcov = curve_fit(exp_fit, bins[1:], hist, sigma = hist_error)
        A, tau, B = popt
        A_err, tau_err, B_err = np.sqrt(np.diag(pcov))
        Tau_v_bins.append(tau)
        A_v_bins.append(A)
        B_v_bins.append(B)
    
    return bins_changing, Tau_v_bins, A_v_bins, B_v_bins, bins_changing_err, tau_err, A_err, B_err
