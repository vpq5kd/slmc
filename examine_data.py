import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

def autocorrelation(magnetization_array):
    M = np.array(magnetization_array)
    M_mean_2 = np.mean(M)**2

    C = []
    lag = 800
    for dt in tqdm(np.arange(lag)):
        A_0 = 0
        for i in range(len(M)-dt):
            A_0 += M[i]*M[i+dt]

        A_0 *= 1/(len(M) -dt)
        C.append(A_0-M_mean_2)

    return np.array(C), lag

def load_metropolis(filename):

    N = 40
    K = 0.2
    J = 1

    data = np.load(filename)
    magnetization_array = data["magnetization_array"]

    ac, m_amount = autocorrelation(magnetization_array)
    
    return  ac, m_amount

def load_wolff(filename):
    data = np.load(filename)
    ac = data["ac"]
    m_amount = data["m_amount"][0]
    
    return ac, m_amount

def display_autocorrelation(ac_tuple_array):

    plt.figure()
    for C, M_amount, label, marker, color in ac_tuple_array:

        delta_taus = np.arange(M_amount)
        plt.plot(delta_taus, C, label=label, marker=marker, linestyle='None', color=color)
        
        
    plt.xlabel(r"$\Delta \tau$")
    plt.ylabel(r"$\langle M(t)M(t+\Delta \tau)\rangle-\langle M \rangle^2$")
    plt.xlim(-25, 800)
    plt.ylim(-0.001,0.035)
    plt.legend()

    plt.show()

def main():
    metropolis_filename = "run_constants.npz"
    wolff_filename = "wolff_vals.npz"

    metropolis_auto_correlation, metropolis_m_amount = load_metropolis(metropolis_filename)
    wolff_auto_correlation, wolff_m_amount = load_wolff(wolff_filename)

    wolff_tuple = (wolff_auto_correlation, wolff_m_amount, 'SLMC Approach', '+', 'palevioletred')
    metro_tuple = (metropolis_auto_correlation, metropolis_m_amount, 'Naive Approach', 'x', 'cornflowerblue')

    ac_array = [wolff_tuple, metro_tuple]

    display_autocorrelation(ac_array)

main()
