import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from BRBM_cuda import RBM

def hamiltonian(spins, J, K):
    Lx, Ly = spins.shape

    H_nn = 0
    H_p = 0 

    for x in range(Lx):
        for y in range(Ly):
            spin = spins[x,y]

            right_neighbor = spins[(x+1) % Lx, y]
            up_neighbor = spins[x, (y+1) % Ly]
            up_right_neighbor = spins[(x+1) % Lx, (y+1) % Ly]

            H_nn += spin*right_neighbor
            H_nn += spin*up_neighbor

            H_p += spin * right_neighbor * up_neighbor * up_right_neighbor

    return -J*H_nn - K*H_p

def setup_rbm():
    rbm = RBM(1600,256)
    rbm.load_model("model.npz")
    return rbm

def sample_rbm(spins, beta, rbm):
    sigma_a = torch.tensor(spins.reshape(-1), dtype=torch.float32,device=rbm.device)
    tau = rbm.sample_hidden(sigma_a)
    sigma_b = rbm.sample_visible(tau)
    spins_test = sigma_b.cpu()
    spins_test = spins_test.numpy().reshape(40,40)
    
    E_A = hamiltonian(spins, 1, 0.2)
    E_B = hamiltonian(spins_test, 1, 0.2)
    F_A = rbm.free_energy(sigma_a).item()
    F_B = rbm.free_energy(sigma_b).item()

    
    if np.random.rand() < min(1,np.exp(-beta*(E_B-E_A)+(F_B-F_A))):
        return spins_test

    return spins

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
    rbm = setup_rbm()

    T = 2.490
    beta = 1/T

    numsteps = 5000
    rbm_slmc_filename = "rbm_slmc_vals"
    
    spins = np.random.choice([-1,1],size = (40,40))

    magnetization_array = []
    for step in tqdm(range(numsteps)):
        spins = sample_rbm(spins, beta, rbm)

        magnetization = np.abs(np.sum(spins))/(40**2)
        magnetization_array.append(magnetization)

    ac, m_amount = autocorrelation(magnetization_array)

    rbm_tuple = (ac, m_amount, 'RBM Approach', 'o', 'darkgoldenrod')
    ac_tuple_array = [rbm_tuple]

    display_autocorrelation(ac_tuple_array)

main()
