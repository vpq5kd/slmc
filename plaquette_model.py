import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def delta_energy_flip(spins, x, y, J, K):
    Lx, Ly = spins.shape
    s = spins[x, y]

    nn_sum = (
        spins[(x + 1) % Lx, y]
        + spins[(x - 1) % Lx, y]
        + spins[x, (y + 1) % Ly]
        + spins[x, (y - 1) % Ly]
    )

    dE_nn = 2 * J * s * nn_sum

    plaquettes = [
        (x, y),
        ((x - 1) % Lx, y),
        (x, (y - 1) % Ly),
        ((x - 1) % Lx, (y - 1) % Ly),
    ]

    p_sum = 0

    for px, py in plaquettes:
        p = (
            spins[px, py]
            * spins[(px + 1) % Lx, py]
            * spins[px, (py + 1) % Ly]
            * spins[(px + 1) % Lx, (py + 1) % Ly]
        )
        p_sum += p

    dE_p = 2 * K * p_sum

    return dE_nn + dE_p


def metropolis_step(spins, J, K, beta):
    Lx, Ly = spins.shape

    x = np.random.randint(0,Lx)
    y = np.random.randint(0,Ly)

    dE = delta_energy_flip(spins, x, y, J, K)

    acceptance_probability = min(1, np.exp(-beta*dE))

    rand_val = np.random.uniform(0,1)

    if rand_val <= acceptance_probability:
        spins[x,y] *= -1 
    return spins


def magnetization(spins):
    return np.sum(spins)

def calculate_B4(magnetization_array):
    M4 = np.mean(magnetization_array**4)
    m2_squared = np.mean(magnetization_array**2)**2

    return  1 - M4/(m2_squared*3)


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
        
def spin_spin_correlations(spins):
    Lx, Ly = spins.shape

    C1 = 0
    C2 = 0
    C3 = 0

    for x in range(Lx):
        for y in range(Ly):
            s = spins[x,y]
            
            right = spins[(x+1)%Lx, y]
            up = spins[x, (y+1)%Ly]

            C1 +=  s * right
            C1 +=  s * up

            up_right = spins[(x + 1) % Lx, (y + 1) % Ly]
            up_left  = spins[(x - 1) % Lx, (y + 1) % Ly]

            C2 += s * up_right
            C2 += s * up_left

            right2 = spins[(x + 2) % Lx, y]
            up2    = spins[x, (y + 2) % Ly]

            C3 += s * right2
            C3 += s * up2
    
    return C1, C2, C3

def display_autocorrelation(C, M_amount):

    delta_taus = np.arange(M_amount)
    plt.figure()
    plt.plot(delta_taus, C, label=f"Naive Approach",marker='o',linestyle='None', color="mediumvioletred")

    
    plt.xlabel(r"$\Delta \tau$")
    plt.ylabel(r"$\langle M(t)M(t+\Delta \tau)\rangle-\langle M \rangle^2$",rotation=0)
    plt.legend()
    plt.show()

def display_energy_vs_c1(C1_array, energy_array, N):
    C1_array = np.array(C1_array)/N**2
    energy_array = np.array(energy_array)/N**2

    plt.figure()
    plt.plot(C1_array, energy_array, label="samples", marker='o',linestyle='None',markerfacecolor='None',color='forestgreen')
    plt.xlabel(r"$\frac{C_1}{N}$")
    plt.ylabel(r"$\frac{E}{N}$",rotation=0)
    plt.legend()
    plt.show()

def main():
    T = 2.493
    N = 40
    K = 0.2
    J = 1

    spins = np.random.choice([-1,1], size = (N,N))
    beta = 1/T
   
    magnetization_array = []
    energy_array = []
    C1_array = []
    for step in tqdm(range(10000000)):
        spins = metropolis_step(spins, J, K, beta)
        
        if step %  N**2 == 0:
            magnetization = np.abs(np.sum(spins)) / N**2
            magnetization_array.append(magnetization)
            
            C1, _, _ = spin_spin_correlations(spins)
            C1_array.append(C1)

            energy = hamiltonian(spins, J, K)
            energy_array.append(energy)
    
    display_energy_vs_c1(C1_array, energy_array, N)


main()
