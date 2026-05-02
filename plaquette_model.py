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
    M = np.abs(magnetization_array)
    M_mean = np.mean(M)

    C = []

    for tau in tqdm(range(5000)):
        if tau == 0:
            val = np.mean(M*M)
        else:
            val = np.mean(M[:-tau] * M[tau:])

        C.append(val - M_mean**2)

    return np.array(C), 5000
    

def main():
    T = 2.5 
    N = 40
    K = 0.2
    J = 1

    spins = np.random.choice([-1,1], size = (N,N))
    beta = 1/T
   
    magnetization_array = []
    for _ in tqdm(range(5000)):
        spins = metropolis_step(spins, J, K, beta)
    for _ in tqdm(range(1000000)):
        spins = metropolis_step(spins, J, K, beta)
        magnetization = np.sum(spins)
        magnetization_array.append(magnetization)

    C,M_amount = autocorrelation(magnetization_array)
    delta_taus = np.arange(M_amount)
    plt.figure()
    plt.plot(delta_taus, C, label=f"Naive Approach",marker='o',linestyle='None', color="mediumvioletred")

    
    plt.xlabel(r"$\Delta \tau$")
    plt.ylabel(r"$\langle M(t)M(t+\Delta \tau)\rangle-\langle M \rangle^2$",rotation=0)
    plt.legend()
    plt.show()



main()
