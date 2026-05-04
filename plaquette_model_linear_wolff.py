import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng()

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

def hamiltonian_effecitve(spins, J1):
    Lx, Ly = spins.shape()

    C1 = 0
    for x in range(Lx):
        for y in range(Ly):
            s = spins[x,y]
            right = spins[(x+1)%Lx, y]
            up = spins[x, (y+1)%Ly]

            C1 +=  s * right
            C1 +=  s * up

    return -J1*C1

def wolff_cluster_logic(N,T,J,J1,K, spins):
    
    E_A = hamiltonian(spins,J,K)
    E_A_eff = hamiltonian_effective(spins, J1)

    spins_test = spins.copy()

    random_site_x = rng.integers(N)
    random_site_y = rng.integers(N)

    random_site = (random_site_x, random_site_y)
    cluster = set()
    cluster.add(random_site)
    f_old = set()
    f_old.add(random_site)

    while len(f_old) > 0:
        f_new = set()
        for test_pair in f_old:
            test_pair_x = test_pair[1]
            test_pair_y = test_pair[0]

            right = (test_pair_y, (test_pair_x + 1)%N)
            down = ((test_pair_y + 1)%N, test_pair_x)
            left = (test_pair_y, (test_pair_x - 1)%N)
            up = ((test_pair_y - 1)%N, test_pair_x)

            neighbors = [up, down, right, left]

            for neighbor in neighbors:

                neighbor_x = neighbor[1]
                neighbor_y = neighbor[0]

                if (neighbor not in cluster) and (spins_test_test[neighbor_y,neighbor_x] == spins_test[test_pair_y, test_pair_x]):

                    B = 1/T
                    if np.random.rand() < (1-np.exp(-2*B)):
                        f_new.add(neighbor)
                        cluster.add(neighbor)
        f_old = f_new
        
    for spin in cluster:
        x = spin[1]
        y = spin[0]
        spins[y,x]*=-1

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
    plt.ylabel(r"$\langle M(t)M(t+\Delta \tau)\rangle-\langle M \rangle^2$")
    plt.xlim(0, 800)
    plt.ylim(-0.001,0.035)
    plt.legend()
    plt.show()

def display_energy_vs_c1(ssca, energy_array, N, E_0, j_values):
    
    ssca = np.array(ssca)
    
    C1_array = ssca[:,0]/N**2
    energy_array = np.array(energy_array)/N**2

    effective_energy = E_0 - j_values[0]*C1_array

    plt.figure()
    plt.plot(C1_array, energy_array, label="samples", marker='o',linestyle='None',markerfacecolor='None',color='forestgreen')
    plt.plot(C1_array, effective_energy, label='fit',color='black')
    plt.xlabel(r"$\frac{C_1}{N}$")
    plt.ylabel(r"$\frac{E}{N}$",rotation=0)
    plt.legend()
    plt.show()


def main():
    T = 2.493
    N = 40
    K = 0.2
    J = 1

    equilibrium_spins = 300*(N**2)
    equilibrium_spins = 0

    spins = np.random.choice([-1,1], size = (N,N))
    beta = 1/T
   
    magnetization_array = []
    energy_array = []
    spin_spin_correlations_array = []

    for step in tqdm(range(30000)):
        spins = wolff_cluster_logic(N,T,spins) 
        

        magnetization = np.abs(np.sum(spins))/N**2
        magnetization_array.append(magnetization)
        
        C1, C2, C3 = spin_spin_correlations(spins)
        spin_spin_correlations_array.append([C1,C2,C3])
    
        energy = hamiltonian(spins, J, K)
        energy_array.append(energy)
                    
           
    ac, m_amount = autocorrelation(magnetization_array)
    display_autocorrelation(ac, m_amount)

    #filename = "run_constats.npz"
    #np.savez(filename, magnetization_array=np.array(magnetization_array), energy_array=np.array(energy_array), spin_spin_correlations_array=np.array(spin_spin_correlations_array))

main()
