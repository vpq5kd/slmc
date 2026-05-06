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

def hamiltonian_effective(spins, J1):
    Lx, Ly = spins.shape

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

                if (neighbor not in cluster) and (spins_test[neighbor_y,neighbor_x] == spins_test[test_pair_y, test_pair_x]):

                    B = 1/T
                    if np.random.rand() < (1-np.exp(-2*B*J1)):
                        f_new.add(neighbor)
                        cluster.add(neighbor)
        f_old = f_new
        
    for spin in cluster:
        x = spin[1]
        y = spin[0]
        spins_test[y,x]*=-1

    
    E_B = hamiltonian(spins_test,J,K)
    E_B_eff = hamiltonian_effective(spins_test, J1)
    
    beta = 1/T

    activation = max(1,np.exp(-beta*((E_B-E_B_eff)-(E_A-E_A_eff))))
    if np.random.rand() < activation:
        return spins_test

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


def run_simulation(numsteps, J1):
    T = 2.490
    N = 40
    K = 0.2
    J = 1

    equilibrium_spins = 300
    equilibrium_spins = 0

    spins = np.random.choice([-1,1], size = (N,N))
    beta = 1/T
   
    magnetization_array = []
    energy_array = []
    spin_spin_correlations_array = []

    for step in tqdm(range(numsteps)):
        spins = wolff_cluster_logic(N,T,J,J1,K,spins) 
        

        magnetization = np.abs(np.sum(spins))/N**2
        magnetization_array.append(magnetization)
        
        C1, C2, C3 = spin_spin_correlations(spins)
        spin_spin_correlations_array.append([C1,C2,C3])
    
        energy = hamiltonian(spins, J, K)
        energy_array.append(energy)
                    
           
    ac, m_amount = autocorrelation(magnetization_array)
    
    return ac, m_amount

def linear_fit_data(spin_spin_correlations_array, energy_array):
    ssca = np.array(spin_spin_correlations_array)
    ea = np.array(energy_array)

    model = LinearRegression()
    model.fit(ssca, ea)

    E0 = model.intercept_
    coeffs = model.coef_
    j_values = -coeffs

    print(E0, j_values)
    return j_values, E0

def load_metropolis(filename):

    N = 40
    K = 0.2
    J = 1

    data = np.load(filename)
    energy_array = data["energy_array"]
    spin_spin_correlations_array = data["spin_spin_correlations_array"]
    magnetization_array = data["magnetization_array"]


    j_values, E0 = linear_fit_data((spin_spin_correlations_array[:,0]/N**2).reshape(-1,1), energy_array/N**2)

    ac, m_amount = autocorrelation(magnetization_array)
    
    return j_values, ac, m_amount

def main():
    numsteps = 10000
    metropolis_filename = "run_constants.npz"

    j_values, metropolis_auto_correlation, metropolis_m_amount = load_metropolis(metropolis_filename)
    j1 = j_values[0]
    wolff_auto_correlation, wolff_m_amount = run_simulation(numsteps, j1)
    
    wolff_tuple = (wolff_auto_correlation, wolff_m_amount, 'SLMC Approach', '+', 'palevioletred')
    metro_tuple = (metropolis_auto_correlation, metropolis_m_amount, 'Naive Approach', 'x', 'cornflowerblue')

    ac_array = [wolff_tuple, metro_tuple]

    display_autocorrelation(ac_array)

main() 
