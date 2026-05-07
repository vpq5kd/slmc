import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

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

def main():
    T = 2.490
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

    for step in tqdm(range(30000*(N**2))):
        spins = metropolis_step(spins, J, K, beta)
        
        if step>equilibrium_spins and step % N**2 == 0:
            magnetization = np.abs(np.sum(spins))/N**2
            magnetization_array.append(magnetization)
            
            C1, C2, C3 = spin_spin_correlations(spins)
            spin_spin_correlations_array.append([C1,C2,C3])
        
            energy = hamiltonian(spins, J, K)
            energy_array.append(energy)
                    
           
    ac, m_amount = autocorrelation(magnetization_array)
    display_autocorrelation(ac, m_amount)

    filename = "run_constants.npz"
    np.savez(filename, magnetization_array=np.array(magnetization_array), energy_array=np.array(energy_array), spin_spin_correlations_array=np.array(spin_spin_correlations_array))


def main_load():

    N = 40
    K = 0.2
    J = 1

    filename = "run_constants.npz"
    data = np.load(filename)
    energy_array = data["energy_array"]
    spin_spin_correlations_array = data["spin_spin_correlations_array"]
    magnetization_array = data["magnetization_array"]


    j_values, E0 = linear_fit_data((spin_spin_correlations_array[:,0]/N**2).reshape(-1,1), energy_array/N**2)
    display_energy_vs_c1(spin_spin_correlations_array,energy_array,N,E0,j_values)


    ac, m_amount = autocorrelation(magnetization_array)
    
    display_autocorrelation(ac, m_amount)

def main_train():

    T = 2.490
    N = 40
    K = 0.2
    J = 1

    equilibrium_spins = 300*(N**2)
    equilibrium_spins = 0

    spins = np.random.choice([-1,1], size = (N,N))
    beta = 1/T
   

    data_set = []
    for step in tqdm(range(30000*(N**2))):
        spins = metropolis_step(spins, J, K, beta)
        
        if step>equilibrium_spins and step % N**2 == 0:
            data_set.append(spins)
        
                    
           

    filename = "naive_data_set.npz"
    np.savez(filename, data_set=data_set)

main_load()
