import numpy as np
import matplotlib.pyplot as plt
from BRBM_cuda_full import RBM

def train_model():
    data_filename = "naive_data_set_fixed.npz"
    load_dict = np.load(data_filename)

    data_set = load_dict["data_set"]

    N = data_set.shape[1]
    M = 256

    rbm = RBM(N,M)
    print(rbm.device,rbm.W.device)

    eta = 0.0001
    k = 5
    n_epochs = 800
    rbm.train_model(eta,k,data_set,n_epochs)
    return rbm

def spin_spin_correlations(spins):
    Lx, Ly = spins.shape

    C1 = 0

    for x in range(Lx):
        for y in range(Ly):
            s = spins[x,y]
            
            right = spins[(x+1)%Lx, y]
            up = spins[x, (y+1)%Ly]

            C1 +=  s * right
            C1 +=  s * up

    return C1

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

def display_c1_validation(c1_tuples,N):
    
    plt.figure()    
    for ea, c1, label, marker, color in c1_tuples:
        ea = ea/N**2
        c1 = c1/N**2
        plt.plot(c1, ea, label=label, marker=marker, color=color,linestyle='None',markerfacecolor='None')

    plt.xlabel(r"$\frac{C_1}{N}$")
    plt.ylabel(r"$\frac{E}{N}$",rotation=0)

    plt.legend()
    plt.show()

def load_model():
    rbm = RBM(1600,256)
    rbm.load_model("model.npz")
    return rbm

def main():
    rbm = train_model()
    rbm.display_epsilon_w("ew.png")
    rbm.save_model("model.npz")
    states = np.array(rbm.generate_rbm_states(num_states=5000))
    states = states.reshape(len(states),40,40)
    
    J = 1
    K = 0.2
    rbm_energy_array = np.array([hamiltonian(spins, J, K) for spins in states])
    rbm_c1_array = np.array([spin_spin_correlations(spins) for spins in states])

    naive_data = np.load("run_constants.npz")
    naive_energy_array = naive_data["energy_array"]
    naive_c1_array = naive_data["spin_spin_correlations_array"][:,0]

    N = 40
    rbm_tuple = (rbm_energy_array, rbm_c1_array, 'RBM', 'o','mediumvioletred')
    naive_tuple = (naive_energy_array, naive_c1_array, 'Naive', '+', 'forestgreen')
    c1_tuples = [naive_tuple, rbm_tuple]
    display_c1_validation(c1_tuples,N)

main()

