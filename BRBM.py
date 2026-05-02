import numpy as np
import matplotlib.pyplot as plt


class RBM:

    epsilon_w_arr = np.array([])

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.rng = np.random.default_rng()
        
        self.W = 0.01 * self.rng.standard_normal((N,M))
        self.theta_v = np.zeros(N)
        self.theta_h = np.zeros(M)

    def E_RBM(self, sigma, tau):
        energy = 0
        
        for i in range(self.N):
            for j in range(self.M):
                  energy -= self.W[i,j]*sigma[i]*tau[j]

        for i in range(self.N):
            energy += self.theta_v[i]*sigma[i]
        
        for j in range(self.M):
            energy += self.theta_h[j]*tau[j]          

        return energy

    def sample_hidden(self, sigma):
        tau = np.zeros(self.M)

        for j in range(self.M):
            h_j = 0.0
            for i in range(self.N):
                h_j += self.W[i,j] *sigma[i]
            h_j -= self.theta_h[j]

            prob = 1.0/(1.0 +np.exp(-2*h_j))
            
            if self.rng.random() < prob:
                tau[j] = 1
            else:
                tau[j] = -1
        
        return tau
    
    def sample_visible(self, tau):
        sigma = np.zeros(self.N)
    
        for i in range(self.N):
            h_i = 0.0
            for j in range(self.M):
                h_i += self.W[i,j] * tau[j]
            h_i -= self.theta_v[i]

            prob = 1.0/(1.0 + np.exp(-2*h_i))

            if self.rng.random() < prob:
                sigma[i] = 1
            else:
                sigma[i] = -1

        return sigma

    def data_average_dw(self, data_set):
        de_dw_array = np.zeros((self.N, self.M))
        for sigma in data_set:
            tau = self.sample_hidden(sigma)
            for i in range(self.N):
                for j in range(self.M):
                    de_dw_array[i,j] += sigma[i]*tau[j]
        
        de_dw_array /= len(data_set)
        
        return de_dw_array

    def model_average_dw(self, data_set, k):
        
        de_dw_array = np.zeros((self.N, self.M))
        for sigma0 in data_set:
            sigma = sigma0.copy()
            tau = self.sample_hidden(sigma)

            for _ in range(k):
                sigma = self.sample_visible(tau)
                tau = self.sample_hidden(sigma)

            for i in range(self.N):
                for j in range(self.M):
                    de_dw_array[i,j] += sigma[i]*tau[j]

        de_dw_array /= len(data_set)
        
        return de_dw_array

    def update_w(self, eta, k, data_set): 
        self.W = self.W +  eta * (self.data_average_dw(data_set)-self.model_average_dw(data_set,k))
    def calculate_epsilon_W(self, w_before):
        epsilon_w = 0
        for i in range(self.N):
            for j in range(self.M):
                epsilon_w += np.abs(self.W[i,j]-w_before[i,j])
        epsilon_w /= (self.N * self.M)
        return epsilon_w

    def train_model(self, eta, k, data_set, n_epochs):
        epsilon_w_arr = []
        for epoch in range(n_epochs):
            w_before = self.W.copy()
            self.update_w(eta,k,data_set)
            epsilon_w = self.calculate_epsilon_W(w_before)
            epsilon_w_arr.append(epsilon_w)
            print(f"Epoch: {epoch} | epsilon_w: {epsilon_w}")
        
        self.epsilon_w_arr = np.array(epsilon_w_arr)
        return self.epsilon_w_arr

    def save_model(self, filename):
        np.savez(filename, W=self.W, epsilon_w_arr=self.epsilon_w_arr)
    
    def load_model(self, filename):
        data = np.load(filename)
        self.W = data["W"]
        self.epsilon_w_arr = data["epsilon_w_arr"]

    def generate_rbm_states(self, num_states=1000, melting_iterations=1000):
        states = []
        sigma = self.rng.choice([-1,1], size=self.N)

        for _ in range(melting_iterations):
            tau = self.sample_hidden(sigma)
            sigma = self.sample_visible(tau)

        for _ in range(num_states):
            tau = self.sample_hidden(sigma)
            sigma = self.sample_visible(tau)
            states.append(sigma.copy())

        return np.array(states)

    def display_epsilon_w(self,filename):
        
        epoch_array = np.arange(len(self.epsilon_w_arr))
        
        plt.figure()
        plt.plot(epoch_array, self.epsilon_w_arr, color='saddlebrown',marker='+', linestyle='None')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlabel('Epoch Number')
        plt.ylabel(r'$\epsilon_W = \frac{1}{NM} \sum_{i, j} |\Delta W_{ij}|$')
        plt.savefig(filename)
        plt.show()
    
    def display_inter_layer_couplings(self,filename):
        
        weights = self.W.flatten()

        plt.figure()
        plt.hist(weights, bins=30, color ='palevioletred',histtype='stepfilled')
        plt.xlabel(r"$W_{ij}$")
        plt.ylabel("Count")
        plt.savefig(filename)
        plt.show()


        
                      
            
    


