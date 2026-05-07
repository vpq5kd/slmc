import torch
import matplotlib.pyplot as plt


class RBM:

    def __init__(self, N, M, device=None):
        self.N = N
        self.M = M

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.W = 0.01 * torch.randn(N, M, device=self.device)
        self.theta_v = torch.zeros(N, device=self.device)
        self.theta_h = torch.zeros(M, device=self.device)

        self.epsilon_w_arr = []

    def E_RBM(self, sigma, tau):
        sigma = sigma.to(self.device)
        tau = tau.to(self.device)

        energy = -sigma @ self.W @ tau
        energy += self.theta_v @ sigma
        energy += self.theta_h @ tau

        return energy

    def free_energy(self, sigma):

        sigma = torch.as_tensor(
            sigma,
            dtype=torch.float32,
            device=self.device
        )

        visible_term = self.theta_v @ sigma

        hidden_input = sigma @ self.W - self.theta_h

        hidden_term = torch.sum(
            torch.log(
                2 * torch.cosh(hidden_input)
            )
        )

        F = visible_term - hidden_term

        return F
    def sample_hidden(self, sigma):
        sigma = sigma.to(self.device)

        h_j = sigma @ self.W - self.theta_h
        prob = 1.0 / (1.0 + torch.exp(-2 * h_j))

        tau = torch.where(
            torch.rand_like(prob) < prob,
            torch.ones_like(prob),
            -torch.ones_like(prob)
        )

        return tau

    def sample_visible(self, tau):
        tau = tau.to(self.device)

        h_i = tau @ self.W.T - self.theta_v
        prob = 1.0 / (1.0 + torch.exp(-2 * h_i))

        sigma = torch.where(
            torch.rand_like(prob) < prob,
            torch.ones_like(prob),
            -torch.ones_like(prob)
        )

        return sigma

    def data_average_dw(self, data_set):
        data_set = torch.as_tensor(data_set, dtype=torch.float32, device=self.device)

        h_j = data_set @ self.W - self.theta_h
        prob = 1.0 / (1.0 + torch.exp(-2 * h_j))

        tau = torch.where(
            torch.rand_like(prob) < prob,
            torch.ones_like(prob),
            -torch.ones_like(prob)
        )

        de_dw_array = data_set.T @ tau
        de_dw_array /= len(data_set)

        return de_dw_array

    def model_average_dw(self, data_set, k):
        data_set = torch.as_tensor(data_set, dtype=torch.float32, device=self.device)

        sigma = data_set.clone()

        h_j = sigma @ self.W - self.theta_h
        prob = 1.0 / (1.0 + torch.exp(-2 * h_j))

        tau = torch.where(
            torch.rand_like(prob) < prob,
            torch.ones_like(prob),
            -torch.ones_like(prob)
        )

        for _ in range(k):
            h_i = tau @ self.W.T - self.theta_v
            prob = 1.0 / (1.0 + torch.exp(-2 * h_i))

            sigma = torch.where(
                torch.rand_like(prob) < prob,
                torch.ones_like(prob),
                -torch.ones_like(prob)
            )

            h_j = sigma @ self.W - self.theta_h
            prob = 1.0 / (1.0 + torch.exp(-2 * h_j))

            tau = torch.where(
                torch.rand_like(prob) < prob,
                torch.ones_like(prob),
                -torch.ones_like(prob)
            )

        de_dw_array = sigma.T @ tau
        de_dw_array /= len(data_set)

        return de_dw_array

    def update_w(self, eta, k, data_set):
        self.W = self.W + eta * (
            self.data_average_dw(data_set)
            - self.model_average_dw(data_set, k)
        )

    def calculate_epsilon_W(self, w_before):
        epsilon_w = torch.mean(torch.abs(self.W - w_before))

        return epsilon_w.item()

    def train_model(self, eta, k, data_set, n_epochs):
        data_set = torch.as_tensor(data_set, dtype=torch.float32, device=self.device)

        epsilon_w_arr = []

        for epoch in range(n_epochs):
            w_before = self.W.clone()

            self.update_w(eta, k, data_set)

            epsilon_w = self.calculate_epsilon_W(w_before)

            epsilon_w_arr.append(epsilon_w)

            print(f"Epoch: {epoch} | epsilon_w: {epsilon_w}")

        self.epsilon_w_arr = torch.tensor(epsilon_w_arr)

        return self.epsilon_w_arr

    def save_model(self, filename):
        torch.save(
            {
                "W": self.W.detach().cpu(),
                "theta_v": self.theta_v.detach().cpu(),
                "theta_h": self.theta_h.detach().cpu(),
                "epsilon_w_arr": self.epsilon_w_arr,
            },
            filename
        )

    def load_model(self, filename):
        data = torch.load(filename, map_location=self.device)

        self.W = data["W"].to(self.device)
        self.theta_v = data["theta_v"].to(self.device)
        self.theta_h = data["theta_h"].to(self.device)
        self.epsilon_w_arr = data["epsilon_w_arr"]

    def generate_rbm_states(self, num_states=1000, melting_iterations=1000):
        states = []

        sigma = torch.where(
            torch.rand(self.N, device=self.device) < 0.5,
            torch.ones(self.N, device=self.device),
            -torch.ones(self.N, device=self.device)
        )

        for _ in range(melting_iterations):
            tau = self.sample_hidden(sigma)
            sigma = self.sample_visible(tau)

        for _ in range(num_states):
            tau = self.sample_hidden(sigma)
            sigma = self.sample_visible(tau)
            states.append(sigma.detach().cpu().numpy().copy())

        return states

    def display_epsilon_w(self, filename):
        epoch_array = range(len(self.epsilon_w_arr))

        plt.figure()

        plt.plot(
            epoch_array,
            self.epsilon_w_arr,
            color='saddlebrown',
            marker='+',
            linestyle='None'
        )

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel('Epoch Number')
        plt.ylabel(r'$\epsilon_W = \frac{1}{NM} \sum_{i, j} |\Delta W_{ij}|$')
        plt.savefig(filename)
        plt.show()

    def display_inter_layer_couplings(self, filename):
        weights = self.W.detach().cpu().numpy().flatten()

        plt.figure()

        plt.hist(
            weights,
            bins=30,
            color='palevioletred',
            histtype='stepfilled'
        )

        plt.xlabel(r"$W_{ij}$")
        plt.ylabel("Count")
        plt.savefig(filename)
        plt.show()
