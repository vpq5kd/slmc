import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


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

    x = np.random.randint(0, Lx)
    y = np.random.randint(0, Ly)

    dE = delta_energy_flip(spins, x, y, J, K)

    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        spins[x, y] *= -1

    return spins


def calculate_B4(magnetization_array):
    M4 = np.mean(magnetization_array**4)
    M2 = np.mean(magnetization_array**2)

    return 1 - M4 / (3 * M2**2)
    

def run_one_temperature(args):
    length, temp, J, K, n_thermal, n_measure = args

    spins = np.random.choice([-1, 1], size=(length, length))
    beta = 1 / temp

    for _ in range(n_thermal):
        spins = metropolis_step(spins, J, K, beta)

    magnetization_array = []

    for _ in range(n_measure):
        spins = metropolis_step(spins, J, K, beta)
        magnetization_array.append(np.sum(spins))

    magnetization_array = np.array(magnetization_array)
    B4 = calculate_B4(magnetization_array)

    return length, temp, B4


def main():
    temps = np.linspace(2, 4, 30)
    L = [10,20,40]

    K = 0.2
    J = 1

    n_thermal = 5000
    n_measure = 5000000

    tasks = [(length, temp, J, K, n_thermal, n_measure)
             for length in L for temp in temps]

    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_one_temperature, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    plt.figure()

    for length in L:
        length_results = [
            (temp, B4)
            for result_length, temp, B4 in results
            if result_length == length
        ]

        length_results.sort(key=lambda x: x[0])

        temps_sorted = [x[0] for x in length_results]
        B4s_sorted = [x[1] for x in length_results]

        plt.plot(
            temps_sorted,
            B4s_sorted,
            label=f"{length}",
            marker="o",
            linestyle="None",
        )

    plt.xlabel(r"$T$")
    plt.ylabel(r"$B_{4}$", rotation=0)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
