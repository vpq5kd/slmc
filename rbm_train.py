import numpy as np
from BRBM_vectorized import RBM

def main():
    data_filename = "data_set.npz"
    load_dict = np.load(data_filename)

    data_set = load_dict["data_set"]

    N = data_set.shape[1]
    M
main()
