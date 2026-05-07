import numpy as np
import matplotlib.pyplot as plt

data = np.load("b4_data_2.npz")
lr_array = data["lr_array"]

plt.figure()
lengths = [10,20,40]
length_index = 0
color_dict = {10:"firebrick",20:"goldenrod",40:"blueviolet"}
for length_results in lr_array:
    length = lengths[length_index]
    temps_sorted = [x[0] for x in length_results]
    B4s_sorted = [x[1] for x in length_results]

    plt.plot(
        temps_sorted,
        B4s_sorted,
        label=f"{length}",
        marker="o",
        linestyle="None",
        color=color_dict[length]
    )
    length_index += 1
plt.axvline(2.48275862, color='black',linestyle='--',label='Critical Point')
plt.xlabel(r"$T$")
plt.ylabel(r"$B_{4}$", rotation=0)
plt.legend()
plt.show()


