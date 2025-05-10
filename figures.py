"""
Generates a scatter plot of fitness v.s. number of synapses in individuals across generations
"""

import os
import os.path
import pickle
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def main():
    # Count number of files - https://stackoverflow.com/q/2632205
    num_generations = len([name for name in os.listdir("./saved_gens") if os.path.isfile(os.path.join("./saved_gens", name))])
    
    all_gens = []
    
    for i in range(num_generations):
        with open(f"./saved_gens/gen_{i}.pkl", "rb") as handle:
            all_gens.append(pickle.load(handle))
        
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)
    cmap = cm.magma

    ax.set_title("Neuroevolution: Fitness v.s. Sparsity")
    ax.set_ylabel("Number of synapses")
    ax.set_xlabel("Fitness score")

    N = len(all_gens)
    for i, gen in enumerate(all_gens):
        ax.scatter(
            [ind["fitness"] for ind in gen[1]],
            [len(ind["connections"].values()) for ind in gen[1]],
            color=cmap.colors[int((i/N)*255)],
            marker="s",
            s=45,
            alpha=0.4,
            edgecolors='none'
        )

    norm = mpl.colors.Normalize(vmin=1, vmax=N)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, orientation='vertical', label='Generation')

    fig.set_dpi(300)
    
    plt.show()

if __name__ == "__main__":
    main()