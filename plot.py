import matplotlib.pyplot as plt


def plot_vectors(vector1, vector2, vector3, seed1, seed2, seed3, filename="plot.png"):
    plt.figure(figsize=(10, 6))

    # Plotting the vectors
    plt.plot(vector1, color="red", label=f"seed={seed1}")
    plt.plot(vector2, color="green", label=f"seed={seed2}")
    plt.plot(vector3, color="blue", label=f"seed={seed3}")

    # Adding title and labels
    plt.title("Plot of different seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Total_Reward")

    # Adding legend
    plt.legend()

    # Saving the plot
    plt.savefig(filename)
    plt.close()



