import matplotlib.pyplot as plt
import numpy as np

def get_var(vector):
    length = 10
    ret = []
    for i in range(len(vector)):
        if i - length < 0:
            ret.append(np.std(vector[:i+1]))
        else:
            ret.append(np.std(vector[i-length:i+1]))
    return ret

def plot_vectors(vector1, vector2, vector3, seed1, seed2, seed3, filename="plot.png"):
    plt.figure(figsize=(10, 6))

    # Plotting the vectors
    plt.plot(vector1, color="red", label=f"seed={seed1}")
    plt.plot(vector2, color="green", label=f"seed={seed2}")
    plt.plot(vector3, color="blue", label=f"seed={seed3}")
    var1 = get_var(vector1)
    var2 = get_var(vector2)
    var3 = get_var(vector3)
    #print(var1, var2, var3)
    plt.plot(var1, color="red", linestyle="--", label=f"{seed1}standard deviation")
    plt.plot(var2, color="green", linestyle="--", label=f"{seed2}standard deviation")
    plt.plot(var3, color="blue", linestyle="--", label=f"{seed3}standard deviation")

    # Adding title and labels
    plt.title("Plot of different seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Total_Reward")

    # Adding legend
    plt.legend()

    # Saving the plot
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Example usage
    vector1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    vector2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    vector3 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    seed1 = 42
    seed2 = 43
    seed3 = 44
    plot_vectors(vector1, vector2, vector3, seed1, seed2, seed3)
