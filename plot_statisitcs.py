import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_interp_spline

def compute_chunk_values(input_sequence, num_chunks, reduction="max"):
    """
    Splits the input sequence into N chunks, computes either the maximal value or mean value
    for each chunk, and assembles a new sequence of these values.

    Args:
        input_sequence (torch.Tensor): The input sequence (1D tensor).
        num_chunks (int): The number of chunks to split the input sequence into.
        reduction (str): The reduction method to apply to each chunk. 
                         Choose between "max" (default) or "mean".

    Returns:
        torch.Tensor: The new sequence of maximal or mean values for each chunk.
    """
    if reduction not in ["max", "mean"]:
        raise ValueError("Reduction must be either 'max' or 'mean'.")

    # Ensure input_sequence is a 1D tensor
    if input_sequence.ndim != 1:
        raise ValueError("Input sequence must be a 1D tensor.")

    # Compute the size of each chunk
    chunk_size = len(input_sequence) // num_chunks

    # Handle the case where the sequence cannot be evenly split into chunks
    if len(input_sequence) % num_chunks != 0:
        raise ValueError("Input sequence length must be divisible by num_chunks.")

    # Split the input sequence into chunks
    chunks = input_sequence.view(num_chunks, chunk_size)

    # Compute the reduced value for each chunk
    if reduction == "max":
        reduced_values, _ = torch.max(chunks, dim=1)  # Max value along each chunk
    elif reduction == "mean":
        reduced_values = torch.mean(chunks, dim=1)  # Mean value along each chunk

    return reduced_values


# Load the data
path = "music_out_gan_music_chunk/"

valid = np.load("validation_set.npy")
vsample = valid[100,0,0,:]

# for i in range(10, 360, 10):
#     output = np.load(path + "sample_1_epoch_" + str(i) + ".npy")
#     #out = output[ :]
#     #vsample = valid[i,0,0,:]
#     out = output[0, :]
#     out = out / out.std()

#     # Compute statistics
#     std = np.std(out)
#     left_quantile = np.quantile(out, 0.01)
#     right_quantile = np.quantile(out, 0.99)

#     # Plot histogram
#     plt.hist(out, range=[-5, 5], bins=200, alpha=0.7, color='skyblue', edgecolor='k')
#     plt.title(f"GenMUS, iter = {i}")

#     # Plot vertical lines
#     plt.axvline(std, color='red', linestyle='--', label='Std Dev')
#     plt.axvline(left_quantile, color='green', linestyle='--', label='1st Quantile')
#     plt.axvline(right_quantile, color='orange', linestyle='--', label='99th Quantile')

#     # Optional: also show -std for symmetry
#     plt.axvline(-std, color='red', linestyle='--', alpha=0.7)

#     plt.legend()
#     plt.show()



for i in range(100, 430, 60):
    output = np.load(path + "sample_10_epoch_" + str(i) + ".npy")
    out = output[0, :]
    out = out / out.std()

    # Compute histogram
    std = np.std(out)
    left_quantile = np.quantile(out, 0.01)
    right_quantile = np.quantile(out, 0.99)
    
    
    counts, bin_edges = np.histogram(out, range=[-5, 5], bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth the histogram using a spline to create a tangent-like curve
    spline = make_interp_spline(bin_centers, counts, k=3)
    smooth_x = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    smooth_y = spline(smooth_x)

    #plt.hist(out, range=[-5, 5], bins=200, alpha=0.6, color='skyblue', edgecolor='k', label='Histogram')
    plt.plot(smooth_x, smooth_y, color='red', linewidth=2, label='Tangent Curve (Spline)')

    plt.axvline(std, color='red', linestyle='--', label='Std Dev')
    plt.axvline(left_quantile, color='green', linestyle='--', label='1st Quantile')
    plt.axvline(right_quantile, color='orange', linestyle='--', label='99th Quantile')

    # Optional: also show -std for symmetry
    plt.axvline(-std, color='red', linestyle='--', alpha=0.7)



    plt.title(f"GenMUS, iter = {i}")
    plt.legend()
   # plt.show()