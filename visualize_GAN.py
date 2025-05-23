import matplotlib.pyplot as plt
import numpy as np
import torch

import numpy as np

def moving_mean(arr, window_size = 10):
    """
    Compute the moving mean of a 1D numpy array using a specified window size.
    
    Parameters:
        arr (np.ndarray): Input 1D array.
        window_size (int): Size of the moving window.
        
    Returns:
        np.ndarray: Array of moving means, length = len(arr) - window_size + 1
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if window_size > len(arr):
        raise ValueError("window_size must be smaller or equal to the length of the array")
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

# Example usage:
# arr = np.array([1, 2, 3, 4, 5])
# print(moving_mean(arr, 3))  # Output: [2. 3. 4.]


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
#path = "music_out_gan_music_chunk_100/"
#path = "music_out_gan_music_chunk_old/"
#path = "music_out_gan_music_chunk_old2/"
#path = "music_out_gan_music_chunk_old_single/"
#path = "music_out_gan_music_vae/"
epoch = 540

output = np.load(path + "sample_10_epoch_" + str(epoch) + ".npy")
valid = np.load("validation_set.npy")
vsample = valid[100,0,0,:]
p=3.
output[output>p]=p
output[output<-p]=-p


output_short = moving_mean(output[0,:1000], window_size = 10)
output_short = moving_mean(output_short, window_size = 3)
output = output/output.std()
output_short = output_short/output_short.std()

# Plot the data
plt.figure(figsize=(10, 6))

# Plot output and target on the same subfigure


plt.plot(1*(output_short[:1000]), label="Output", color="blue")

# Add labels, title, and legend
plt.title(f"Comparison of Output and Target (Epoch {epoch})")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


vsample = vsample/vsample.std()
# Add labels, title, and legend
plt.plot(1*(vsample[:1000]), label="vsample", color="blue")
plt.title("Validation sample")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()





p = 5
n = np.random.randn(120000)
n[n>p]=p
n[n<-p]= -p
# plt.plot(n)
# plt.show()

n = n/n.std()
out = output[0,:]
out = out/out.std()

# plt.hist(n,range=[-5,5], bins = 100)
# plt.title("Noise")
# plt.show()


# plt.hist(out,range=[-5,5], bins = 100)
# plt.title("GenMUS, iter = " + str(epoch))
# plt.show()


# vsample=vsample/vsample.std()
# plt.hist(vsample,range=[-5,5], bins = 100)
# plt.title("Vsample, iter = " + str(epoch))
# plt.show()

# # Convert target[0] to a PyTorch tensor
