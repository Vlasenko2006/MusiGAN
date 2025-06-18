import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from vae_model import  AudioDataset
import os


def compute_chunkwise_stats_loss(fake_music, real_music, num_chunks=200, lambda_mean=0.1, lambda_std=0.1, lambda_max=0.05):  # num_chunks = 200
   # print("real_music.size() = ", real_music.size())
    _,n_channels, seq_len = real_music.size()
    chunk_len = seq_len // num_chunks
    local_mean_loss = 0
    local_std_loss = 0
    local_max_loss = 0
    for i in range(num_chunks):
        start = i * chunk_len
        end = (i + 1) * chunk_len if i < num_chunks - 1 else seq_len
        real_chunk = real_music[..., start:end]
        fake_chunk = fake_music[..., start:end]
        mean_real = real_chunk.mean(dim=-1, keepdim=True)
        mean_fake = fake_chunk.mean(dim=-1, keepdim=True)
        std_real = real_chunk.std(dim=-1, keepdim=True)
        std_fake = fake_chunk.std(dim=-1, keepdim=True)
        max_real = real_chunk.abs().amax(dim=-1, keepdim=True)
        max_fake = fake_chunk.abs().amax(dim=-1, keepdim=True)
        local_mean_loss += torch.mean((mean_fake - mean_real) ** 2)
        local_std_loss  += torch.mean((std_fake - std_real) ** 2)
        local_max_loss  += torch.mean((max_fake - max_real) ** 2)
    # Average over chunks
    local_mean_loss /= num_chunks
    local_std_loss  /= num_chunks
    local_max_loss  /= num_chunks
    return (
        lambda_mean * local_mean_loss +
        lambda_std * local_std_loss +
        lambda_max * local_max_loss
    )

# Assume you have: compute_chunkwise_stats_loss, AudioDataset, and your dataset loaded as 'dataset'
# For demonstration, let's say dataset is a torch.utils.data.Dataset
# Each __getitem__ returns: target_norm, target_mean, target_std

def compute_recon_cost_matrix(dataset, device='cpu'):
    N = len(dataset)
    # If your data is large, you may want to limit N or sample a subset
    Rcost = np.zeros((N, N), dtype=np.float32)
    print(f"Computing all-pairs reconstruction cost for {N} samples...")

    # Preload all normalized waveforms to memory for speed (if fits!)
    all_norm = []
    for i in tqdm(range(N), desc="Loading samples"):
        norm, mean, std = dataset[i]
        all_norm.append(norm)  # Add batch dim

    # Compute all-pairs cost
    for i in tqdm(range(N), desc="Recon cost rows"):
        xi = all_norm[i].to(device)
        for j in range(N):
            xj = all_norm[j].to(device)
            # [1, channels, seq_len] for both
            # compute_chunkwise_stats_loss expects (batch, channels, seq_len)
            # We'll compare xi (real) vs xj (fake)
            cost = compute_chunkwise_stats_loss(xj, xi).item()
            Rcost[i, j] = cost

    return Rcost

def plot_recon_cost_histogram(Rcost, bins=50, save_path=None):
    flat_Rcost = Rcost.flatten()
    plt.figure(figsize=(6,4))
    plt.hist(flat_Rcost, bins=bins, alpha=0.7, color='blue')
    plt.title("Histogram of Reconstruction Cost (all pairs)")
    plt.xlabel("Recon cost")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path)
    plt.show()



dataset_folder = "../dataset"
data =  np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)
dataset = AudioDataset(data)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Rcost = compute_recon_cost_matrix(dataset, device)
np.save('Rcost.npy', Rcost)
plot_recon_cost_histogram(Rcost, bins=50, save_path="recon_cost_hist.png")
