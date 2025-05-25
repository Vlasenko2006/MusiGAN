import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from vae_model import VariationalAttentionModel
from discriminator import Discriminator
from vae_utilities_cpu import load_checkpoint
from noise_fun import noise_fun

# --- Configuration ---
generator_ckpt = "../../checkpoints/generator_epoch_690.pt"
discriminator_ckpt_pattern = "../../checkpoints/discriminator_epoch_{}.pt"
epochs = list(range(540, 701, 10))
eval_batch_size = 32
seq_len = 120000
n_channels = 2
noise_dim = seq_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load generator ---
generator = VariationalAttentionModel(sound_channels=n_channels, seq_len=seq_len).to(device)
g_optimizer = torch.optim.Adam(generator.parameters())
load_checkpoint(generator_ckpt, generator, g_optimizer)
generator.eval()

# --- Generate fake music batch once ---
with torch.no_grad():
    noise = noise_fun(batch_size=eval_batch_size, n_channels=n_channels, seq_len=noise_dim, device=device)
    fake_music, _, _ = generator(noise)
    fake_music = fake_music.cpu()  # Move to CPU for evaluation

# --- Evaluate all discriminator checkpoints ---
gan_criterion = torch.nn.BCEWithLogitsLoss()
epoch_losses = []

for epoch in epochs:
    disc_ckpt = discriminator_ckpt_pattern.format(epoch)
    if not os.path.exists(disc_ckpt):
        print(f"Checkpoint not found: {disc_ckpt}")
        continue

    discriminator = Discriminator(input_dim=n_channels, n_channels=n_channels, seq_len=seq_len).to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters())
    load_checkpoint(disc_ckpt, discriminator, d_optimizer)
    discriminator.eval()

    with torch.no_grad():
        fake_music_device = fake_music.to(device)
        d_fake_logits = discriminator(fake_music_device)
        d_loss_fake = gan_criterion(d_fake_logits, torch.ones_like(d_fake_logits))
        mean_logit = d_fake_logits.mean().item()
        mean_prob = torch.sigmoid(d_fake_logits).mean().item()
        epoch_losses.append({
            "epoch": epoch,
            "mean_d_loss_fake": d_loss_fake.item(),
            "mean_logit": mean_logit,
            "mean_prob": mean_prob
        })
        print(f"Epoch {epoch}: mean D loss={d_loss_fake.item():.4f}, mean logit={mean_logit:.4f}, mean prob={mean_prob:.4f}")

# --- Plot results ---
epochs_plotted = [x["epoch"] for x in epoch_losses]
d_losses = [x["mean_d_loss_fake"] for x in epoch_losses]
probs = [x["mean_prob"] for x in epoch_losses]

plt.figure(figsize=(10,6))
plt.plot(epochs_plotted, d_losses, marker='o', label='Mean D Loss (label=real)')
plt.plot(epochs_plotted, probs, marker='x', label='Mean D Probability (real)')
plt.xlabel("Discriminator Epoch")
plt.ylabel("Value")
plt.title("Discriminator 'strictness' vs. Epoch (lower loss = more fooled by fakes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()