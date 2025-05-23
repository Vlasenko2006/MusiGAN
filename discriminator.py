import torch
import torch.nn as nn

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim, n_channels=2, seq_len=120000):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, n_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(n_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(n_channels * 2, 1, kernel_size=seq_len // 4),
            # nn.Sigmoid(),  # REMOVE this line!
        )

    def forward(self, x):
        # Output shape: [batch_size, 1]
        return self.model(x).view(x.size(0), -1)
