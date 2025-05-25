import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class Discriminator(nn.Module):
    def __init__(self, input_dim, n_channels=2, seq_len=12000, max_layers=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.seq_len = seq_len

        kernel_size = 4
        stride = 2
        padding = 1

        layers = []
        in_channels = input_dim
        channels = n_channels
        cur_seq_len = seq_len

        # Downsampling layers
        for i in range(max_layers):
            # Calculate output length after conv
            out_len = math.floor((cur_seq_len + 2 * padding - kernel_size) / stride + 1)
            layers.append(nn.Conv1d(in_channels, channels, kernel_size, stride, padding))
            if i > 0:
                # Use InstanceNorm1d instead of BatchNorm1d
                layers.append(nn.InstanceNorm1d(channels, affine=True))
            layers.append(nn.LeakyReLU(0.2))
            in_channels = channels
            channels *= 2
            cur_seq_len = out_len
            if cur_seq_len <= 4:
                break

        # Instead of a fixed kernel size, use AdaptiveAvgPool1d to reduce to 1
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Conv1d(in_channels, 1, kernel_size=1))
        # Output: [batch, 1, 1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out.view(x.size(0), -1)  # [batch, 1]

# Gradient clipping utility function
def clip_discriminator_gradients(discriminator, max_norm=5.0):
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)


# Discriminator Network
class Simple_Discriminator(nn.Module):
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

