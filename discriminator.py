import torch
import torch.nn as nn
import math
import torch.nn.utils as utils


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super(MinibatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        """
        Minibatch discrimination layer to compute pairwise similarity and enhance feature diversity.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).
        Returns:
            torch.Tensor: Enhanced tensor with minibatch discrimination applied.
        """
        M = x.matmul(self.T.view(x.size(1), -1))  # Linear transformation
        M = M.view(x.size(0), -1, self.T.size(2))  # Reshape to (batch_size, out_features, kernel_dims)
        # Pairwise L1 distances between samples in the minibatch
        out = torch.exp(-torch.abs(M.unsqueeze(0) - M.unsqueeze(1)).sum(3))  # [batch, batch, out_features]
        out = out.sum(0) - 1  # Sum over batch and remove self-similarity
        return torch.cat([x, out], dim=1)  # Concatenate original features with minibatch discrimination


class Discriminator_with_mdisc(nn.Module):
    def __init__(self, input_dim, 
                 n_channels=2,
                 seq_len=12000,
                 max_layers=8,
                 minibatch_out_features=10,
                 kernel_dims=5):
        
        
        super(Discriminator_with_mdisc, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.feature_layer_idx = -2  # Second-to-last layer for features

        kernel_size = 4
        stride = 2
        padding = 1

        layers = nn.ModuleList()
        in_channels = input_dim
        channels = n_channels
        cur_seq_len = seq_len

        self.residual_blocks = nn.ModuleList()

        # Downsampling layers with residual connections
        for i in range(max_layers):
            out_len = math.floor((cur_seq_len + 2 * padding - kernel_size) / stride + 1)
            conv = nn.Conv1d(in_channels, channels, kernel_size, stride, padding)
            if i == 0:
                block = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.25)
                )
            else:
                conv = utils.spectral_norm(conv)
                block = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.25)
                )
            layers.append(block)

            if in_channels != channels:
                res_conv = nn.Conv1d(in_channels, channels, kernel_size=1)
            else:
                res_conv = nn.Identity()
            self.residual_blocks.append(res_conv)

            in_channels = channels
            channels *= 2
            cur_seq_len = out_len
            if cur_seq_len <= 4:
                break

        self.layers = layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=1)

        # Minibatch discrimination layer
        self.minibatch_discrimination = MinibatchDiscrimination(in_features=in_channels, 
                                                                 out_features=minibatch_out_features, 
                                                                 kernel_dims=kernel_dims)

        # Final fully connected layer after minibatch discrimination
        self.fc = nn.Linear(in_channels + minibatch_out_features, 1)

    def forward(self, x):
        for i, (layer, res_layer) in enumerate(zip(self.layers, self.residual_blocks)):
            identity = x
            out = layer(x)
            res = res_layer(identity)
            out = out + res[..., :out.shape[-1]]
            x = out

        # Pooling and feature extraction
        features = self.pool(x).view(x.size(0), -1)  # [batch, in_channels]

        # Apply minibatch discrimination
        enhanced_features = self.minibatch_discrimination(features)  # [batch, in_channels + minibatch_out_features]

        # Final classification
        out = self.fc(enhanced_features)
        return out.view(out.size(0), -1)  # [batch, 1]

    def get_intermediate_features(self, x):
        out = x
        for i, (layer, res_layer) in enumerate(zip(self.layers, self.residual_blocks)):
            identity = out
            out = layer(out)
            res = res_layer(identity)
            out = out + res[..., :out.shape[-1]]
            if i == self.feature_layer_idx:
                return out
        # fallback if not found
        return out



# Gradient clipping utility function
def clip_discriminator_gradients(discriminator, max_norm=5.0):
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)



