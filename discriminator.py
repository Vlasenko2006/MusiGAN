import torch
import torch.nn as nn
import math
import torch.nn.utils as utils



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
    def __init__(self, input_dim, n_channels=2, seq_len=12000, max_layers=8, minibatch_out_features=10, kernel_dims=5):
        super(Discriminator_with_mdisc, self).__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
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





class FDiscriminator(nn.Module):
    def __init__(self, input_dim, n_channels=2, seq_len=12000, max_layers=7, nfreq_chunks = 40):
        super(FDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
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

        for i in range(max_layers):
            out_len = math.floor((cur_seq_len + 2 * padding - kernel_size) / stride + 1)
            conv = nn.Conv1d(in_channels, channels, kernel_size, stride, padding)
            block = nn.Sequential(conv, nn.LeakyReLU(0.25))
            layers.append(block)
            res_conv = nn.Conv1d(in_channels, channels, kernel_size=1) if in_channels != channels else nn.Identity()
            self.residual_blocks.append(res_conv)
            in_channels = channels
            channels *= 2
            cur_seq_len = out_len
            if cur_seq_len <= 4:
                break

        self.layers = layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=1)

        # --- NEW: Define N and final_linear ---
        self.N = nfreq_chunks
        self.final_linear = nn.Linear(1 + 2*self.N - 1, 1)

    def forward(self, x):
        # x: [batch, channels, seq_len]
        # --- main conv path ---
        for i, (layer, res_layer) in enumerate(zip(self.layers, self.residual_blocks)):
            identity = x
            out = layer(x)
            res = res_layer(identity)
            out = out + res[..., :out.shape[-1]]
            x = out
    
        pooled = self.pool(x)             # [batch, features, 1]
        pooled = self.final_conv(pooled)  # [batch, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [batch, 1]
    
        # --- frequency chunk analysis ---
        # Split into N chunks along time axis
        batch, channels, seq_len = x.shape
        chunk_size = seq_len // self.N
        freq_feats = []
        for i in range(self.N):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.N-1 else seq_len
            chunk = x[..., start:end]  # [batch, features, chunk_len]
            # Collapse features for FFT (e.g., mean across features if needed)
            chunk_flat = chunk.mean(dim=1)  # [batch, chunk_len]
            # Compute FFT magnitude
            fft = torch.fft.rfft(chunk_flat)
            mag = torch.abs(fft).mean(dim=-1, keepdim=True)  # [batch, 1]
            freq_feats.append(mag)
        freq_feats = torch.cat(freq_feats, dim=1)  # [batch, N]
    
        # Optionally: compute differences between neighboring chunks
        freq_deltas = freq_feats[:, 1:] - freq_feats[:, :-1]  # [batch, N-1]
        freq_summary = torch.cat([freq_feats, freq_deltas], dim=1)  # [batch, 2*N-1]
    
        # --- combine with main path ---
        out = torch.cat([pooled, freq_summary], dim=1)  # [batch, 1 + 2*N - 1]
    
        # Final decision layer (add a linear layer to output real/fake score)
        out = self.final_linear(out)  # nn.Linear(1 + 2*N - 1, 1)
        return out
    
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

    def frequency_analyze(self, x, n_fft=1024, hop_length=256):
        """
        Computes the mean log STFT magnitude (spectral energy) for each input in the batch.
        Args:
            x: Input tensor of shape [batch, channels, seq_len]
            n_fft: FFT window size
            hop_length: Hop size
        Returns:
            Tensor of shape [batch, channels, freq_bins, time_frames] (STFT magnitude)
        """
        # If input is [batch, channels, seq_len], loop over channels for STFT
        batch_size, n_channels, seq_len = x.shape
        stft_mag_list = []
        for c in range(n_channels):
            # [batch, seq_len] for a channel
            x_chan = x[:, c, :]  # shape: [batch, seq_len]
            stft = torch.stft(
                x_chan,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True
            )
            mag = stft.abs()  # [batch, freq_bins, time_frames]
            stft_mag_list.append(mag)
        # Stack to shape [batch, channels, freq_bins, time_frames]
        stft_mag = torch.stack(stft_mag_list, dim=1)
        return stft_mag

    def mean_log_spectral_energy(self, x, n_fft=1024, hop_length=256, eps=1e-7):
        """
        Returns the mean log spectral energy for each sample in the batch.
        """
        stft_mag = self.frequency_analyze(x, n_fft=n_fft, hop_length=hop_length)
        # Compute log magnitude
        log_mag = torch.log(stft_mag + eps)
        # Mean over freq and time
        mean_log_energy = log_mag.mean(dim=[2, 3])  # shape: [batch, channels]
        return mean_log_energy




class Discriminator(nn.Module):
    def __init__(self, input_dim, n_channels=2, seq_len=12000, max_layers=7):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
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

    def forward(self, x):
        for i, (layer, res_layer) in enumerate(zip(self.layers, self.residual_blocks)):
            identity = x
            out = layer(x)
            res = res_layer(identity)
            out = out + res[..., :out.shape[-1]]
            x = out
        out = self.pool(x)
        out = self.final_conv(out)
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


class Discriminator_smart2(nn.Module):
    def __init__(self, input_dim, n_channels=2, seq_len=12000, max_layers=6):
        super(Discriminator_smart2, self).__init__()
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.seq_len = seq_len

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
                # First layer: no normalization
                block = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.25)
                )
            else:
                # Remaining layers: spectral normalization and residual (no InstanceNorm1d)
                conv = utils.spectral_norm(conv)
                block = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.25)
                )
            layers.append(block)

            # Residual connection (1x1 conv to match channels if needed)
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

        # Adaptive pooling to 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        for idx, (layer, res_layer) in enumerate(zip(self.layers, self.residual_blocks)):
            identity = x
            out = layer(x)
            # Match the shape for addition
            res = res_layer(identity)
            out = out + res[..., :out.shape[-1]]
            x = out
        out = self.pool(x)
        out = self.final_conv(out)
        return out.view(out.size(0), -1)  # [batch, 1]


class Discriminator_smart(nn.Module):
    def __init__(self, input_dim, n_channels=2, seq_len=12000, max_layers=6):
        super(Discriminator_smart, self).__init__()
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
        super(Simple_Discriminator, self).__init__()
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

