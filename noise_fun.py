import torch



import torch

def noise_fun(batch_size=1, n_channels=2, seq_len=200000, noise_dim=300, device='cpu'):
    tail = seq_len // 3
    seq_len_extended = seq_len + 2 * tail
    pi = torch.tensor(torch.pi, device=device)

    # Randomly choose the number of sinusoids per example/channel
    num_sines = torch.randint(50, noise_dim + 1, (batch_size, n_channels), device=device)

    # Normalized positions [seq_len_extended]
    L = torch.arange(seq_len_extended, device=device) / seq_len_extended
    L = L.view(1, 1, seq_len_extended)  # [1, 1, seq_len_extended]

    # Prepare output
    A = torch.zeros(batch_size, n_channels, seq_len_extended, device=device)

    for k in range(noise_dim):
        # Check for each batch/channel if this sinusoid should be used
        mask = (k < num_sines).float()  # [batch_size, n_channels]

        # Random amplitude and phase per batch/channel
        amps = torch.rand(batch_size, n_channels, device=device)
        phases = 2 * pi * torch.rand(batch_size, n_channels, device=device)

        # Frequency for this sinusoid
        freq = k + 1  # 1-based indexing

        # Compute the sinusoid for all batch/channels at once
        sines = torch.sin(2 * pi * freq * L + phases.unsqueeze(-1))  # [batch, ch, seq]

        # Only add for those who use this sinusoid
        A += (amps * mask).unsqueeze(-1) * sines

    # Remove tails
    A = A[:, :, tail:-tail]

    # Robust clipping
    STD = 2 * A.std()
    A = torch.clamp(A, -STD, STD)
    # Normalize to [-1, 1]
    if A.abs().max() > 0:
        A = A / A.abs().max()
    return A






def noise_fun_vec(batch_size=1, n_channels=2, seq_len=200000, noise_dim=300, device='cpu'):
    tail = seq_len // 3
    seq_len_extended = seq_len + 2 * tail
    pi = torch.tensor(torch.pi, device=device)

    # Randomly choose number of sinusoids (per example/channel): shape [batch_size, n_channels]
    num_sines = torch.randint(50, noise_dim + 1, (batch_size, n_channels), device=device)

    # For vectorization, use the maximum possible: [batch_size, n_channels, max_dim, seq_len_extended]
    max_dim = noise_dim

    # Frequencies: [1, ..., max_dim], shape [1, 1, max_dim, 1]
    freqs = torch.arange(1, max_dim + 1, device=device).view(1, 1, max_dim, 1)
    # Normalized positions: shape [1, 1, 1, seq_len_extended]
    L = torch.arange(seq_len_extended, device=device).view(1, 1, 1, seq_len_extended) / seq_len_extended

    # Random amplitudes and phases: [batch_size, n_channels, max_dim]
    amps = torch.rand(batch_size, n_channels, max_dim, device=device)
    phases = 2 * pi * torch.rand(batch_size, n_channels, max_dim, device=device)

    # Build sines: [batch_size, n_channels, max_dim, seq_len_extended]
    sines = torch.sin(2 * pi * freqs * L + phases.unsqueeze(-1))

    # Mask out unused sinusoids (set amplitude to zero)
    # Build mask: [batch_size, n_channels, max_dim], True where dim < chosen number
    idx = torch.arange(max_dim, device=device).view(1, 1, max_dim)
    mask = idx < num_sines.unsqueeze(-1)  # shape: [batch_size, n_channels, max_dim]
    amps = amps * mask.float()

    # Weighted sum over max_dim (sinusoids): [batch_size, n_channels, seq_len_extended]
    A = torch.sum(amps.unsqueeze(-1) * sines, dim=2)

    # Remove tails
    A = A[:, :, tail:-tail]

    # Robust clipping
    STD = 2 * A.std()
    A = torch.clamp(A, -STD, STD)
    # Normalize to [-1, 1]
    if A.abs().max() > 0:
        A = A / A.abs().max()
    return A
