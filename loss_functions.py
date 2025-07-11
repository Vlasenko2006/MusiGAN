#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:23:42 2025

@author: andrey
"""


import torch
import torch.nn.functional as F



def silence_loss(waveform, beats, weight=1.0):
    """
    Penalizes non-zero values in quiet intervals (regions of low loudness).
    Args:
        waveform: Tensor of shape [batch, channels, seq_len], generated waveform.
        beats: Tensor of detected beat positions (in samples).
        weight: Loss scaling factor.
    Returns:
        silence_loss: Tensor, silence regularity loss.
    """
    batch_size, seq_len = waveform.shape[0], waveform.shape[-1]

    # Create silence mask for all batches at once
    silence_mask = torch.ones_like(waveform, dtype=torch.bool)

    # Vectorized masking of beat intervals
    for b in range(batch_size):
        beat_indices = beats[b].long().clamp(max=seq_len - 1)  # Ensure indices are within bounds
        for i in range(beat_indices.size(0) - 1):
            silence_mask[b, :, beat_indices[i]:beat_indices[i + 1]] = False

    # Define silence regions as parts where absolute waveform amplitude is less than 0.5 * std
    std_per_channel = waveform.std(dim=-1, keepdim=True)  # Compute std along the sequence dimension
    silence_condition = waveform.abs() < 0.5 * std_per_channel
    silence_mask &= silence_condition

    # Penalize non-zero values in silence regions
    silence_loss = torch.mean(torch.abs(waveform[silence_mask]))
    return weight * silence_loss



def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())




def no_seconds_loss(signal, keys=33, epsilon = 0.001):
    length = len(signal) // keys
    aux_total = torch.zeros_like(signal[:, :, :length])  # Initialize tensor for accumulation
    for i in range(0, keys - 1):
        aux_total += torch.abs(
            signal[:, :, i * length:length * (i + 1)] - 
            signal[:, :, (i + 1) * length:length * (i + 2)]
        )

    return -torch.mean(torch.log(aux_total+ epsilon))
        
        
def no_silence_loss(signal, keys=33, epsilon=0.001, silence_threshold=0.05, sharpness_threshold=0.1, sharpness_scaling=-0.001):
    """
    Computes a loss that penalizes silence and enforces consistency across chunks of the input signal.
    
    Args:
        signal (torch.Tensor): Input tensor of shape [batch, n_channels, lengths].
        keys (int): Number of chunks to divide the signal into.
        epsilon (float): Small constant for numerical stability.
        silence_threshold (float): Threshold for detecting silence.
        sharpness_threshold (float): Threshold applied to chunk-wise means.
        sharpness_scaling (float): Scaling factor for sharpness penalty.
    
    Returns:
        torch.Tensor: Computed loss value.
    """
    batch, n_channels, lengths = signal.shape
    length = lengths // keys

    # Silence detection
    silence = torch.where(signal < silence_threshold, 0, signal)  # Threshold silence
    num_zeros = 20 * torch.sum(silence == 0) / (batch * n_channels * lengths)
    silence_cost = torch.exp(-num_zeros)

    # Chunk-wise statistics
    chunk_means = torch.zeros(batch, keys, device=signal.device)  # Allocate tensor for chunk means
    chunk_stds = torch.zeros(batch, keys, device=signal.device)  # Allocate tensor for chunk stds
    for i in range(keys):
        chunk = signal[:, :, i * length: (i + 1) * length]
        chunk_means[:, i] = torch.mean(chunk, dim=(-1, -2))
        chunk_stds[:, i] = torch.std(chunk, dim=(-1, -2))

    # Compute std_cost using vectorized operations
    std_cost = torch.mean(torch.log(chunk_stds + epsilon))

    # Sharpness thresholding
    sharp_output = F.threshold(chunk_means, threshold=sharpness_threshold, value=0)
    sharpness_penalty = sharpness_scaling * torch.mean(torch.log(sharp_output + epsilon))

    # Combine all components
    return sharpness_penalty - std_cost + silence_cost

def melody_loss(signal, keys=33, epsilon=0.001):
    batch, nchanels, lengths = signal.shape
    length = lengths // keys
    
    msec_ch1 = torch.zeros(batch, device=signal.device)  # Allocate tensors on the same device
    msec_ch2 = torch.zeros(batch, device=signal.device)
    chan_real = torch.zeros(batch, device=signal.device)

    # Explicitly ensure keys is an integer
    if not isinstance(keys, int):
        keys = int(keys.item())  # Convert tensor to integer if necessary
    
    # Initialize aux tensor with proper dimensions
    aux = torch.zeros((batch, keys - 2))  # Use a tuple of integers
    
    reg1 = signal[:, :, : -length]
    reg2 = signal[:, :, length : ]
    
    
    for i in range(0, batch):
        stack_ch1 = torch.stack([reg1[i,0,:],reg2[i,0,:]], dim = 0)
        stack_ch2 = torch.stack([reg1[i,1,:],reg2[i,1,:]], dim = 0)
        stack_channels = torch.stack([reg1[i,0,:],reg1[i,1,:]], dim = 0)    
    
                               
        msec_ch1[i] = torch.corrcoef(stack_ch1)[0,1]
        msec_ch2[i] = torch.corrcoef(stack_ch2)[0,1]    
        chan_real[i] = torch.corrcoef(stack_channels)[0,1]
        
        sum_cost=  (stack_ch1 + stack_ch2 + stack_channels)/3
                          
    sharp_output = torch.nn.functional.threshold(sum_cost, threshold=0.01, value=0.01)
    
    return -torch.mean(torch.log(sharp_output + epsilon))
    


def melody_loss_d(real_music, fake_music, keys=33, epsilon=0.001, d_fake_logits = None):
    batch, nchanels, lengths = real_music.shape
    length = lengths // keys

    # Ensure aux is initialized with proper dimensions
    msec_real_ch1 = torch.zeros(batch, device=real_music.device)  # Allocate tensors on the same device
    msec_real_ch2 = torch.zeros(batch, device=real_music.device)
    msec_fake_ch1 = torch.zeros(batch, device=real_music.device)  # Allocate tensors on the same device
    msec_fake_ch2 = torch.zeros(batch, device=real_music.device)


    chan_real = torch.zeros(batch, device=real_music.device)  # Allocate tensors on the same device
    chan_fake = torch.zeros(batch, device=real_music.device)

    reg_real1 = real_music[:, :, : -length]
    reg_real2 = real_music[:, :, length : ]
    
    reg_fake1 = fake_music[:, :, : -length]
    reg_fake2 = fake_music[:, :, length : ]
    
    
    for i in range(0, batch):
        stack_real_ch1 = torch.stack([reg_real1[i,0,:],reg_real2[i,0,:]], dim = 0)
        stack_real_ch2 = torch.stack([reg_real1[i,1,:],reg_real2[i,1,:]], dim = 0)
        stack_fake_ch1 = torch.stack([reg_fake1[i,0,:],reg_fake2[i,0,:]], dim = 0)
        stack_fake_ch2 = torch.stack([reg_fake1[i,1,:],reg_fake2[i,1,:]], dim = 0)
        
        stack_channels_real = torch.stack([reg_real1[i,0,:],reg_real1[i,1,:]], dim = 0)    
        stack_channels_fake = torch.stack([reg_fake1[i,0,:],reg_fake1[i,1,:]], dim = 0)    
                               
        msec_real_ch1[i] = torch.corrcoef(stack_real_ch1)[0,1]
        msec_real_ch2[i] = torch.corrcoef(stack_real_ch2)[0,1]    
        msec_fake_ch1[i] = torch.corrcoef(stack_fake_ch1)[0,1]
        msec_fake_ch2[i] = torch.corrcoef(stack_fake_ch2)[0,1]
        
        chan_real[i] = torch.corrcoef(stack_channels_real)[0,1]
        chan_fake[i] = torch.corrcoef(stack_channels_fake)[0,1]
        
    


    if d_fake_logits == None :
        cost = torch.mean( ( msec_real_ch1 - msec_fake_ch1 )**2 +
                          ( msec_real_ch2 - msec_fake_ch2 )**2 +
                          ( chan_real - chan_fake )**2 
                          )
    else:
        prob_true_music  = torch.sigmoid(d_fake_logits)
        cost = torch.mean(  prob_true_music.T @ ( 
                            ( msec_real_ch1 - msec_fake_ch1 )**2 +
                            ( msec_real_ch2 - msec_fake_ch2 )**2 +
                            ( chan_real - chan_fake )**2 
                           ) )
    return cost




def silence_loss_d(real_music, fake_music, keys=33, epsilon=0.001, d_fake_logits = None):
    batch, nchanels, lengths = real_music.shape
    length = lengths // keys
    silence_real = torch.where(real_music < 0.05, 3, real_music)  # Threshold silence
    silence_fake = torch.where(fake_music < 0.05, 3, fake_music)  # Threshold silence

    sr = torch.sum(torch.where(silence_real < 3, 0, real_music),dim = -2)  # Threshold silence
    sf = torch.sum(torch.where(silence_fake< 3, 0, fake_music),dim = -2)  # Threshold silence
  
    sr = torch.where(sr < 3, 0, sr)  # Threshold silence
    sf = torch.where(sr < 3, 0, sf)  # Threshold silence
  
   
  
    if d_fake_logits == None :
        cost = 1
        
    else:
        prob_true_music  = torch.sigmoid(d_fake_logits)
        cost = torch.mean( prob_true_music.T @ ( sr - sf))
    
    return cost 
    



def compute_std_loss_d(real_music, fake_music, keys=33, d_fake_logits = None):
    """
    Computes the Mean Squared Error (MSE) loss for chunkwise standard deviations
    between real and fake music signals.
    
    Args:
        real_music (torch.Tensor): Real music tensor of shape [batch, channels, length].
        fake_music (torch.Tensor): Fake music tensor of shape [batch, channels, length].
        num_chunks (int): Number of chunks to divide the sequence into.
    
    Returns:
        torch.Tensor: MSE loss for chunkwise standard deviations.
    """
    batch_size, n_channels, seq_len = real_music.size()
    chunk_len = seq_len // keys

    # Initialize loss
    mse_loss = 0.0

    for i in range(keys):
        start = i * chunk_len
        end = (i + 1) * chunk_len if i < keys - 1 else seq_len
        
        # Extract chunks for real and fake music
        real_chunk = real_music[:, :, start:end]
        fake_chunk = fake_music[:, :, start:end]
        
        # Compute standard deviations for each chunk
        std_real = real_chunk.std(dim=-1, keepdim=True)
        std_fake = fake_chunk.std(dim=-1, keepdim=True)
        
    # Compute squared differences
    std_real = torch.mean(std_real, dim = 1)
    std_fake = torch.mean(std_fake, dim = 1)
    if d_fake_logits == None :
        mse_loss += torch.mean((std_real - std_fake) ** 2)
    else:
        prob_true_music  = torch.sigmoid(d_fake_logits)
      #  print("prob_true_music.shape", prob_true_music.shape)
      #  print("std_real .shape", std_real .shape)
        mse_loss += torch.mean(  prob_true_music.T @ ( (std_real - std_fake) ** 2))
    # Average loss over chunks
    return mse_loss / keys

def min_max_stats_loss_d(real_music, fake_music, keys=33, d_fake_logits = None,  min_max_weights = 3):
    """
    Computes chunkwise losses between real and fake music signals based on:
    - Average maximum values
    - Average minimum values
    - Mean values

    Args:
        real_music (torch.Tensor): Real music tensor of shape [batch, channels, length].
        fake_music (torch.Tensor): Fake music tensor of shape [batch, channels, length].
        num_chunks (int): Number of chunks to divide the sequence into.

    Returns:
        dict: Loss values for average maximum, average minimum, and mean values.
    """
    batch_size, n_channels, seq_len = real_music.size()
    chunk_len = seq_len // keys

    # Initialize loss values
    avg_max_loss = 0.0
    avg_min_loss = 0.0
    mean_loss = 0.0
    
    max_real  = torch.zeros([batch_size,keys], device=real_music.device)
    min_real  = torch.zeros([batch_size,keys], device=real_music.device)    
    mean_real  = torch.zeros([batch_size,keys], device=real_music.device)

    max_fake  = torch.zeros([batch_size,keys], device=real_music.device)
    min_fake  = torch.zeros([batch_size,keys], device=real_music.device)    
    mean_fake  = torch.zeros([batch_size,keys], device=real_music.device)



    for i in range(keys):
        start = i * chunk_len
        end = (i + 1) * chunk_len if i < keys - 1 else seq_len

        # Extract chunks for real and fake music
        real_chunk = real_music[:, :, start:end]
        fake_chunk = fake_music[:, :, start:end]

        # Compute maximum, minimum, and mean values
        max_real[:,i] = torch.amax(real_chunk, dim=(-2, -1))
        max_fake[:,i] = torch.amax(fake_chunk, dim=(-2, -1))

        min_real[:,i] = torch.amin(real_chunk, dim=(-2, -1))
        min_fake[:,i] = torch.amin(fake_chunk, dim=(-2, -1))

        mean_real[:,i] = real_chunk.mean(dim=(-2, -1))
        mean_fake[:,i] = fake_chunk.mean(dim=(-2, -1))






        # Compute squared differences
        avg_max_loss += torch.mean((max_real - max_fake) ** 2)
        avg_min_loss += torch.mean((min_real - min_fake) ** 2)
        mean_loss += torch.mean((mean_real - mean_fake) ** 2)


    
    if d_fake_logits == None :
        cost = torch.mean( ( max_real - max_fake )**2 +
                          ( min_real - min_fake  )**2 +
                          ( mean_real - mean_fake  )**2 
                          )
    else:
        prob_true_music  = torch.sigmoid(d_fake_logits)
        cost = torch.mean(  prob_true_music.T @ ( min_max_weights * ( max_real - max_fake )**2 +
                          ( min_real - min_fake  )**2 +
                          ( mean_real - mean_fake  )**2 
                          ))


    return cost



def kl_song_divergence(distribution1, distribution2, epsilon=1e-6):
    """
    Compute the KL divergence between two probability distributions.
    
    Args:
        distribution1 (torch.Tensor): First distribution tensor (probabilities).
        distribution2 (torch.Tensor): Second distribution tensor (probabilities).
        epsilon (float): Small value to prevent division by zero or log(0).
        
    Returns:
        torch.Tensor: KL divergence.
    """
    # Safeguard: Ensure inputs are non-negative and sum to 1 (valid probability distributions)
    distribution1 = torch.clamp(distribution1, min=epsilon)
    distribution2 = torch.clamp(distribution2, min=epsilon)
    
    # Normalize distributions to sum to 1 (if they are not already normalized)
    distribution1 = distribution1 / torch.sum(distribution1)
    distribution2 = distribution2 / torch.sum(distribution2)
    
    # Compute KL divergence
    kl_div = torch.sum(distribution1 * torch.log(distribution1 / distribution2))
    return kl_div


def kl_loss_songs_d(real_music, fake_music, d_fake_logits=None, epsilon=1e-6):
    """
    Compute a cost function based on KL divergence for songs.
    
    Args:
        real_music (torch.Tensor): Real music tensor.
        fake_music (torch.Tensor): Fake music tensor.
        d_fake_logits (torch.Tensor, optional): Discriminator logits for fake music.
        epsilon (float): Small value to prevent division by zero or log(0).
        
    Returns:
        torch.Tensor: Computed cost.
    """
    # Safeguard: Ensure the batch size is divisible by 2 to avoid mismatched indexing
    if real_music.size(0) % 2 != 0 or fake_music.size(0) % 2 != 0:
        raise ValueError("Batch size must be even to divide into even and odd indices.")

    # Split real and fake music into even and odd batches
    real_music_even = real_music[::2]  # Selects elements with even batch indexes
    real_music_odd = real_music[1::2]  # Selects elements with odd batch indexes

    fake_music_even = fake_music[::2]  # Selects elements with even batch indexes
    fake_music_odd = fake_music[1::2]  # Selects elements with odd batch indexes

    # Compute KL divergence with safeguards
    real_kl = kl_song_divergence(real_music_even, real_music_odd, epsilon=epsilon)
    fake_kl = kl_song_divergence(fake_music_even, fake_music_odd, epsilon=epsilon)

    # Compute the cost
    if d_fake_logits is None:
        cost = torch.mean((real_kl - fake_kl) ** 2)
    else:
        # Safeguard: Ensure logits are finite and normalized
        d_fake_logits = torch.clamp(d_fake_logits, min=-1e5, max=1e5)
        prob_true_music = torch.sigmoid(d_fake_logits)
        cost = torch.mean(prob_true_music.T @ ((real_kl - fake_kl) ** 2))

    return cost

