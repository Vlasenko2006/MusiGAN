# Musical Generative Adversarial Networks (MusiGANs)

![Example of a fameplay](https://github.com/Vlasenko2006/MusiGAN/blob/main/logo/1.jpg)


I probably took on one of the most challenging tasks for myself: building a GAN neural network capable of generating music from white noise.

### Initial Conditions

1. My training dataset consists of music from my own player.
2. The author (that's me) has no formal musical education.
3. Hardware limitations: a single GPU and about 20 GB of memory.

Let me say right away: I believe I managed to accomplish the task. But more on that later. Check out musical samples crearted by my neural network 

🎵 **[Music samples](
https://github.com/Vlasenko2006/MusiGAN/tree/main/Music_samples)** 🎵

---

To solve my problem, I used Generative Adversarial Networks (GANs).

#### Generative Adversarial Networks in Simple Terms

A GAN consists of a pair of neural networks that learn from each other. One network, the generator, tries to create artificial data (for example, images or music), while the other, the discriminator, learns to distinguish real data from fake. The generator gradually gets better at "fooling" the discriminator, creating increasingly realistic examples, while the discriminator becomes more skilled at spotting fakes. In the process, both networks improve: one in making "realistic" music, the other in identifying forgeries.

---

### How the Music Generator Works

A neural network learns by example: if you show it thousands of musical pieces, it absorbs the concepts of melody, rhythm, and harmony. For instance, I have a network that takes a 10-second music fragment and continues it. But with a GAN, the task is tougher: the network must turn white noise into music. How does it do that? From my observations, during training it "remembers" musical structures (I suspect these are stored in the convolutional layers of the decoder). At the generation stage, the input white noise—essentially just random numbers—determines which patterns (rhythmic, melodic, etc.) the network will "assemble" from these remembered elements. Theoretically, overlaying random patterns should result in cacophony (try mixing several tracks together—you get noise). But here's where the "magic" happens: the transformer, having learned musical rules, selects patterns that both follow musical grammar and are compatible with the "request" made by the white noise.

---

### How the Music Discriminator Works (and Why It's Hard to Train on Diverse Music)

In general, the music discriminator operates much like any other discriminator in a GAN: it receives two signals—one "genuine" (real music from the dataset) and one fake, produced by the generator. It's important to realize that the quality of the generator's output directly depends on the quality of the discriminator: if the discriminator doesn't learn to recognize what music is in general, the generator will never be able to create convincing fakes.

### But what's the challenge then?

In standard GANs, during the first epochs, the generator is still untrained, so its outputs are easy to spot—the discriminator's job seems trivial: just distinguish obvious noise and artifacts from real music. If we were training it on just one genre (say, waltz), it would quickly learn to tell it apart from others. But my dataset is a hodgepodge of genres, so the discriminator has to find common patterns across all music, and that's the main difficulty. For example, what could possibly be common between waltz, reggae, and punk rock when even their rhythmic structures are different?

Musical rules are specific to each genre. For instance, in the 1940s, rock and roll baffled conservative listeners raised on classical music because it broke established conventions, and not everyone recognized it as music. The same goes for later genres like punk, rock and others.

In such a situation, the discriminator resembles a musical "conservatory professor": it has to learn to decide what qualifies as music in general, without relying solely on familiar patterns from a single genre.


# Data Preparation

The creation of any neural network begins with the careful selection and analysis of the data it will work with. The most straightforward approach would be to use songs in MIDI format as NN's output, since this format is well-structured and widely used for representing musical information. However, in recent years, access to open libraries of MIDI compositions has become significantly restricted, so I had to develop my own method for converting musical pieces into an original format, essentially analogous to MIDI but tailored to the specifics of my task. The MusiGAN uses this format as the output.

Before diving into the technical details, let’s look at music through the eyes of someone with a mathematical, but not musical, background.

---

## What is music from a mathematical perspective?

I perceive music primarily as a mathematical structure. Music consists of notes, and notes are specific frequencies of sound waves. Music has rhythm, melody, and harmony, all of which can be described mathematically. For example, in the lower octaves, the difference between adjacent notes is about 10–20 Hz, while in the higher octaves this difference increases to hundreds of hertz.

At first glance, one might assume that a well-tuned instrument reproduces only these strictly defined frequencies. But the physics of solid bodies makes its own adjustments: any musical instrument produces not only the fundamental frequency (the main tone), but also a set of harmonics—additional frequencies with lower amplitude. The number, distribution, and amplitude of these harmonics determine the timbre and richness of each instrument’s sound.

---

## Translating music into the "language" of piano keys

For analysis, I decomposed musical pieces into frequencies corresponding to the notes on a piano keyboard. However, as mentioned above, no instrument is limited to producing sound at just one frequency—each has its own “fingerprint” of harmonics. The pattern of these additional frequencies is unique for each instrument. In my work, I chose a Gaussian distribution as a model, which, as my experiments showed, describes the spectrum of, for example, a jaw harp quite well.

## Data pre-postprocessing functions

## `prepare_dataset.py`
Converts .mp3 musical files in your player into numpy vaweformat arrays. Each array represents exactly 2-minute musical pattern. 

## `create_dataset.py`
General purpose musical dataset creation function. It takes numpy arrays representing muical chunks in waveformat obtained from **prepare_dataset.py**. Combine pairs of these chunks: [starting musical pattern, continuation of the pattern] and scrambles them, splitting by trainin and validation sets.   

## `numpy_2_piano_keys.py`
Converts each waveformat chunk generated by `create_dataset.py` into the chunks with the corresponding piano keys. 

## `piano_keys_2_mp3.py`
Converts chunks with piano keys generated either by `numpy_2_piano_keys.py` or by MusiGAN back into `.mp3` format. 


# MusiGAN Dependency Tree and Function Explanations

## Dependency Tree

```
piano_main.py
  ├─ config_reader.py           (load_config_from_yaml)
  ├─ split_and_append_chunks.py (split_and_append_chunks)
  ├─ utilities.py               (load_checkpoint)
  ├─ update_checkpoint.py       (update_checkpoint)
  ├─ piano_model.py
  │    └─ piano_encoder_decoder.py (VariationalEncoderDecoder, intantiates in piano_main.py)
  ├─ discriminator.py
  ├─ piano_pretrain.py          (pretrain_generator, pretrain_discriminator)
  └─ piano_train.py
       ├─ vae_utilities.py      (save_checkpoint)
       └─ loss_functions.py     (kl_divergence_loss, no_seconds_loss, etc.)
```

---

## Brief Explanation of Each Module/Function

### `piano_main.py`
The main script orchestrating the whole workflow:  
- Loads configuration and datasets  
- Initializes models (Generator and Discriminator)
- Initializes optimizers for generator and discriminator: `g_optimizer` and `d_optimizer` respectively.
- Initializes loss criterions:
  --pretrain_criterion `nn.SmoothL1Loss()`
  --gan_criterion `nn.BCEWithLogitsLoss()` 
- Loads checkpoints if needed  
- Pretrains models  
- Launches the main GAN training loop

---

### `config_reader.py`
**Function:** `load_config_from_yaml`  
Loads configuration parameters from a YAML file, making it easy to manage settings for model training and evaluation. It holds model hyperparameters, training parameters etc.

---

### `split_and_append_chunks.py`
**Function:** `split_and_append_chunks`  
Splits waveform data into smaller, fixed-size chunks and optionally appends them together. Prepares audio data for model input.

---

### `utilities.py`
**Function:** `load_checkpoint`  
Handles loading of saved model checkpoints, allowing training to resume or evaluation of pre-trained models.

---

### **update_checkpoint.py**
**Function:** `update_checkpoint`  
Updates the checkpoint for a given model, typically saving the latest weights or optimizer state.

---

### `piano_model.py`

Represents the Generator with the following structure:
**1. Generator:**  
- Variational encoder → transformer → decoder.  
- The encoder and decoder are built as sequences of Conv-ReLU-BatchNorm layers with skip connections to prevent gradient vanishing.  
- At the decoder output: normalization → Hard Sigmoid → Adaptive Pooling.

**Classes:**  
- `VariationalAttentionModel`: The main generator model combining variational autoencoding and attention mechanisms for music generation.
- `AudioDataset`: Processes and normalizes audio data for model input.

**Depends on:**  
- `piano_encoder_decoder.py` for the encoder-decoder architecture.

---

### `piano_encoder_decoder.py`
**Class:** `VariationalEncoderDecoder`  
Defines the encoder and decoder structures used in the generator, supporting variational inference for improved generative modeling.

---

### `discriminator.py`
Represents the discriminator with the following structure:
- Multiple blocks of Conv-ReLU-BatchNorm layers with skip connections.  
- The penultimate layer performs Batch Feature Matching.
**Class:** `Discriminator_with_mdisc`  
Implements the discriminator model for the GAN, distinguishing between real and generated audio samples.

---

### `piano_pretrain.py`
**Functions:**  
- `pretrain_generator`: Trains the generator alone before adversarial training.
- `pretrain_discriminator`: Trains the discriminator alone for improved GAN stability.

---

### `piano_train.py`
**Function:** `train_vae_gan`  
Runs the adversarial training loop for the VAE-GAN, handling generator and discriminator updates, loss computation, and checkpointing. This trainer orchestrates the training process of a VAE-GAN model for music generation, coordinating both the generator and the discriminator. The workflow includes adversarial training, reconstruction, and a suite of domain-specific regularization losses.

#### Optimizers
- **Generator Optimizer:**  
  The generator's parameters are updated via a user-supplied optimizer (`g_optimizer`), typically an Adam or AdamW optimizer in PyTorch use cases.
- **Discriminator Optimizer:**  
  The discriminator is trained with a separate optimizer (`d_optimizer`), also user-supplied Adam-based for stable GAN training.

#### Loss Functions (Criterions)

#### Adversarial and Reconstruction Losses
- **Adversarial Loss (`criterion_gan`)**  
  `torch.nn.BCEWithLogitsLoss` is used as the primary GAN objective for both generator and discriminator, promoting realistic generation and robust discrimination.
- **Reconstruction Loss (`criterion_reconstruction`)**  
  `torch.nn.SmoothL1Loss` is used to encourage accurate reconstruction of real music signals.

#### Generator-Specific Losses
- **KL Divergence Loss**  
  Regularizes the latent space of the variational encoder (`kl_divergence_loss`).
- **No-Seconds Loss, No-Silence Loss**  
  Penalize excessive silence and unwanted gaps in generated music (`no_seconds_loss`, `no_silence_loss`).
- **Melody and Statistical Losses**  
  Encourage melodic structure and match statistical properties of the data (`melody_loss`, `melody_loss_d`, `compute_std_loss_d`, `min_max_stats_loss_d`).

#### Discriminator-Specific Losses
- **Silence and Melody Losses**  
  Penalize unrealistic silence or lack of melody in generated samples (`silence_loss_d`, `melody_loss_d`).
- **Statistical and KL Losses**  
  Enforce statistical similarity and regularization (`compute_std_loss_d`, `min_max_stats_loss_d`, `kl_loss_songs_d`).

### Training Procedure
- **Epoch Loop:** Alternates between generator and discriminator updates per batch, with support for gradient accumulation (`accumulation_steps`) for large-scale or memory-constrained training.
- **Loss Weighting:** All losses are weighted via a `loss_weights` dictionary, allowing fine-grained tuning of the training signal.
- **Gradient Clipping:** Optional clipping is present (commented out), suggesting mitigation for exploding gradients.
- **Checkpointing:** Model and optimizer states are periodically saved for reproducibility and evaluation.

### Additional Features
- **Label Smoothing:** Optional smoothing of real labels for adversarial robustness.
- **Freeze Options:** The discriminator can be frozen for part of the training schedule.
- **Loss Tracking:** Extensive per-loss tracking for diagnostics and visualization.
- **Adaptive Training:** Number of discriminator steps per generator update is configurable.


**Depends on:**  
- `vae_utilities.py` (for saving checkpoints)
- `loss_functions.py` (for custom loss functions)

---

### `vae_utilities.py`
**Function:** `save_checkpoint`  
Saves model and optimizer states to disk during training.

---

### `loss_functions.py`

To assist the discriminator, I created “helpers”—special additional terms in its cost function that require any signal classified as “music” to conform to certain basic statistical and structural properties of an audio signal. As a result, the discriminator not only distinguishes between real and generated examples, but also learns to pay attention to key characteristics of music, such as dynamics, variability, the absence of excessive silence or repetition, and other musical patterns. This approach makes training more stable and allows the generator to produce more meaningful and musical results. All these “helpers” are coded here:

**Functions:**  
Contains custom loss functions such as KL divergence, silence/melody losses, and other domain-specific metrics for training the GAN and VAE.

---
## How to Set Up the Environment and Install the Model

The installation and environment setup process is similar to the instructions provided here: [https://github.com/Vlasenko2006/NeuroNMusE](https://github.com/Vlasenko2006/NeuroNMusE)


  
