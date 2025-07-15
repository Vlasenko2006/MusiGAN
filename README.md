# Musical Generative Adversarial Networks (MusiGANs)

![Example of a fameplay](https://github.com/Vlasenko2006/MusiGAN/tree/main/logo)


I probably took on one of the most challenging tasks for myself: building a GAN neural network capable of generating music from white noise.

### Initial Conditions

1. My training dataset consists of music from my own player.
2. The author (that's me) has no formal musical education.
3. Hardware limitations: a single GPU and about 20 GB of memory.

Let me say right away: I believe I managed to accomplish the task. But more on that later.

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

But what's the challenge then?

In standard GANs, during the first epochs, the generator is still untrained, so its outputs are easy to spot—the discriminator's job seems trivial: just distinguish obvious noise and artifacts from real music. If we were training it on just one genre (say, waltz), it would quickly learn to tell it apart from others. But my dataset is a hodgepodge of genres, so the discriminator has to find common patterns across all music, and that's the main difficulty. For example, what could possibly be common between waltz, reggae, and punk rock when even their rhythmic structures are different?

Musical rules are specific to each genre. For instance, in the 1940s, rock and roll baffled conservative listeners raised on classical music because it broke established conventions, and not everyone recognized it as music. The same goes for later genres like punk rock and others.

In such a situation, the discriminator resembles a musical "conservatory professor": it has to learn to decide what qualifies as music in general, without relying solely on familiar patterns from a single genre.


## Data Preparation

## prepare_dataset.py
Converts .mp3 musical files in your player into numpy vaweformat arrays. Each array represents exactly 2-minute musical pattern. 
## create_dataset.py
General purpose musical dataset creation function. It takes numpy arrays representing muical chunks in vaweformat obtained from **prepare_dataset.py**. Combine pairs of these chunks: [starting musical pattern, continuation of the pattern] and scrambles them, splitting by trainin and validation sets.   



# Dependency Tree and Function Explanations

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

### **piano_main.py**
The main script orchestrating the whole workflow:  
- Loads configuration and datasets  
- Initializes models (Generator and Discriminator)  
- Loads checkpoints if needed  
- Pretrains models  
- Launches the main GAN training loop

---

### **config_reader.py**
**Function:** `load_config_from_yaml`  
Loads configuration parameters from a YAML file, making it easy to manage settings for model training and evaluation.

---

### **split_and_append_chunks.py**
**Function:** `split_and_append_chunks`  
Splits waveform data into smaller, fixed-size chunks and optionally appends them together. Prepares audio data for model input.

---

### **utilities.py**
**Function:** `load_checkpoint`  
Handles loading of saved model checkpoints, allowing training to resume or evaluation of pre-trained models.

---

### **update_checkpoint.py**
**Function:** `update_checkpoint`  
Updates the checkpoint for a given model, typically saving the latest weights or optimizer state.

---

### **piano_model.py**
**Classes:**  
- `VariationalAttentionModel`: The main generator model combining variational autoencoding and attention mechanisms for music generation.
- `AudioDataset`: Processes and normalizes audio data for model input.

**Depends on:**  
- `piano_encoder_decoder.py` for the encoder-decoder architecture.

---

### **piano_encoder_decoder.py**
**Class:** `VariationalEncoderDecoder`  
Defines the encoder and decoder structures used in the generator, supporting variational inference for improved generative modeling.

---

### **discriminator.py**
**Class:** `Discriminator_with_mdisc`  
Implements the discriminator model for the GAN, distinguishing between real and generated audio samples.

---

### **piano_pretrain.py**
**Functions:**  
- `pretrain_generator`: Trains the generator alone before adversarial training.
- `pretrain_discriminator`: Trains the discriminator alone for improved GAN stability.

---

### **piano_train.py**
**Function:** `train_vae_gan`  
Runs the adversarial training loop for the VAE-GAN, handling generator and discriminator updates, loss computation, and checkpointing.

**Depends on:**  
- `vae_utilities.py` (for saving checkpoints)
- `loss_functions.py` (for custom loss functions)

---

### **vae_utilities.py**
**Function:** `save_checkpoint`  
Saves model and optimizer states to disk during training.

---

### **loss_functions.py**
**Functions:**  
Contains custom loss functions such as KL divergence, silence/melody losses, and other domain-specific metrics for training the GAN and VAE.

---




  


# Understanding GAN Loss Values: D, G, Recon & Oscillation

This document explains the key loss values in your VAE-GAN training: Discriminator loss (**D Loss**), Generator loss (**G Loss**), and Reconstruction loss (**Recon**). It also covers what values are considered good, bad, or critical, the meaning of the "confusion point", and why oscillation in these losses is important.

---

## Discriminator Loss (**D Loss**)

**What it measures:**  
How well the discriminator distinguishes real samples from generated (fake) samples.

| D Loss Value        | Interpretation                                                                 |
|---------------------|--------------------------------------------------------------------------------|
| **~0.693**          | **Confusion point**: D is guessing randomly; can't tell real from fake.        |
| **< 0.693**         | **Good**: D is detecting fakes more often than not (learning, but not perfect).|
| **≫ 0.693**         | **Critical/Bad**: D is being fooled more than random (G is overpowering D).    |
| **≪ 0.693** (e.g. <0.2) | **Critical/Bad**: D is dominating and G is struggling to learn.            |

**Summary:**  
- Healthy D Loss usually oscillates between **0.4 and 0.7**.
- Values **much lower than 0.4**: D is too strong.
- Values **much higher than 0.7–0.8**: D is being easily fooled; possibly mode collapse or unstable training.

---

## Generator Loss (**G Loss**)

**What it measures:**  
How well the generator fools the discriminator and reconstructs the data (in VAE-GAN).

| G Loss Value        | Interpretation                                                                 |
|---------------------|--------------------------------------------------------------------------------|
| **~0.693** (BCE part) | **Confusion point**: G is fooling D half the time; D is guessing.            |
| **< 0.693**         | **Good**: G is fooling D more than half the time (rare, but possible).         |
| **≫ 0.693**         | **Bad**: G is being caught by D more often than not.                           |
| **Stably 0 or exploding** | **Critical**: Deadlock or divergence.                                    |

**Note:**  
- In your setup, total G Loss can be higher (e.g., **2–5**) because it also includes reconstruction, KL, and stats losses.
- **Good**: G Loss is stable and gently oscillates in your expected range (e.g., 2–5).
- **Bad**: G Loss is flat (no learning) or explodes (instability).

---

## Reconstruction Loss (**Recon**)

**What it measures:**  
How closely the generated samples match the real data (using Smooth L1 or MSE).

| Recon Value         | Interpretation                                                                 |
|---------------------|--------------------------------------------------------------------------------|
| **Low (~0.4–0.5)**  | **Good**: Generated outputs are close to real samples.                         |
| **High (rising steadily)** | **Bad**: Generator is neglecting reconstruction; possible overfitting to adversarial loss. |
| **Sudden jumps**    | **Critical**: Instability or training issue.                                   |

---

## Confusion Point

- The confusion point for **BCE loss** is **0.693** (i.e., \(-\log(0.5)\)).
- When D Loss or G Loss hovers around this value, it means the discriminator is **guessing randomly**—it’s maximally confused by the generator.

---

## Oscillation

**Oscillation** is a natural part of GAN training:
- **Healthy GANs oscillate:** D and G losses go up and down as each network adapts to the other.
- **Why?**: When D improves, it gets better at catching fakes (D Loss decreases, G Loss increases). Then, G learns to fool D, causing D Loss to rise and G Loss to fall.
- **No oscillation (flat lines):** Indicates a deadlock, mode collapse, or a learning rate problem.
- **Wild oscillation:** May signal instability, but gentle oscillation is a sign of a productive adversarial game.

---

## Quick Reference Table

| Loss      | Good Value Range   | Confusion Point | Critical/Bad Value           |
|-----------|-------------------|-----------------|-----------------------------|
| D Loss    | 0.4 – 0.7         | 0.693           | <<0.4 (D dominates), >>0.7 (D fooled) |
| G Loss    | 2 – 5 (your setup)| 0.693 (BCE part)| Flat or explosive           |
| Recon     | ~0.4 – 0.5        | —               | Rising/unstable             |

---

## Summary

- **Healthy training:** All losses gently oscillate within their expected ranges.
- **Unhealthy:** Any loss is flat, exploding, or always at confusion point.
- **Goal:** D and G should continually challenge each other, resulting in creative, realistic outputs.

---


## Check_discriminator.py


# Discriminator Loss Trend Analysis for Generator Checkpoint 690

## Overview
Cheks the discriminator's performance over epochs for a given generator's checkpoint (690 in the example below).


![Discriminator_over_epochs](https://github.com/Vlasenko2006/MusiGAN/blob/main/discriminator_over_epochs.png)


## Interpretation

- **Early Discriminators (Low Epochs, Very Low Loss):**
  - Discriminators trained at earlier epochs are easily fooled by the generator at epoch 690.
  - Their loss is extremely low, indicating they classify the generator’s fakes as "real" with high confidence.
  - This is expected, as the generator has adapted far beyond what these early discriminators have seen.

- **Mid-to-Late Discriminators (Rising Loss):**
  - As you progress to more recent discriminator checkpoints, their loss when judging fakes rises steadily.
  - This means newer discriminators are better at detecting the generator's fakes—i.e., they are not as easily fooled.
  - The generator's samples look less "real" to these more recent discriminators.

- **Latest Discriminators (High Loss):**
  - The most recent discriminators (epochs 690, 700) have the highest loss.
  - This indicates the discriminators are correctly identifying the generator's fakes as "fake" more often.

## What This Says About Training

- **Healthy GAN Dynamics:**
  - The generator is able to fool older discriminators but is challenged by newer ones. This "arms race" is at the heart of adversarial training.
- **No Mode Collapse or Instability:**
  - If all losses stayed low, your discriminator would not be learning. If all were high, the generator would be failing. The smooth, rising trend is desirable.
- **Progression:**
  - The generator and discriminator are pushing each other to improve, which is the sign of a healthy training regime.

## Summary Table

| Discriminator Epoch | Mean d_loss (label=real) | Interpretation                      |
|---------------------|--------------------------|-------------------------------------|
| 540 - ~590          | ~0.0003 → ~0.0003        | Discriminator fooled (old D)        |
| ~600 - ~640         | 0.0025 → 0.0089          | D catching up, less fooled          |
| 650 - 690           | 0.0458 → 0.5882          | D strong, generator challenged      |

## Conclusion

Your GAN training is progressing well, with generator and discriminator in a healthy adversarial contest. This is the ideal pattern for productive GAN development: the generator outsmarts old discriminators, but is pushed by new ones to improve further.
