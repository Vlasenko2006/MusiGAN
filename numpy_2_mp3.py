#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:25:45 2025

@author: andreyvlasenko
"""

import os
import librosa
import soundfile as sf
from pydub import AudioSegment



# Ensure ffmpeg and ffprobe are configured
os.environ["PATH"] += os.pathsep + "/usr/local/bin"
AudioSegment.converter = "/usr/local/bin/ffmpeg"
AudioSegment.ffprobe = "/usr/local/bin/ffprobe"

def mp3_to_numpy(mp3_file, target_sr=12000):
    """
    Converts an MP3 file to a NumPy array.
    Parameters:
    - mp3_file: Path to the input MP3 file.
    - target_sr: Target sample rate for the waveform (default = 16 kHz).
    Returns:
    - waveform: NumPy array of audio data.
    - sample_rate: Sample rate of the audio.
    """
    # Load the MP3 file as a waveform
    waveform, sample_rate = librosa.load(mp3_file, sr=target_sr, mono=False)
    print(f"Converted MP3 to NumPy array: {waveform.shape}, Sample Rate: {sample_rate}")
    return waveform, sample_rate

def numpy_to_mp3(waveform, sample_rate, output_mp3_file="output.mp3"):
    """
    Converts a NumPy array back to an MP3 file.
    Parameters:
    - waveform: NumPy array of audio data.
    - sample_rate: Sample rate of the waveform.
    - output_mp3_file: Path to the output MP3 file.
    """
    # Save the NumPy array as a .wav file
    temp_wav_file = "temp_output.wav"
    sf.write(temp_wav_file, waveform.T, sample_rate)  # Transpose if stereo

    # Convert the .wav file to .mp3 using pydub
    audio = AudioSegment.from_wav(temp_wav_file)
    audio.export(output_mp3_file, format="mp3")
    print(f"Converted NumPy array to MP3: {output_mp3_file}")

    # Remove the temporary .wav file
    os.remove(temp_wav_file)


