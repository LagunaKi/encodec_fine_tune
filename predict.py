# Copyright (c) 2025 Your Name
# All rights reserved.

import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

from .model_ext import EncodecComplexityModel
from .data_loader import ComplexityDataset

def load_model(model_path: str, device: str = "cuda") -> EncodecComplexityModel:
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model to
        
    Returns:
        Loaded model in eval mode
    """
    model = EncodecComplexityModel(freeze_encoder=True).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_audio_complexity(model: EncodecComplexityModel, 
                           audio_path: str,
                           sample_rate: int = 24000) -> float:
    """Predict complexity score for an audio file.
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Predicted complexity score
    """
    # Load and preprocess audio
    waveform, orig_sr = torchaudio.load(audio_path)
    if orig_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_sr, sample_rate)
    
    # Add batch dimension and move to model device
    waveform = waveform.unsqueeze(0).to(next(model.parameters()).device)
    
    # Predict
    with torch.no_grad():
        complexity = model.predict_complexity(waveform)
    
    return complexity.item()

def plot_complexity(audio_path: str, 
                   complexity: float,
                   save_path: str = None):
    """Plot waveform with complexity score.
    
    Args:
        audio_path: Path to audio file
        complexity: Predicted complexity score
        save_path: Path to save plot (optional)
    """
    # Load audio for plotting
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.numpy()[0]
    times = np.arange(len(waveform)) / sr
    
    # Create plot
    plt.figure(figsize=(12, 4))
    plt.plot(times, waveform, alpha=0.6)
    plt.title(f"Audio Waveform\nPredicted Complexity: {complexity:.2f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict audio complexity')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to trained model')
    parser.add_argument('--save_plot', type=str, default=None,
                      help='Path to save plot (optional)')
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)
    
    # Predict complexity
    complexity = predict_audio_complexity(model, args.audio_path)
    print(f"Predicted complexity: {complexity:.2f}")
    
    # Plot results
    plot_complexity(args.audio_path, complexity, args.save_plot)

if __name__ == '__main__':
    main()
