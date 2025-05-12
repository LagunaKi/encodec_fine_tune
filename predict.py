import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np

from fine_tune.model_ext import RelativeComplexityModel
from fine_tune.data_loader import SegmentPairDataset

def load_model(model_path: str, device: str = "cuda") -> RelativeComplexityModel:
    """Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model to
        
    Returns:
        Loaded model in eval mode
    """
    model = RelativeComplexityModel(freeze_encoder=True).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_segment_complexity(model: RelativeComplexityModel,
                             prev_seg_path: str,
                             curr_seg_path: str,
                             sample_rate: int = 24000) -> float:
    """Predict complexity score for an audio file.
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Predicted complexity score
    """
    # Load and preprocess segments
    def load_segment(path):
        waveform, orig_sr = torchaudio.load(path)
        if orig_sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sample_rate)
        return waveform.unsqueeze(0).to(next(model.parameters()).device)
    
    prev_seg = load_segment(prev_seg_path)
    curr_seg = load_segment(curr_seg_path)
    
    # Predict delta
    with torch.no_grad():
        delta = model(prev_seg, curr_seg)
    
    return delta.item()

def plot_comparison(prev_seg_path: str,
                  curr_seg_path: str,
                  delta: float,
                  save_path: str = None):
    """Plot waveform with complexity score.
    
    Args:
        audio_path: Path to audio file
        complexity: Predicted complexity score
        save_path: Path to save plot (optional)
    """
    # Load segments for plotting
    def load_plot_data(path):
        wav, sr = torchaudio.load(path)
        return wav.numpy()[0], sr
    
    prev_wav, sr = load_plot_data(prev_seg_path)
    curr_wav, _ = load_plot_data(curr_seg_path)
    
    # Create plot
    plt.figure(figsize=(16, 6))
    
    # Plot previous segment
    plt.subplot(2, 1, 1)
    times = np.arange(len(prev_wav)) / sr
    plt.plot(times, prev_wav, alpha=0.6, color='blue')
    plt.title("Previous Segment")
    plt.grid(True, alpha=0.3)
    
    # Plot current segment
    plt.subplot(2, 1, 2)
    times = np.arange(len(curr_wav)) / sr
    plt.plot(times, curr_wav, alpha=0.6, color='orange')
    plt.title(f"Current Segment\nPredicted ΔComplexity: {delta:.2f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict audio complexity')
    parser.add_argument('prev_seg_path', type=str, help='Path to previous segment')
    parser.add_argument('curr_seg_path', type=str, help='Path to current segment')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to trained model')
    parser.add_argument('--save_plot', type=str, default=None,
                      help='Path to save plot (optional)')
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)
    
    # Predict complexity
    delta = predict_segment_complexity(model, args.prev_seg_path, args.curr_seg_path)
    print(f"Predicted complexity delta: {delta:.2f}")
    
    # Plot comparison
    plot_comparison(args.prev_seg_path, args.curr_seg_path, delta, args.save_plot)

if __name__ == '__main__':
    main()
