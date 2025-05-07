import torch
import torchaudio
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import numpy as np

class ComplexityDataset:
    """Dataset for audio complexity prediction with EnCodec.
    
    Args:
        audio_dir: Path to directory containing audio files
        labels_csv: Path to CSV file containing complexity labels
        sample_rate: Target sample rate for audio
        device: Device to load data to (cpu/cuda)
    """
    def __init__(self, 
                 audio_dir: str = "dataset/verified_mp3",
                 labels_csv: str = "dataset/labels.csv",
                 sample_rate: int = 24000,
                 device: str = "cuda"):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.device = device
        
        # Load and filter labels
        self.labels_df = pd.read_csv(labels_csv)
        self.labels_df = self.labels_df[self.labels_df["状态"] == "确定链接正确"]
        
        # Build file list
        self.audio_files = list(self.audio_dir.glob("*.mp3"))
        
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load audio segment and corresponding complexity label.
        
        Returns:
            tuple: (audio_tensor, complexity_score)
                  audio_tensor: shape (1, samples)
                  complexity_score: shape (n_segments,)
        """
        # Load audio
        audio_path = self.audio_files[idx]
        waveform, orig_sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate)
            
        # Get corresponding label data
        file_id = audio_path.stem
        label_row = self.labels_df[self.labels_df["name"] == file_id].iloc[0]
        
        # Convert boundary points to sample indices
        boundaries = eval(label_row["boundary"])
        boundaries = [int(t * self.sample_rate) for t in boundaries]
        
        # Get complexity scores
        complexity = torch.tensor(
            eval(label_row["complexity"]), 
            dtype=torch.float32,
            device=self.device)
        
        # Move audio to device
        waveform = waveform.to(self.device)
        
        return waveform, complexity
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for DataLoader.
        
        Args:
            batch: List of (waveform, complexity) tuples
            
        Returns:
            tuple: (waveforms, complexities)
                  waveforms: padded batch of audio tensors
                  complexities: stacked complexity tensors
        """
        waveforms, complexities = zip(*batch)
        
        # Pad waveforms to max length in batch
        waveforms = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True)
            
        # Stack complexities
        complexities = torch.stack(complexities)
        
        return waveforms, complexities
