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
        # Load audio and convert to mono if needed
        audio_path = self.audio_files[idx]
        waveform, orig_sr = torchaudio.load(str(audio_path))
        
        # Convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:  # Check if stereo (2 channels)
            waveform = waveform.mean(dim=0, keepdim=True)
            
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
        complexity_val = label_row["complexity"]
        if pd.isna(complexity_val) or complexity_val == "":
            complexity_val = "[5]"  # Default value if empty
        if isinstance(complexity_val, str) and complexity_val.startswith("["):
            complexity_val = eval(complexity_val)
        complexity = torch.tensor(
            complexity_val,
            dtype=torch.float32,
            device=self.device)
        
        # Move audio to device
        waveform = waveform.to(self.device)
        
        return waveform, complexity
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for DataLoader with variable length audio and complexities.
        
        Args:
            batch: List of (waveform, complexity) tuples
            
        Returns:
            tuple: (waveforms, complexities, masks)
                  waveforms: padded batch of audio tensors (B, 1, T)
                  complexities: padded batch of complexity tensors (B, N)
                  masks: attention mask for padded positions (B, 1, T)
        """
        waveforms, complexities = zip(*batch)
        
        # Get max lengths
        max_audio_len = max(w.shape[1] for w in waveforms)
        max_complexity_len = max(c.shape[0] for c in complexities)
        
        # Pad waveforms and create masks
        padded_waveforms = []
        masks = []
        for w in waveforms:
            # Pad waveform
            padding = max_audio_len - w.shape[1]
            padded = torch.nn.functional.pad(w, (0, padding))
            padded_waveforms.append(padded)
            
            # Create mask (1 for real, 0 for padded)
            mask = torch.ones((1, w.shape[1]), dtype=torch.float32)
            if padding > 0:
                mask = torch.nn.functional.pad(mask, (0, padding))
            masks.append(mask)
        
        # Pad complexities
        padded_complexities = []
        for c in complexities:
            padding = max_complexity_len - c.shape[0]
            padded = torch.nn.functional.pad(c, (0, padding), value=0)  # Pad with 0
            padded_complexities.append(padded)
        
        # Stack tensors
        waveforms = torch.stack(padded_waveforms)
        complexities = torch.stack(padded_complexities)
        masks = torch.stack(masks)
        
        return waveforms, complexities, masks
