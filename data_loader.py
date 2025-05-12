import torch
import torchaudio
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import json

class SegmentPairDataset:
    """Dataset for relative complexity prediction between adjacent segments.
    
    Args:
        audio_dir: Path to directory containing audio files
        labels_csv: Path to CSV file containing complexity labels
        sample_rate: Target sample rate for audio
        device: Device to load data to (cpu/cuda)
        context_ratio: Ratio of previous segment to include as context
    """
    def __init__(self, 
                audio_dir: str = "dataset/verified_mp3",
                labels_csv: str = "dataset/labels.csv",
                sample_rate: int = 24000,
                device: str = "cuda",
                context_ratio: float = 1.0):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.device = device
        self.context_ratio = context_ratio
        
        # Load labels
        self.labels_df = pd.read_csv(labels_csv)
        
        # Verify audio files exist
        audio_files = set(f.name for f in self.audio_dir.glob("*.mp3"))
        self.labels_df = self.labels_df[
            self.labels_df["name"].apply(lambda x: f"{x}.mp3" in audio_files)
        ]
        
        # Parse segment boundaries and complexities
        self.segments = self._parse_segments()
        
    def _parse_segments(self) -> List[Dict]:
        """Parse audio segments from labels dataframe."""
        segments = []
        for _, row in self.labels_df.iterrows():
            boundaries = json.loads(row["boundary"])
            
            # Handle empty or NaN complexity values
            if pd.isna(row["complexity"]) or not row["complexity"].strip():
                complexities = [0] * len(boundaries)  # Default zero complexity
            else:
                complexities = json.loads(row["complexity"])
            
            for i in range(1, len(boundaries)):
                segments.append({
                    "audio_file": row["name"] + ".mp3",
                    "prev_start": boundaries[i-1],
                    "current_start": boundaries[i],
                    "prev_complexity": complexities[i-1],
                    "current_complexity": complexities[i],
                    "delta_complexity": complexities[i] - complexities[i-1]
                })
        return segments
        
    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load adjacent segment pair and complexity delta.
        
        Returns:
            tuple: (prev_segment, current_segment, delta_complexity)
                  prev_segment: shape (1, samples)
                  current_segment: shape (1, samples) 
                  delta_complexity: scalar tensor
        """
        segment = self.segments[idx]
        audio_path = self.audio_dir / segment["audio_file"]
        
        # Load full audio
        waveform, orig_sr = torchaudio.load(str(audio_path))
        
        # Convert to mono and resample
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sample_rate)
        
        # Extract segments
        prev_start = int(segment["prev_start"] * self.sample_rate)
        current_start = int(segment["current_start"] * self.sample_rate)
        
        prev_segment = waveform[:, prev_start:current_start]
        current_segment = waveform[:, current_start:]
        
        # Apply context ratio to previous segment
        context_samples = int(prev_segment.shape[1] * self.context_ratio)
        prev_segment = prev_segment[:, -context_samples:]
        
        # Prepare outputs
        delta_complexity = torch.tensor(
            segment["delta_complexity"],
            dtype=torch.float32,
            device=self.device)
        
        return prev_segment.to(self.device), current_segment.to(self.device), delta_complexity
        
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for DataLoader with variable length segments.
        
        Args:
            batch: List of (prev_segment, current_segment, delta_complexity) tuples
            
        Returns:
            tuple: (prev_segments, current_segments, deltas, masks)
                  prev_segments: padded batch (B, 1, T1)
                  current_segments: padded batch (B, 1, T2)
                  deltas: batch of complexity deltas (B,)
                  masks: attention masks for both segments (B, 2, max(T1,T2))
        """
        prev_segments, current_segments, deltas = zip(*batch)
        
        # Get max lengths
        max_prev_len = max(p.shape[1] for p in prev_segments)
        max_current_len = max(c.shape[1] for c in current_segments)
        max_len = max(max_prev_len, max_current_len)
        
        # Pad segments and create masks
        padded_prev = []
        padded_current = []
        masks = []
        
        for p, c in zip(prev_segments, current_segments):
            # Pad segments
            pad_prev = max_len - p.shape[1]
            pad_current = max_len - c.shape[1]
            
            padded_prev.append(torch.nn.functional.pad(p, (0, pad_prev)))
            padded_current.append(torch.nn.functional.pad(c, (0, pad_current)))
            
            # Create combined mask (1=valid, 0=padding)
            mask = torch.zeros((2, max_len), dtype=torch.float32)
            mask[0, :p.shape[1]] = 1  # prev segment mask
            mask[1, :c.shape[1]] = 1  # current segment mask
            masks.append(mask)
        
        # Stack tensors
        prev_segments = torch.stack(padded_prev)
        current_segments = torch.stack(padded_current)
        deltas = torch.stack(deltas)
        masks = torch.stack(masks)
        
        return prev_segments, current_segments, deltas, masks
