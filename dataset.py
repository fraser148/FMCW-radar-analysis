import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
import radar
import os

class RadarDataset(Dataset):
  def __init__(self, root, transform=None):
    self.root = root
    timestamps_path = os.path.join(os.path.join(root, 'radar.timestamps'))
    self.radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
    self.transform = transform

  def __getitem__(self, idx):
    if is_tensor(idx):
      idx = idx.tolist()

    filename = os.path.join(self.root, "radar", str(self.radar_timestamps[idx]) + '.png')
    timestamps, azimuths, valid, fft_data, radar_resolution = radar.load_radar(filename)
    sample = fft_data

    if self.transform:
      sample = self.transform(sample)
    
    return sample

  def __len__(self):
    return len(self.radar_timestamps)
    