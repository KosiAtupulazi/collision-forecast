# Defines the CollisionForecastDataset class

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

#self: memory of this dataset
class CollisionForecastDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None):
        self.data = pd.read_csv(csv_path)
        self.split = split
        self.transform = transform # Optional transform
    
    #when the model wants a clip at index
    def __getitem__(self, index):
        row = self.data.iloc[index] #row[index] of the csv
        project_root = os.path.dirname(os.path.dirname(__file__))  # gets path to /collision-detection
        # builds the full file path by combining the label (as folder) and clip name, e.g., 'frames/crash/00822_clip.npy' from the csv
        clip_path = os.path.join(project_root, 'frames', self.split, row['label'], row['clip_name']) 
        clip = np.load(clip_path) #loading the npy files
        
        # Normalize using ImageNet pretrained VideoMAE stats
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        
        clip = clip / 255.0  # Scale from [0,255] to [0,1]
        clip = (clip - mean) / std  # Normalize channel-wise

        #apply the transformation if its set
        if self.transform:
            clip = self.transform(clip) # Apply transform if provided (e.g normalization)

        # For regression: return time_of_alert value
        # Check if time_of_alert column exists, otherwise use a default value
        if 'time_of_alert' in row:
            time_value = row['time_of_alert']
        else:
            # If no time_of_alert column, use label as proxy (crash = 0 seconds, no_crash = some positive value)
            time_value = 0.0 if row['label'] == 'crash' else 5.0  # Default values
        
        # clip 
        # - float32 because it normalizes the pixels to b/w 0, 1 rather than 0-255, which allows for blending
        #   and helps the model understand better
        # time_value
        # - float32 for regression (continuous value)

        return torch.tensor(clip, dtype=torch.float32), torch.tensor(time_value, dtype=torch.float32) #returns the clip and the time value in pytorch format
        
    def __len__(self):
        return len(self.data) #returns the length of the dataset