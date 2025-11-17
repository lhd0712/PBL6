# rtfm_dataset.py
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

class RTFMDataset(Dataset):
    def __init__(self, list_file):
        self.video_paths = []
        self.labels = []
        
        with open(list_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                # (SỬA) Giữ nguyên nhãn 0-13
                self.labels.append(int(label)) 
                self.video_paths.append(path)
                
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        features = np.load(path)
        
        num_clips = features.shape[0]
        target_clips = 32
        
        if num_clips > target_clips:
            indices = np.random.choice(num_clips, target_clips, replace=False)
            indices.sort()
            features = features[indices]
        elif num_clips < target_clips:
            indices = np.arange(num_clips)
            indices = np.tile(indices, (target_clips // num_clips) + 1)[:target_clips]
            indices.sort()
            features = features[indices]
            
        # (QUAN TRỌNG) label phải là kiểu Long (int64) cho CrossEntropyLoss
        return torch.from_numpy(features).float(), torch.tensor(label).long()
    
# Sampler đảm bảo 50% Normal - 50% Abnormal trong 1 batch
class BalancedSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Tự động lọc index dựa trên nhãn nhị phân
        self.normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        self.abnormal_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            # Lấy n/2 Normal
            normal_sample = np.random.choice(self.normal_indices, self.batch_size // 2, replace=True)
            # Lấy n/2 Abnormal
            abnormal_sample = np.random.choice(self.abnormal_indices, self.batch_size // 2, replace=True)
            
            batch_indices = np.concatenate([normal_sample, abnormal_sample])
            np.random.shuffle(batch_indices)
            
            yield batch_indices.tolist()

    def __len__(self):
        return self.num_batches