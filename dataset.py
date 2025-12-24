import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ConditionalDataset(Dataset):
    def __init__(self, root):
        self.source_paths = sorted([os.path.join(root, "source", f) for f in os.listdir(os.path.join(root, "source"))])
        self.target_paths = sorted([os.path.join(root, "target", f) for f in os.listdir(os.path.join(root, "target"))])
        self.cond_d_paths = sorted([os.path.join(root, "condition_d", f) for f in os.listdir(os.path.join(root, "condition_d"))])
        self.cond_v_paths = sorted([os.path.join(root, "condition_v", f) for f in os.listdir(os.path.join(root, "condition_v"))])

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, idx):
        source = np.load(self.source_paths[idx]).astype(np.float32)
        target = np.load(self.target_paths[idx]).astype(np.float32)
        cond_d = np.load(self.cond_d_paths[idx]).astype(np.float32)
        cond_v = np.load(self.cond_v_paths[idx]).astype(np.float32)

        return (
            torch.from_numpy(source),  # input image
            torch.from_numpy(target),  # GT (for supervision, shape: [1,128,128])
            torch.from_numpy(cond_d),
            torch.from_numpy(cond_v),
        )
