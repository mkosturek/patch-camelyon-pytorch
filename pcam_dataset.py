import h5py
import torch
import torch.utils.data as data
from pathlib import Path

class PatchCamelyonDataset(data.Dataset):
    
    def __init__(self, root, image_set):
        data_path = Path(root) / f"camelyonpatch_level_2_split_{image_set}_x.h5"
        target_path = Path(root) / f"camelyonpatch_level_2_split_{image_set}_y.h5"
        self.data = h5py.File(data_path)["x"]
        self.target = h5py.File(target_path)["y"]
        
    def __getitem__(self, index):            
        return (torch.from_numpy(self.data[index,:,:,:]).float(),
                torch.from_numpy(self.target[index,:,:,:]).float())

    def __len__(self):
        return self.data.shape[0]
