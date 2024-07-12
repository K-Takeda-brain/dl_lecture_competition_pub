#%%
import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import dataclasses
import typing as tp
from torch.utils.data import DataLoader, Dataset

import mne
from mne.datasets import sample, spm_face, testing
from mne.io import (
    read_raw_artemis123,
    read_raw_bti,
    read_raw_ctf,
    read_raw_fif,
    read_raw_kit,
)
from copy import deepcopy

def custom_collate(batch):
    batch_dict = {}
    for key in batch[0]:
        if key == 'recording':
            batch_dict[key] = [item[key] for item in batch]  # list of MyMNEinfo objects
        else:
            batch_dict[key] = torch.stack([item[key] for item in batch], dim=0)
    return batch_dict

class MEGDataset(Dataset):
    def __init__(self, meg_data, image_feature_dir, image_list_file):
        self.meg_data = meg_data['data']
        self.labels = meg_data.get('labels')
        self.subject_idxs = meg_data['subject_idxs']
        self.image_feature_dir = image_feature_dir
        self.recording = meg_data['recording']

        # Read the image list file
        with open(image_list_file, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.meg_data)

    def __getitem__(self, idx):
        meg = self.meg_data[idx]
        subject_idx = self.subject_idxs[idx]

        # Extract the corresponding image feature
        image_name = self.image_list[idx]
        image_feature_path = os.path.join(self.image_feature_dir, image_name.replace('.jpg', '.npy'))
        image_feature = np.load(image_feature_path)

        sample = {
            'meg': torch.tensor(meg, dtype=torch.float32),
            'image_feature': torch.tensor(image_feature, dtype=torch.float32),
            'subject_index': subject_idx
        }

        if self.labels is not None:
            sample['label'] = self.labels[idx]

        return sample

        
@dataclasses.dataclass
class SegmentBatch:
    meg: torch.Tensor
    features: torch.Tensor
    features_mask: torch.Tensor
    subject_index: torch.Tensor
    recording_index: torch.Tensor
    positions: torch.Tensor

    def to(self, device: tp.Any) -> "SegmentBatch":
        """Creates a new instance on the appropriate device."""
        out: tp.Dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if isinstance(data, torch.Tensor):
                out[field.name] = data.to(device)
            else:
                out[field.name] = data
        return SegmentBatch(**out)

    def replace(self, **kwargs) -> "SegmentBatch":
        cls = self.__class__
        kw = {}
        for field in dataclasses.fields(cls):
            if field.name in kwargs:
                kw[field.name] = kwargs[field.name]
            else:
                kw[field.name] = getattr(self, field.name)
        return cls(**kw)

    def __getitem__(self, index) -> "SegmentBatch":
        cls = self.__class__
        kw = {}
        indexes = torch.arange(
            len(self), device=self.meg.device)[index].tolist()  # explicit indexes for lists
        for field in dataclasses.fields(cls):
            data = getattr(self, field.name)
            if isinstance(data, list):
                if data:
                    value = [data[idx] for idx in indexes]
                else:
                    value = []
            else:
                value = data[index]
            kw[field.name] = value
        return cls(**kw)

    def __len__(self) -> int:
        return len(self.meg)

    @classmethod
    def collate_fn(cls, meg_features_list: tp.List["SegmentBatch"]) -> "SegmentBatch":
        out: tp.Dict[str, torch.Tensor] = {}
        for field in dataclasses.fields(cls):
            data = [getattr(mf, field.name) for mf in meg_features_list]
            if isinstance(data[0], torch.Tensor):
                out[field.name] = torch.cat([d.unsqueeze(0) for d in data], dim=0)
            else:
                out[field.name] = [x for y in data for x in y]
        meg_features = SegmentBatch(**out)
        # check that list sizes are either 0 or batch size
        batch_size = meg_features.meg.shape[0]
        for field in dataclasses.fields(meg_features):
            val = out[field.name]
            if isinstance(val, list):
                assert len(val) in (0, batch_size), f"Incorrect size for {field.name}"
        return meg_features



class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", recording = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X_preprocessed.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            self.image_features = torch.load(os.path.join(data_dir, f"{split}_X_image.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        self.channel_pos = recording.get_channel_positions()
        self.channel_pos = torch.tensor(self.channel_pos, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i], self.channel_pos, self.image_features[i]
        else:
            return self.X[i], self.subject_idxs[i], self.channel_pos
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

class MyMNEinfo:
    def __init__(self, info, channels_to_remove):
        self.layout = mne.channels.find_layout(info)
        self.channels_to_remove = channels_to_remove
        self.modified_layout = self._create_modified_layout()
        self.ch_names = self.modified_layout.names

    def _create_modified_layout(self):
        selected_names = [name for name in self.layout.names if name not in self.channels_to_remove]
        new_layout = deepcopy(self.layout)
        new_layout.names = selected_names
        new_layout.pos = [self.layout.pos[idx] for idx, name in enumerate(self.layout.names) if name not in self.channels_to_remove]
        new_layout.ids = [self.layout.ids[idx] for idx, name in enumerate(self.layout.names) if name not in self.channels_to_remove]
        return new_layout

    def plot_layout(self):
        self.modified_layout.plot()

    def get_layout_info(self):
        return self.modified_layout
    
    def get_channel_positions(self):
        pos = np.array(self.modified_layout.pos)
        pos = pos[:, :2]
        return pos

#%%
if __name__ == "__main__":
    raw = read_raw_ctf(
        spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
    )
    info = raw.info
    recording = MyMNEinfo(info, ["MLF25",  "MRF43", "MRO13", "MRO11"]) #MLF25,  MRF43, MRO13, MRO11
    
    pos = recording.get_channel_positions()
# %%
