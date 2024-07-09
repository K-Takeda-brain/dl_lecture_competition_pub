#%%
import os, sys
import torch
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torchaudio
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import ThingsMEGDataset

from src.simpleconv import SimpleConv
from src.losses import ClipLoss

#%%
### dataset

args = DictConfig({"data_dir": "../data", "batch_size": 32, "num_workers": 4})

loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
train_set = ThingsMEGDataset("train", args.data_dir)
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
val_set = ThingsMEGDataset("val", args.data_dir)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
test_set = ThingsMEGDataset("test", args.data_dir)
test_loader = torch.utils.data.DataLoader(
    test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
)
# laod subject data
sub_idx = torch.load("../data/train_subject_idxs.pt")

#%%
# set up the brain model
model = SimpleConv(
    in_channels=train_set.num_channels,
    out_channels=train_set.num_classes,
    seq_len=train_set.seq_len,
)

#%%
train_y = torch.load("../data/train_y.pt")
# %%
# check the dataset
print(train_set.X.shape) # (n_samples, n_channels, n_timepoints)
# %%
