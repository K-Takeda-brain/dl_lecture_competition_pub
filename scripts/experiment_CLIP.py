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
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import mne
from mne.datasets import sample, spm_face, testing
from mne.io import (
    read_raw_artemis123,
    read_raw_bti,
    read_raw_ctf,
    read_raw_fif,
    read_raw_kit,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import ThingsMEGDataset, SegmentBatch, MEGDataset, custom_collate
from src.common import MyMNEinfo

from src.simpleconv import SimpleConv
from src.losses import ClipLoss

#%%
### dataset
data_dir = "../data"
image_feature_dir = "../data/image/CLIPvision"
train_image_list_file = "../data/train_image_paths.txt"
val_image_list_file = "../data/val_image_paths.txt"

args = DictConfig({"data_dir": "../data", "batch_size": 32, "num_workers": 4})

loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
train_set = ThingsMEGDataset("train", args.data_dir)
val_set = ThingsMEGDataset("val", args.data_dir)
test_set = ThingsMEGDataset("test", args.data_dir)

train_meg_data = {
    'data': train_set.X,
    'subject_idxs': train_set.subject_idxs,
    'label': train_set.y
}

train_meg = MEGDataset(train_meg_data, image_feature_dir, train_image_list_file)
dataloader = DataLoader(train_meg, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
raw = read_raw_ctf(
    spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
)
info = raw.info
recording = MyMNEinfo(info, ["MLF25",  "MRF43", "MRO13", "MRO11"]) #MLF25,  MRF43, MRO13, MRO11

model = SimpleConv(
    in_channels={"meg": 271},
    out_channels=768,
    hidden={"meg": 512},
    depth=4,
    concatenate=False,
    kernel_size=5,
    batch_norm=True,
    dropout=0.1,
    spatial_attention=False,
    pos_dim=256
)


# 訓練関数の定義
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader):
            meg_batch = batch['meg'].to(device)
            image_feature_batch = batch['image_feature'].to(device)
            subject_index_batch = batch['subject_index'].to(device)

            # フォワードパス
            outputs = model({'meg': meg_batch}, {'subject_index': subject_index_batch})
            loss = criterion(outputs, image_feature_batch)

            # バックワードパスと最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Training finished.')

#%%

criterion = ClipLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer, num_epochs=1)

#%%
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model({'meg': batch['meg']}, batch)
            loss = criterion(outputs, batch['features'])
            total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')

# ダミーデータの準備
train_data = [{
    'meg': torch.randn(271, 181),  # (C, T)
    'features': torch.randn(768, 181),
    'features_mask': torch.ones(768, 181),
    'subject_index': torch.randint(0, 200, (1,)),
    'recording_index': torch.randint(0, 100, (1,)),
    'recording': MyMNEinfo(info, ["MLF25", "MRF43", "MRO13", "MRO11"])  # 仮のrecordingオブジェクト
} for _ in range(10)]

test_data = [{
    'meg': torch.randn(271, 181),  # (C, T)
    'features': torch.randn(768, 181),
    'features_mask': torch.ones(768, 181),
    'subject_index': torch.randint(0, 200, (1,)),
    'recording_index': torch.randint(0, 100, (1,)),
    'recording': MyMNEinfo(info, ["MLF25", "MRF43", "MRO13", "MRO11"])  # 仮のrecordingオブジェクト
} for _ in range(2)]


train_dataset = MEGDataset(meg_data, image_feature_dir, image_list_file)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# クリテリオンとオプティマイザの設定
criterion = ClipLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 訓練の実行
train(model, train_loader, criterion, optimizer, num_epochs=10)

# 推論の実行
evaluate(model, test_loader, criterion)

# %%
