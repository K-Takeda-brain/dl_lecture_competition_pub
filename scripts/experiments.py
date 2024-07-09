#%%
import os, sys
import torch
from efficientnet_pytorch import EfficientNet
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import ThingsMEGDataset

# 事前学習済みモデルの読み込み
model = EfficientNet.from_pretrained('efficientnet-b0')

num_classes = 1854  # 分類したいクラス数
in_features = model._fc.in_features
model._fc = torch.nn.Linear(in_features, num_classes)


### dataset

args = DictConfig({"data_dir": "../data", "batch_size": 32, "num_workers": 4})

loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
train_set = ThingsMEGDataset("train", args.data_dir)
#train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
val_set = ThingsMEGDataset("val", args.data_dir)
#val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
test_set = ThingsMEGDataset("test", args.data_dir)
#test_loader = torch.utils.data.DataLoader(
#    test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
#)

#%%
# check the dataset
print(train_set.X.shape) # (n_samples, n_channels, n_timepoints)

#%%
# laod subject data
sub_idx = torch.load("../data/train_subject_idxs.pt")
#%%
# convert MEG to the spectrogram
n_channels = 3
n_samples = 1000
sampling_rate = 100  # サンプリング周波数（Hz）

# スペクトログラムの設定
n_fft = 256  # FFTのサイズ
win_length = None  # 窓のサイズ、Noneはn_fftと同じ
hop_length = None  # ストライド、Noneはwin_length/2と同じ

time_series_data = train_set.X[0, :n_channels, :n_samples]
for i in range(n_channels):
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2.0)
    Sxx = spectrogram_transform(time_series_data[i])
    Sxx_log = 10 * torch.log10(Sxx + 1e-10)  # デシベルスケールに変換
