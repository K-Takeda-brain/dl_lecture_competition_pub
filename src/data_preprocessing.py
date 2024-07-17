#%%
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

# サンプリングレートの設定
sampling_rate_original = 200
sampling_rate_new = 120
downsample_factor = sampling_rate_original // sampling_rate_new

# ベースライン補正の設定
baseline_start = 0  # ベースライン開始インデックス
baseline_end = 281  # ベースライン終了インデックス（新しいサンプリングレートでのインデックス）

# ダウンサンプリング
def downsample(data, factor):
    return data[:, :, ::factor]

# ベースライン補正
def baseline_correction(data, baseline_start, baseline_end):
    baseline = data[:, :, baseline_start:baseline_end].mean(axis=-1, keepdims=True)
    return data - baseline

# ロバストスケーリングとクリッピング
def robust_scaling_clipping(data, clip_range=(-20, 20)):
    scaler = RobustScaler()
    num_samples, num_channels, num_timepoints = data.shape
    data_reshaped = data.reshape(-1, num_timepoints)
    data_scaled = scaler.fit_transform(data_reshaped)
    data_scaled = np.clip(data_scaled, clip_range[0], clip_range[1])
    return data_scaled.reshape(num_samples, num_channels, num_timepoints)

# 前処理関数
def preprocess_meg(data, downsample_factor, baseline_start, baseline_end, device):
    # GPUにデータを移動
    data = torch.tensor(data, device=device, dtype=torch.float32)

    # ダウンサンプリング
    data = data[:, :, ::downsample_factor]

    # ベースライン補正
    baseline = data[:, :, baseline_start:baseline_end].mean(dim=-1, keepdim=True)
    data = data - baseline

    # CPUにデータを移動してロバストスケーリングとクリッピング
    data = data.cpu().numpy()
    data = robust_scaling_clipping(data)

    # GPUにデータを戻す
    data = torch.tensor(data, device=device, dtype=torch.float32)

    return data

#%%
if __name__ == '__main__':
    
    for split in ['val', 'test']: #'train', 
        data = torch.load(f'../data/{split}_X.pt')
        # ダウンサンプリング、ベースライン補正、スケーリング、クリッピング
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        processed_data = preprocess_meg(data, downsample_factor, baseline_start, baseline_end, device)

        # 前処理後のデータを保存
        processed_data = processed_data.cpu()
        torch.save(processed_data, f'../data/{split}_X_preprocessed.pt', pickle_protocol=4)

    #for split in ['train', 'val', 'test']:
    #    data = torch.load(f'../data/{split}_X_preprocessed.pt')
    #    data = data.to('cpu')
    #    torch.save(data, f'../data/{split}_X_preprocessed.pt', pickle_protocol=4)
        
# %%
