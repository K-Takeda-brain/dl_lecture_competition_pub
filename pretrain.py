#%%
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import ThingsMEGDataset, MyMNEinfo
from src.modified_models import BasicConvClassifier
from src.utils import set_seed
from src.losses import CLIPLoss

import mne
from mne.datasets import sample, spm_face, testing
from mne.io import (
    read_raw_artemis123,
    read_raw_bti,
    read_raw_ctf,
    read_raw_fif,
    read_raw_kit,
)

# CUDAの初期化確認
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# CUDAデバイスの初期化
torch.cuda.init()
#%%
@hydra.main(version_base=None, config_path="configs", config_name="config_pretrain")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_dir = os.path.join("outputs", "pretrain")
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    raw = read_raw_ctf(
        spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
    )
    info = raw.info
    recording = MyMNEinfo(info, ["MLF25",  "MRF43", "MRO13", "MRO11"])
    
    train_set = ThingsMEGDataset("train", args.data_dir, recording)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir, recording)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir, recording)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, in_channels={'meg': 270}, hidden={'meg': 320},
    ).to(args.device)
    
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # ------------------
    # Start pretraining
    # ------------------
    # pretraining using CLIP loass
    max_val_acc = 0
    for epoch in range(args.pretrain_epochs):
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}")
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        
        model.train()
        for X, _, subject_idxs, pos, image_feature in tqdm(train_loader, desc="Pretrain"):
            X, subject_idxs, pos, image_feature = X.to(args.device), subject_idxs.to(args.device), pos.to(args.device), image_feature.to(args.device)

            y_pred = model(X, subject_idxs, pos)
            probability, label = CLIPLoss().get_probabilities(y_pred, image_feature)
            loss = CLIPLoss()(y_pred, image_feature)
            train_loss.append(loss.item())
            
            # calculate top 1 accuracy
            acc = torch.sum(torch.argmax(probability, dim=1) == label) / len(label)
            train_acc.append(acc.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(np.mean(train_loss))
        
        model.eval()
        for X, _, subject_idxs, pos, image_feature in tqdm(val_loader, desc="Validation"):
            X, subject_idxs, pos, image_feature = X.to(args.device), subject_idxs.to(args.device), pos.to(args.device), image_feature.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs, pos)
                probability, label = CLIPLoss().get_probabilities(y_pred, image_feature)
                loss = CLIPLoss()(y_pred, image_feature)
                val_loss.append(loss.item())
                
                # calculate accuracy
                acc = torch.sum(torch.argmax(probability, dim=1) == label) / len(label)
                val_acc.append(acc.item())
            
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_pretrain_last.pt"))
        if args.use_wandb:
            wandb.log({"pretrain_loss": np.mean(train_loss)})
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_pretrain_best.pt"))
            torch.save(model.state_dict(), os.path.join(save_dir, "model_pretrain_best.pt"))
            max_val_acc = np.mean(val_acc)

if __name__ == "__main__":
    run()