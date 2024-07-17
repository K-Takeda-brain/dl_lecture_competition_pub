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
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
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
    
    if args.pretrain:
        weights = torch.load(os.path.join("outputs", "pretrain", "model_pretrain_best.pt"))
        # to load the model trained with DataParallel to model without DataParallel
        weights = {k.replace("module.", ""): v for k, v in weights.items()} 
        model.load_state_dict(weights) #, map_location=args.device
    
    model.classify = True
    
    ## モデル全体のパラメータの勾配計算を無効にする
    #for param in model.parameters():
    #    param.requires_grad = False
    #
    ## 分類ヘッドのパラメータの勾配計算を有効にする
    #for param in model.classifier.parameters():
    #    param.requires_grad = True

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
        
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    # ------------------
    #   Start fine-tuning
    # ------------------  
    # load pretrained model
    
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs, pos, _ in tqdm(train_loader, desc="Train"):
            X, y, subject_idxs, pos = X.to(args.device), y.to(args.device), subject_idxs.to(args.device), pos.to(args.device)

            y_pred = model(X, subject_idxs, pos)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
        #scheduler.step(np.mean(train_loss))
        
        model.eval()
        for X, y, subject_idxs, pos, _ in tqdm(val_loader, desc="Validation"):
            X, y, subject_idxs, pos = X.to(args.device), y.to(args.device), subject_idxs.to(args.device), pos.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_idxs, pos)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())
        #scheduler.step(np.mean(val_loss))

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    #weights = torch.load(os.path.join(logdir, "model_best.pt"))
    ## to load the model trained with DataParallel to model without DataParallel
    #weights = {k.replace("module.", ""): v for k, v in weights.items()}
    #model.load_state_dict(weights)
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs, pos in tqdm(test_loader, desc="Validation"):     
        X, subject_idxs, pos = X.to(args.device), subject_idxs.to(args.device), pos.to(args.device)
        y_pred = model(X, subject_idxs, pos)  
        preds.append(y_pred.detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
