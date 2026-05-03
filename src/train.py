import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse

from src.data.dataset import BreathDataset, SSASTBreathDataset
from src.models.base_ast import ASTClassifier
from src.models.ssast import SSASTModel
from src.optim.sam import SAM
from src.utils.seed import set_seed
from src.utils.metrics import compute_icbhi_metrics

def train(args):
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # Load preprocessed data
    data = np.load(args.data_path)
    X_train, y_train, d_train = data['X_train'], data['y_train'], data['device_train']
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    # --- Model Selection ---
    if args.model == 'ssast':
        # SSAST: uses raw log-mel spectrograms
        train_ds = SSASTBreathDataset(X_train, y_train, d_train, train=True)
        test_ds = SSASTBreathDataset(X_test, y_test, d_test, train=False)
        model = SSASTModel(
            label_dim=4, fshape=16, tshape=16, fstride=10, tstride=10,
            input_fdim=128, input_tdim=1024, model_size='base',
            load_pretrained_mdl_path=args.pretrained_path
        ).to(DEVICE)
    else:
        # Baseline AST: uses HuggingFace feature extractor
        processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        train_ds = BreathDataset(X_train, y_train, d_train, processor, train=True)
        test_ds = BreathDataset(X_test, y_test, d_test, processor, train=False)
        model = ASTClassifier(num_classes=4).to(DEVICE)


    # Weighted Random Sampler (handles class imbalance)
    counts = np.bincount(y_train)
    weights = [1.0 / counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Optimizer + Loss
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = args.start_epoch
    best_score = args.best_score

    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint and args.start_epoch == 0:
                start_epoch = checkpoint['epoch']
            if 'best_score' in checkpoint:
                best_score = checkpoint['best_score']
            print(f"Resumed from epoch {start_epoch}")
        else:
            # It's just a raw state dict
            model.load_state_dict(checkpoint)

    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        print(f"\nStarting Epoch {epoch+1}...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for i, (inputs, labels, _) in enumerate(progress_bar):
            if i == 0: print("  -> Batch 0 loaded. Moving to device...")
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if i == 0: print("  -> Running first forward pass...")
            # SAM Step 1: forward + backward at current weights
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            if i == 0: print("  -> Running first backward pass...")
            loss.backward()
            
            if i == 0: print("  -> Running SAM first_step...")
            optimizer.first_step(zero_grad=True)

            if i == 0: print("  -> Running second forward/backward pass...")
            # SAM Step 2: forward + backward at perturbed weights
            criterion(model(inputs), labels).backward()
            
            if i == 0: print("  -> Running SAM second_step...")
            optimizer.second_step(zero_grad=True)
            
            if i == 0: print("  -> Batch 0 completed successfully!")

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Evaluate after each epoch
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        se, sp, score, cm = compute_icbhi_metrics(all_labels, all_preds)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Se={se:.4f} | Sp={sp:.4f} | Score={score:.4f}")

        # Save checkpoint for every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': max(score, best_score)
        }
        epoch_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, epoch_path)
        print(f"    --> Saved checkpoint for epoch {epoch+1}")

        # Save best model
        if score > best_score:
            best_score = score
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"    --> Saved best model (Score={best_score:.4f})")

    print(f"\nBest Score: {best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/processed/icbhi_ast_16k_8s.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="./results/baseline/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default="ast", choices=["ast", "ssast"])
    parser.add_argument("--pretrained_path", type=str, default="./pretrained/SSAST-Base-Patch-400.pth")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from")
    parser.add_argument("--best_score", type=float, default=0.0, help="Initial best score to beat when resuming from raw weights")
    args = parser.parse_args()
    train(args)
