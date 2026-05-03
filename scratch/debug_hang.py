import sys
sys.path.append('d:/Deep-Learning-Project')
print("starting debug", flush=True)

import torch
print("imported torch", flush=True)

from src.evaluate import evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/processed/icbhi_ast_16k_8s.npz")
parser.add_argument("--model_path", type=str, default="./results/best_model.pth")
parser.add_argument("--output_plot", type=str, default="./results/exp_p1_wavelets/confusion_matrix.png")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--model", type=str, default="ssast", choices=["ast", "ssast"])
parser.add_argument("--pretrained_path", type=str, default="./pretrained/SSAST-Base-Patch-400.pth")
args = parser.parse_args([])

print("calling evaluate", flush=True)
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE, flush=True)

data = np.load(args.data_path)
X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']
print("Loaded data", flush=True)

from src.data.dataset import SSASTBreathDataset
from src.models.ssast import SSASTModel
test_ds = SSASTBreathDataset(X_test, y_test, d_test, train=False)
print("init model", flush=True)
model = SSASTModel(
    label_dim=4, fshape=16, tshape=16, fstride=10, tstride=10,
    input_fdim=128, input_tdim=1024, model_size='base',
    load_pretrained_mdl_path=args.pretrained_path
).to(DEVICE)
print("model to device done", flush=True)

from torch.utils.data import DataLoader
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
print("test loader done", flush=True)

print("loading weights", flush=True)
state_dict = torch.load(args.model_path, map_location=DEVICE)
print("weights loaded from disk", flush=True)
model.load_state_dict(state_dict)
model.eval()
print("model loaded dict", flush=True)

all_preds, all_labels = [], []
with torch.no_grad():
    print("starting inference", flush=True)
    for inputs, labels, _ in test_loader:
        print("batch loaded", flush=True)
        inputs = inputs.to(DEVICE)
        print("inputs to device", flush=True)
        logits = model(inputs)
        print("forward done", flush=True)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        break
print("debug script finished", flush=True)
