import torch
import numpy as np
from transformers import ASTFeatureExtractor
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data.dataset import BreathDataset
from src.models.base_ast import ASTClassifier
from src.utils.metrics import compute_icbhi_metrics


def evaluate(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = np.load(args.data_path)
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    test_loader = DataLoader(
        BreathDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size, shuffle=False
    )

    # Load model
    model = ASTClassifier(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    se, sp, score, cm = compute_icbhi_metrics(all_labels, all_preds)
    print(f"\n{'='*40}")
    print(f"  Sensitivity (Se): {se:.4f}")
    print(f"  Specificity (Sp): {sp:.4f}")
    print(f"  ICBHI Score:      {score:.4f}")
    print(f"{'='*40}\n")

    # Plot confusion matrix
    labels_list = ['Normal', 'Crackle', 'Wheeze', 'Both']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Score={score:.4f})')
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Saved confusion matrix to {args.output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/processed/icbhi_ast_16k_8s.npz")
    parser.add_argument("--model_path", type=str, default="./results/baseline/checkpoints/best_model.pth")
    parser.add_argument("--output_plot", type=str, default="./results/baseline/confusion_matrix.png")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args)
