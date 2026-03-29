"""
evaluate_model.py
-----------------
Loads the saved model and val split, runs inference, and writes:
    results/tables/model_performance.csv   -- accuracy, precision, recall, F1 per class
    results/tables/confusion_matrix.csv    -- raw confusion matrix counts

Usage:
    python src/evaluate_model.py \
        --split_csv   results/tables/image_split.csv \
        --model_path  results/models/model.pth \
        --output_dir  results/tables
"""

import argparse
import csv
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Dataset (val-only, no augmentation)
# ---------------------------------------------------------------------------

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class EyeValDataset(Dataset):
    def __init__(self, rows, class_to_idx):
        self.rows = rows
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        filepath, class_name = self.rows[idx]
        image = Image.open(filepath).convert("RGB")
        return VAL_TRANSFORM(image), self.class_to_idx[class_name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_split_csv(csv_path):
    rows = defaultdict(list)
    classes = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["split"]].append((row["filepath"], row["class"]))
            classes.add(row["class"])
    return rows, sorted(classes)


def compute_metrics(y_true, y_pred, classes):
    """
    Compute per-class precision, recall, F1 and overall accuracy.
    Returns (metrics_list, confusion_matrix_dict).
    """
    n = len(classes)
    # confusion matrix as dict[true][pred] -> count
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    metrics = []
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = sum(cm[i])

        metrics.append({
            "class":     cls,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        })

    accuracy = sum(cm[i][i] for i in range(n)) / max(len(y_true), 1)
    return metrics, cm, round(accuracy, 4)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split_csv",   default="results/tables/image_split.csv")
    p.add_argument("--model_path",  default="results/models/model.pth")
    p.add_argument("--output_dir",  default="results/tables")
    p.add_argument("--batch_size",  type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"[evaluate_model] Loading split from {args.split_csv} ...")
    split_rows, classes = read_split_csv(args.split_csv)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    print(f"  Classes: {classes}")

    val_dataset = EyeValDataset(split_rows["val"], class_to_idx)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    print(f"  Val samples: {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[evaluate_model] Loading model from {args.model_path} ...")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()

    # --- run inference ---
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    # --- compute metrics ---
    metrics, cm, accuracy = compute_metrics(y_true, y_pred, classes)
    print(f"  Overall val accuracy: {accuracy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- write per-class metrics ---
    perf_path = os.path.join(args.output_dir, "model_performance.csv")
    with open(perf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall",
                                                "f1", "support"])
        writer.writeheader()
        writer.writerows(metrics)
        # overall accuracy as a summary row
        f.write(f"\noverall_accuracy,{accuracy},,,\n")
    print(f"  Wrote metrics -> {perf_path}")

    # --- write confusion matrix ---
    cm_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ pred"] + classes)
        for i, cls in enumerate(classes):
            writer.writerow([cls] + cm[i])
    print(f"  Wrote confusion matrix -> {cm_path}")
    print("[evaluate_model] Done.")


if __name__ == "__main__":
    main()
