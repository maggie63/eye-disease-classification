"""
train_model.py
--------------
Fine-tunes ResNet-50 on the 4-class eye disease dataset.
Reads image paths from results/tables/image_split.csv (written by load_data.py).

Outputs:
    results/models/model.pth              -- best checkpoint by val accuracy
    results/tables/training_history.csv  -- per-epoch loss and accuracy

Usage:
    python src/train_model.py \
        --split_csv   results/tables/image_split.csv \
        --output_dir  results \
        --num_epochs  25 \
        --batch_size  4 \
        --lr          0.001 \
        --seed        42
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from tempfile import TemporaryDirectory

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EyeDataset(Dataset):
    TRANSFORMS = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    def __init__(self, rows, class_to_idx, split):
        self.rows = rows                    # list of (filepath, class_name)
        self.class_to_idx = class_to_idx
        self.transform = self.TRANSFORMS[split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        filepath, class_name = self.rows[idx]
        image = Image.open(filepath).convert("RGB")
        return self.transform(image), self.class_to_idx[class_name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_split_csv(csv_path):
    """Return {split: [(filepath, class_name), ...]} and sorted class list."""
    rows = defaultdict(list)
    classes = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["split"]].append((row["filepath"], row["class"]))
            classes.add(row["class"])
    return rows, sorted(classes)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split_csv",   default="results/tables/image_split.csv")
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--num_epochs",  type=int,   default=25)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, dataloaders, dataset_sizes, criterion, optimizer,
          scheduler, num_epochs, device):
    """Train model, return (best_model_state_dict, history list of dicts)."""
    history = []

    with TemporaryDirectory() as tmpdir:
        best_path = os.path.join(tmpdir, "best.pt")
        torch.save(model.state_dict(), best_path)
        best_acc = 0.0
        since = time.time()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}  " + "-" * 20)
            epoch_row = {"epoch": epoch + 1}

            for phase in ["train", "val"]:
                model.train() if phase == "train" else model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc  = (running_corrects.double() / dataset_sizes[phase]).item()

                epoch_row[f"{phase}_loss"] = round(epoch_loss, 4)
                epoch_row[f"{phase}_acc"]  = round(epoch_acc,  4)
                print(f"  {phase:5s}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_path)

            history.append(epoch_row)

        elapsed = time.time() - since
        print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Best val accuracy: {best_acc:.4f}")

        model.load_state_dict(torch.load(best_path))

    return model, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # --- read split CSV ---
    print(f"[train_model] Reading {args.split_csv} ...")
    split_rows, classes = read_split_csv(args.split_csv)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    num_classes = len(classes)
    print(f"  Classes ({num_classes}): {classes}")
    print(f"  Class -> index: {class_to_idx}")

    # --- build datasets & loaders ---
    datasets_ = {
        split: EyeDataset(split_rows[split], class_to_idx, split)
        for split in ["train", "val"]
    }
    dataloaders = {
        split: DataLoader(datasets_[split], batch_size=args.batch_size,
                          shuffle=(split == "train"), num_workers=4)
        for split in ["train", "val"]
    }
    dataset_sizes = {split: len(datasets_[split]) for split in ["train", "val"]}
    print(f"  Dataset sizes: {dataset_sizes}")

    # --- device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  No GPU found — training on CPU.")

    # --- model ---
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 4-class output
    model = model.to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler_ = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- train ---
    model, history = train(
        model, dataloaders, dataset_sizes,
        criterion, optimizer, scheduler_,
        args.num_epochs, device,
    )

    # --- save model ---
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model, model_path)
    print(f"  Saved model -> {model_path}")

    # --- save training history ---
    tables_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    history_path = os.path.join(tables_dir, "training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                                "val_loss",   "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    print(f"  Saved training history -> {history_path}")
    print("[train_model] Done.")


if __name__ == "__main__":
    main()
