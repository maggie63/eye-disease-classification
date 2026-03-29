"""
load_data.py
------------
Reads images from data/raw/ (one subfolder per class), performs a stratified
80/20 train/val split, and writes:
    results/tables/image_split.csv      -- filepath, class, split for every image
    results/tables/dataset_summary.csv  -- per-class train/val counts

Usage:
    python src/load_data.py \
        --data_dir   data/raw \
        --output_dir results/tables \
        --val_split  0.2 \
        --seed       42
"""

import argparse
import csv
import os
import random
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/raw")
    p.add_argument("--output_dir", default="results/tables")
    p.add_argument("--val_split",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def collect_images(data_dir):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    class_images = defaultdict(list)
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if os.path.splitext(fname)[1].lower() in supported:
                class_images[class_name].append(os.path.join(class_path, fname))
    return class_images


def stratified_split(class_images, val_split, seed):
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for class_name, paths in class_images.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        for p in shuffled[:n_val]:
            val_rows.append((p, class_name, "val"))
        for p in shuffled[n_val:]:
            train_rows.append((p, class_name, "train"))
    return train_rows, val_rows


def write_csv(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "split"])
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {filepath}")


def write_summary(train_rows, val_rows, output_dir):
    counts = defaultdict(lambda: {"train": 0, "val": 0})
    for _, cls, _ in train_rows:
        counts[cls]["train"] += 1
    for _, cls, _ in val_rows:
        counts[cls]["val"] += 1

    path = os.path.join(output_dir, "dataset_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "train_count", "val_count", "total"])
        for cls in sorted(counts):
            t, v = counts[cls]["train"], counts[cls]["val"]
            writer.writerow([cls, t, v, t + v])
    print(f"  Wrote dataset summary -> {path}")


def main():
    args = parse_args()
    print(f"[load_data] Scanning {args.data_dir} ...")
    class_images = collect_images(args.data_dir)

    if not class_images:
        raise FileNotFoundError(f"No image subfolders found in {args.data_dir}")

    for cls, paths in class_images.items():
        print(f"  {cls}: {len(paths)} images")

    train_rows, val_rows = stratified_split(class_images, args.val_split, args.seed)
    print(f"  Split -> {len(train_rows)} train / {len(val_rows)} val")

    write_csv(train_rows + val_rows, os.path.join(args.output_dir, "image_split.csv"))
    write_summary(train_rows, val_rows, args.output_dir)
    print("[load_data] Done.")


if __name__ == "__main__":
    main()
