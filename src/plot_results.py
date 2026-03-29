"""
plot_results.py
---------------
Reads the CSVs produced by train_model.py and evaluate_model.py, and writes:
    results/figures/training_curves.png   -- loss + accuracy over epochs
    results/figures/confusion_matrix.png  -- heatmap of val predictions
    results/figures/class_distribution.png -- train/val counts per class

Usage:
    python src/plot_results.py \
        --tables_dir  results/tables \
        --figures_dir results/figures
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
})


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_training_history(path):
    epochs, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    return epochs, train_acc, val_acc, train_loss, val_loss


def read_confusion_matrix(path):
    classes, matrix = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        classes = header[1:]           # skip "true\pred" column
        for row in reader:
            matrix.append([int(x) for x in row[1:]])
    return classes, np.array(matrix)


def read_dataset_summary(path):
    classes, train_counts, val_counts = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            classes.append(row["class"])
            train_counts.append(int(row["train_count"]))
            val_counts.append(int(row["val_count"]))
    return classes, train_counts, val_counts


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(epochs, train_acc, val_acc, train_loss, val_loss, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_acc,  label="Train", marker="o", markersize=3)
    ax1.plot(epochs, val_acc,    label="Val",   marker="o", markersize=3, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy over epochs")
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax1.legend()

    ax2.plot(epochs, train_loss, label="Train", marker="o", markersize=3)
    ax2.plot(epochs, val_loss,   label="Val",   marker="o", markersize=3, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.set_title("Loss over epochs")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_confusion_matrix(classes, matrix, out_path):
    n = len(classes)
    # Normalise rows to get recall per class
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    norm_matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised)")
    ax.grid(False)

    for i in range(n):
        for j in range(n):
            color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n({norm_matrix[i, j]:.0%})",
                    ha="center", va="center", fontsize=9, color=color)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_class_distribution(classes, train_counts, val_counts, out_path):
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, train_counts, width, label="Train")
    bars2 = ax.bar(x + width / 2, val_counts,   width, label="Val")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tables_dir",  default="results/tables")
    p.add_argument("--figures_dir", default="results/figures")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.figures_dir, exist_ok=True)

    # Training curves
    history_path = os.path.join(args.tables_dir, "training_history.csv")
    if os.path.exists(history_path):
        print("[plot_results] Plotting training curves ...")
        data = read_training_history(history_path)
        plot_training_curves(*data,
                             out_path=os.path.join(args.figures_dir, "training_curves.png"))
    else:
        print(f"  Skipping training curves (not found: {history_path})")

    # Confusion matrix
    cm_path = os.path.join(args.tables_dir, "confusion_matrix.csv")
    if os.path.exists(cm_path):
        print("[plot_results] Plotting confusion matrix ...")
        classes, matrix = read_confusion_matrix(cm_path)
        plot_confusion_matrix(classes, matrix,
                              out_path=os.path.join(args.figures_dir, "confusion_matrix.png"))
    else:
        print(f"  Skipping confusion matrix (not found: {cm_path})")

    # Class distribution
    summary_path = os.path.join(args.tables_dir, "dataset_summary.csv")
    if os.path.exists(summary_path):
        print("[plot_results] Plotting class distribution ...")
        classes, train_counts, val_counts = read_dataset_summary(summary_path)
        plot_class_distribution(classes, train_counts, val_counts,
                                out_path=os.path.join(args.figures_dir, "class_distribution.png"))
    else:
        print(f"  Skipping class distribution (not found: {summary_path})")

    print("[plot_results] Done.")


if __name__ == "__main__":
    main()
