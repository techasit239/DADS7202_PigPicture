"""Training, evaluation, and prediction helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import CLASS_NAMES, INDEX_TO_CLASS
from .utils import json_dump


def make_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    class_counts = df["class"].value_counts()
    total = class_counts.sum()
    weights = torch.tensor(
        [total / (len(CLASS_NAMES) * class_counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    return weights


def build_training_components(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
):
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_df, device))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return criterion, optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += batch_size
        progress.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct / max(total, 1):.3f}")
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(1)

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
        "preds": all_preds,
        "labels": all_labels,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "report": report,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: str | Path,
):
    criterion, optimizer, scheduler = build_training_components(
        model, train_df=train_df, device=device, lr=lr, weight_decay=weight_decay, epochs=epochs
    )
    history = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )
        val_metrics = evaluate(model, val_loader, criterion, device, desc="val")
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return history, {"best_f1": best_f1, "best_epoch": best_epoch, "checkpoint_path": str(checkpoint_path)}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for images, image_paths in tqdm(loader, desc="predict", leave=False):
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for path, pred_idx, prob_row in zip(image_paths, preds, probs):
            record = {
                "path": path,
                "pred_class": INDEX_TO_CLASS[int(pred_idx)],
                "confidence": float(prob_row[pred_idx]),
            }
            for index, class_name in enumerate(CLASS_NAMES):
                record[f"prob_{class_name}"] = float(prob_row[index])
            rows.append(record)
    return rows


def save_metrics_json(path: str | Path, model_name: str, metrics: dict, extra: dict | None = None) -> None:
    payload = {
        "model": model_name,
        "results": {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "loss": float(metrics["loss"]),
        },
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": metrics["confusion_matrix"],
        },
        "classification_report": metrics["report"],
    }
    if extra:
        payload["extra"] = extra
    json_dump(payload, path)


def save_confusion_matrix_artifacts(output_dir: str | Path, model_name: str, metrics: dict) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    csv_path = out_dir / f"{model_name}_confusion_matrix.csv"
    png_path = out_dir / f"{model_name}_confusion_matrix.png"

    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {model_name}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {"csv": str(csv_path.resolve()), "png": str(png_path.resolve())}


def save_prediction_csv(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_prediction_rows(rows: list[dict]) -> dict[str, float]:
    pred_classes = [row["pred_class"] for row in rows]
    counts = pd.Series(pred_classes).value_counts()
    return {key: float(value) for key, value in counts.to_dict().items()}
