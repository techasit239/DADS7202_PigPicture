from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from freshcheck.config import MODEL_NAMES
from freshcheck.data import (
    ClassificationDataset,
    PredictionDataset,
    build_eval_transform,
    build_train_transform,
    group_split_dataframe,
    iter_image_paths,
    load_dataframe,
    prepare_kaggle_dataframe,
    prepare_phase2_manifest,
)
from freshcheck.models import build_model
from freshcheck.trainer import evaluate, fit, predict, save_metrics_json, save_prediction_csv
from freshcheck.utils import choose_device, ensure_dir, json_dump, set_seed


def parse_models(raw_models: list[str]) -> list[str]:
    if raw_models == ["all"]:
        return MODEL_NAMES[:]
    invalid = sorted(set(raw_models) - set(MODEL_NAMES))
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}. Choose from {MODEL_NAMES} or all.")
    return raw_models


def make_loader(df: pd.DataFrame, img_size: int, batch_size: int, num_workers: int, train: bool):
    transform = build_train_transform(img_size) if train else build_eval_transform(img_size)
    dataset = ClassificationDataset(df, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def command_prepare_splits(args) -> None:
    set_seed(args.seed)
    df = prepare_kaggle_dataframe(args.data_dir)
    train_df, val_df = group_split_dataframe(df, train_ratio=args.train_ratio, seed=args.seed)
    out_dir = ensure_dir(args.output_dir)
    train_path = out_dir / "kaggle_train.csv"
    val_path = out_dir / "kaggle_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    summary = {
        "total_images": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_classes": train_df["class"].value_counts().to_dict(),
        "val_classes": val_df["class"].value_counts().to_dict(),
    }
    json_dump(summary, out_dir / "split_summary.json")
    print(f"Saved train split: {train_path}")
    print(f"Saved val split:   {val_path}")


def command_train(args) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    models = parse_models(args.models)
    train_df = load_dataframe(args.train_csv)
    val_df = load_dataframe(args.val_csv)

    train_loader = make_loader(train_df, args.img_size, args.batch_size, args.num_workers, train=True)
    val_loader = make_loader(val_df, args.img_size, args.batch_size, args.num_workers, train=False)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")

    for model_name in models:
        print(f"\n=== Training {model_name} on {device} ===")
        model = build_model(model_name, pretrained=args.pretrained, dropout=args.dropout).to(device)
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
        history, train_summary = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_path=checkpoint_path,
        )
        history_path = metrics_dir / f"{model_name}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        save_metrics_json(
            metrics_dir / f"{model_name}_train_summary.json",
            model_name,
            {"accuracy": 0.0, "macro_f1": train_summary["best_f1"], "loss": 0.0, "report": {}},
            extra={
                "best_epoch": train_summary["best_epoch"],
                "checkpoint_path": train_summary["checkpoint_path"],
                "history_csv": str(history_path.resolve()),
            },
        )
        print(f"Best checkpoint: {checkpoint_path}")


def command_evaluate(args) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    models = parse_models(args.models)
    eval_df = load_dataframe(args.csv)
    loader = make_loader(eval_df, args.img_size, args.batch_size, args.num_workers, train=False)
    output_dir = ensure_dir(args.output_dir)

    criterion = torch.nn.CrossEntropyLoss()
    for model_name in models:
        checkpoint = Path(args.checkpoint_dir) / f"{model_name}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")
        model = build_model(model_name, pretrained=False, dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        metrics = evaluate(model, loader, criterion, device, desc=f"eval:{model_name}")
        metrics_path = output_dir / f"{model_name}_eval.json"
        save_metrics_json(metrics_path, model_name, metrics, extra={"checkpoint": str(checkpoint.resolve())})
        print(
            f"{model_name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"loss={metrics['loss']:.4f}"
        )


def command_predict(args) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    models = parse_models(args.models)
    image_root = Path(args.input_path)
    if image_root.is_dir():
        image_paths = sorted(iter_image_paths(image_root))
    else:
        image_paths = [image_root]
    if not image_paths:
        raise ValueError(f"No input images found under {args.input_path}")

    loader = DataLoader(
        PredictionDataset(image_paths, build_eval_transform(args.img_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    output_dir = ensure_dir(args.output_dir)

    for model_name in models:
        checkpoint = Path(args.checkpoint_dir) / f"{model_name}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")
        model = build_model(model_name, pretrained=False, dropout=args.dropout).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        rows = predict(model, loader, device)
        csv_path = output_dir / f"{model_name}_predictions.csv"
        save_prediction_csv(csv_path, rows)
        print(f"Saved predictions for {model_name}: {csv_path}")


def command_prepare_cvat(args) -> None:
    df, gt_masks_dir = prepare_phase2_manifest(
        thai_data_dir=args.thai_data_dir,
        cvat_xml_path=args.cvat_xml_path,
        output_dir=args.output_dir,
    )
    out_dir = ensure_dir(args.output_dir)
    csv_path = out_dir / "thai_test_set.csv"
    df.to_csv(csv_path, index=False)
    json_dump(
        {
            "rows": len(df),
            "unique_pieces": int(df["piece_id"].nunique()),
            "class_distribution": df["class"].value_counts().to_dict(),
            "source_distribution": df["source"].value_counts().to_dict(),
            "gt_masks_dir": str(gt_masks_dir.resolve()),
        },
        out_dir / "phase2_foundation_summary.json",
    )
    print(f"Saved manifest: {csv_path}")
    print(f"Saved masks:    {gt_masks_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FreshCheck local runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_splits = subparsers.add_parser("prepare-splits", help="Create group-aware train/val CSV splits")
    prepare_splits.add_argument("--data-dir", required=True)
    prepare_splits.add_argument("--output-dir", required=True)
    prepare_splits.add_argument("--train-ratio", type=float, default=0.8)
    prepare_splits.add_argument("--seed", type=int, default=42)
    prepare_splits.set_defaults(func=command_prepare_splits)

    common_model_args = {
        "models": dict(nargs="+", default=["all"], help=f"Models to run: {MODEL_NAMES} or all"),
        "img_size": dict(type=int, default=224),
        "batch_size": dict(type=int, default=16),
        "num_workers": dict(type=int, default=0),
        "device": dict(default="auto"),
        "dropout": dict(type=float, default=0.3),
        "seed": dict(type=int, default=42),
    }

    train = subparsers.add_parser("train", help="Train one or more classifiers")
    train.add_argument("--train-csv", required=True)
    train.add_argument("--val-csv", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    for key, kwargs in common_model_args.items():
        train.add_argument(f"--{key.replace('_', '-')}", **kwargs)
    train.set_defaults(func=command_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate checkpoints on a labeled CSV")
    evaluate_parser.add_argument("--csv", required=True)
    evaluate_parser.add_argument("--checkpoint-dir", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    for key, kwargs in common_model_args.items():
        evaluate_parser.add_argument(f"--{key.replace('_', '-')}", **kwargs)
    evaluate_parser.set_defaults(func=command_evaluate)

    predict_parser = subparsers.add_parser("predict", help="Predict labels for one image or a folder")
    predict_parser.add_argument("--input-path", required=True)
    predict_parser.add_argument("--checkpoint-dir", required=True)
    predict_parser.add_argument("--output-dir", required=True)
    for key, kwargs in common_model_args.items():
        predict_parser.add_argument(f"--{key.replace('_', '-')}", **kwargs)
    predict_parser.set_defaults(func=command_predict)

    prepare_cvat = subparsers.add_parser("prepare-cvat", help="Convert CVAT XML to GT masks + manifest CSV")
    prepare_cvat.add_argument("--thai-data-dir", required=True)
    prepare_cvat.add_argument("--cvat-xml-path", required=True)
    prepare_cvat.add_argument("--output-dir", required=True)
    prepare_cvat.set_defaults(func=command_prepare_cvat)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
