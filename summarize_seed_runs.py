from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"]


def load_eval_rows(root_dir: Path, model_name: str) -> list[dict]:
    rows: list[dict] = []
    for seed_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        eval_path = seed_dir / f"{model_name}_eval.json"
        if not eval_path.exists():
            continue
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        results = payload.get("results", {})
        row = {"seed_run": seed_dir.name, "eval_json": str(eval_path.resolve())}
        for key in METRIC_KEYS:
            row[key] = float(results.get(key, 0.0))
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation JSON files were found for the requested model.")

    df = pd.DataFrame(rows)
    summary = {
        "n_runs": int(len(df)),
        "runs": df.to_dict(orient="records"),
        "mean": {key: float(df[key].mean()) for key in METRIC_KEYS},
        "std": {key: float(df[key].std(ddof=1)) if len(df) > 1 else 0.0 for key in METRIC_KEYS},
        "min": {key: float(df[key].min()) for key in METRIC_KEYS},
        "max": {key: float(df[key].max()) for key in METRIC_KEYS},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated seed runs from FreshCheck evaluation JSON files."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Directory containing per-seed subdirectories, e.g. seed_42, seed_52, ...",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to summarize, e.g. efficientnet_b0 or swin_t",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save summary CSV and JSON files.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(root_dir, args.model)
    summary = summarize_rows(rows)

    runs_df = pd.DataFrame(summary["runs"])
    runs_csv = output_dir / f"{args.model}_seed_runs.csv"
    runs_df.to_csv(runs_csv, index=False)

    summary_json = output_dir / f"{args.model}_seed_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved run table: {runs_csv}")
    print(f"Saved summary:   {summary_json}")
    print()
    print(f"Model: {args.model}")
    print(f"Runs:  {summary['n_runs']}")
    for key in METRIC_KEYS:
        print(
            f"{key}: mean={summary['mean'][key]:.4f} "
            f"std={summary['std'][key]:.4f} "
            f"min={summary['min'][key]:.4f} "
            f"max={summary['max'][key]:.4f}"
        )


if __name__ == "__main__":
    main()
