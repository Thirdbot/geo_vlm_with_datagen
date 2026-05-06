import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_history(path):
    path = Path(path)
    if not path.exists():
        candidates = sorted(Path(".").glob("checkpoints/**/history.csv"))
        message = f"History file not found: {path}"
        if candidates:
            message += "\nAvailable history files:\n"
            message += "\n".join(f"- {candidate}" for candidate in candidates)
        raise SystemExit(message)

    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "step": int(row["step"]),
                    "epoch": int(row["epoch"]),
                    "split": row.get("split", "train"),
                    "task": row["task"],
                    "loss": float(row["loss"]),
                    "lr": float(row["lr"]),
                    "bce_loss": _optional_float(row.get("bce_loss")),
                    "dice_loss": _optional_float(row.get("dice_loss")),
                    "dice_score": _optional_float(row.get("dice_score")),
                    "positive_ratio": _optional_float(row.get("positive_ratio")),
                    "predicted_positive_ratio": _optional_float(row.get("predicted_positive_ratio")),
                }
            )
    return rows


def _optional_float(value):
    if value in {None, ""}:
        return None
    return float(value)


def moving_average(values, window):
    if window <= 1:
        return values

    averaged = []
    running = []
    for value in values:
        running.append(value)
        if len(running) > window:
            running.pop(0)
        averaged.append(sum(running) / len(running))
    return averaged


def plot_points(rows, metric, x_axis, smooth, aggregate_epoch=False):
    if x_axis == "step" and not aggregate_epoch:
        x_values = [row["step"] for row in rows]
        y_values = [row[metric] for row in rows]
        return x_values, moving_average(y_values, smooth)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["epoch"]].append(row[metric])

    x_values = sorted(grouped)
    y_values = [sum(grouped[epoch]) / len(grouped[epoch]) for epoch in x_values]
    return x_values, moving_average(y_values, smooth)


def plot_history(args):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with: "
            ".venv/bin/python -m pip install matplotlib"
        ) from exc

    rows = read_history(args.history)
    if not rows:
        raise SystemExit(f"No rows found in {args.history}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_task = defaultdict(list)
    for row in rows:
        rows_by_task[row["task"]].append(row)

    plt.figure(figsize=(10, 6))
    for (task, split), task_rows in sorted(_group_rows(rows, "task", "split").items()):
        x_values, losses = plot_points(
            task_rows,
            "loss",
            args.x_axis,
            args.smooth,
            aggregate_epoch=(split != "train"),
        )
        plt.plot(x_values, losses, label=f"{task}/{split}")

    plt.xlabel(args.x_axis.title())
    plt.ylabel("Loss")
    plt.title("Training Loss By Task")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / "loss_by_task.png"
    plt.savefig(loss_path, dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    train_rows = [row for row in rows if row.get("split") == "train"]
    x_values, lrs = plot_points(train_rows or rows, "lr", args.x_axis, args.smooth)
    plt.plot(x_values, lrs)
    plt.xlabel(args.x_axis.title())
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lr_path = output_dir / "learning_rate.png"
    plt.savefig(lr_path, dpi=160)
    plt.close()

    dice_rows = [row for row in rows if row.get("dice_score") is not None]
    dice_path = None
    if dice_rows:
        plt.figure(figsize=(10, 5))
        for split, split_rows in sorted(_group_rows(dice_rows, "split").items()):
            x_values, dice_scores = plot_points(
                split_rows,
                "dice_score",
                args.x_axis,
                args.smooth,
                aggregate_epoch=(split != "train"),
            )
            plt.plot(x_values, dice_scores, label=f"{split} dice_score")
        plt.xlabel(args.x_axis.title())
        plt.ylabel("Dice Score")
        plt.title("Segmentation Dice Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        dice_path = output_dir / "dice_score.png"
        plt.savefig(dice_path, dpi=160)
        plt.close()

    print(loss_path)
    print(lr_path)
    if dice_path is not None:
        print(dice_path)

    ratio_rows = [
        row for row in rows
        if row.get("positive_ratio") is not None
        and row.get("predicted_positive_ratio") is not None
    ]
    if ratio_rows:
        plt.figure(figsize=(10, 5))
        for split, split_rows in sorted(_group_rows(ratio_rows, "split").items()):
            x_values, target_ratios = plot_points(
                split_rows,
                "positive_ratio",
                args.x_axis,
                args.smooth,
                aggregate_epoch=(split != "train"),
            )
            _, pred_ratios = plot_points(
                split_rows,
                "predicted_positive_ratio",
                args.x_axis,
                args.smooth,
                aggregate_epoch=(split != "train"),
            )
            plt.plot(x_values, target_ratios, label=f"{split} target positive ratio")
            plt.plot(x_values, pred_ratios, label=f"{split} predicted positive ratio")
        plt.xlabel(args.x_axis.title())
        plt.ylabel("Positive Ratio")
        plt.title("Segmentation Foreground Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        ratio_path = output_dir / "positive_ratio.png"
        plt.savefig(ratio_path, dpi=160)
        plt.close()
        print(ratio_path)


def _group_rows(rows, *keys):
    grouped = defaultdict(list)
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        if len(group_key) == 1:
            group_key = group_key[0]
        grouped[group_key].append(row)
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training history CSV.")
    parser.add_argument("--history", default="checkpoints/history.csv")
    parser.add_argument("--output-dir", default="checkpoints/plots")
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--x-axis", choices=["step", "epoch"], default="epoch")
    return parser.parse_args()


if __name__ == "__main__":
    plot_history(parse_args())
