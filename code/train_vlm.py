import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from collator import JsonlTaskDataset, MultiTaskCollator, TaskGroupedBatchSampler
from vlm import TEXT_TASKS, multitaskVLM


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device, non_blocking=(device.type == "cuda"))
        else:
            moved[key] = value
    return moved


def batch_outputs_and_loss(model, batch):
    task = batch.pop("task")
    outputs = model(task=task, **batch)
    if task in TEXT_TASKS:
        return task, outputs, outputs.loss
    if task == "image_seg":
        return task, outputs, outputs["loss"]
    raise ValueError(f"Unsupported task: {task}")


class MetricHistory:
    def __init__(self, output_dir, resume=False):
        self.output_dir = Path(output_dir)
        self.jsonl_path = self.output_dir / "history.jsonl"
        self.csv_path = self.output_dir / "history.csv"
        self.fieldnames = [
            "step",
            "epoch",
            "split",
            "task",
            "loss",
            "lr",
            "bce_loss",
            "dice_loss",
            "dice_score",
            "positive_ratio",
            "predicted_positive_ratio",
        ]

        if not resume:
            self.jsonl_path.write_text("", encoding="utf-8")
            with self.csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
        elif not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row):
        clean_row = {
            "step": int(row["step"]),
            "epoch": int(row["epoch"]),
            "split": row.get("split", "train"),
            "task": row["task"],
            "loss": float(row["loss"]),
            "lr": float(row["lr"]),
            "bce_loss": self._optional_float(row.get("bce_loss")),
            "dice_loss": self._optional_float(row.get("dice_loss")),
            "dice_score": self._optional_float(row.get("dice_score")),
            "positive_ratio": self._optional_float(row.get("positive_ratio")),
            "predicted_positive_ratio": self._optional_float(row.get("predicted_positive_ratio")),
        }

        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(clean_row) + "\n")

        with self.csv_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(clean_row)

    @staticmethod
    def _optional_float(value):
        if value is None:
            return ""
        return float(value)


class RowsDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def split_key(row):
    return (
        row.get("reference_image_path")
        or row.get("target_image_path")
        or row.get("reference_text")
        or row.get("question")
        or row.get("instruction")
        or "unknown"
    )


def split_rows(rows, val_ratio=0.1, test_ratio=0.1, seed=42):
    rows_by_key = defaultdict(list)
    for row in rows:
        rows_by_key[split_key(row)].append(row)

    keys = list(rows_by_key)
    rng = random.Random(seed)
    rng.shuffle(keys)

    total_keys = len(keys)
    test_count = int(total_keys * test_ratio)
    val_count = int(total_keys * val_ratio)

    test_keys = set(keys[:test_count])
    val_keys = set(keys[test_count:test_count + val_count])

    train_rows = []
    val_rows = []
    test_rows = []
    for key, key_rows in rows_by_key.items():
        if key in test_keys:
            test_rows.extend(key_rows)
        elif key in val_keys:
            val_rows.extend(key_rows)
        else:
            train_rows.extend(key_rows)

    return train_rows, val_rows, test_rows


def make_dataloader(
    rows,
    batch_size,
    collator,
    device,
    shuffle,
    drop_last,
    num_workers,
):
    dataset = RowsDataset(rows)
    batch_sampler = TaskGroupedBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def format_parameter_count(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def parameter_counts(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def print_parameter_report(model):
    total, trainable = parameter_counts(model)
    percent = (trainable / total * 100.0) if total else 0.0
    print(
        "parameters: "
        f"trainable={format_parameter_count(trainable)} "
        f"total={format_parameter_count(total)} "
        f"({percent:.2f}%)"
    )

    modules = [
        ("vision_encoder", model.vision_encoder),
        ("qformer", model.qformer),
        ("visual_projection", model.visual_projection),
        ("segmentation_decoder", model.segmentation_decoder),
        ("llm", model.llm),
    ]
    for name, module in modules:
        module_total, module_trainable = parameter_counts(module)
        module_percent = (module_trainable / module_total * 100.0) if module_total else 0.0
        print(
            f"  {name}: "
            f"trainable={format_parameter_count(module_trainable)} "
            f"total={format_parameter_count(module_total)} "
            f"({module_percent:.2f}%)"
        )

    query_trainable = model.query_tokens.numel() if model.query_tokens.requires_grad else 0
    print(
        "  query_tokens: "
        f"trainable={format_parameter_count(query_trainable)} "
        f"total={format_parameter_count(model.query_tokens.numel())} "
        f"({100.0 if model.query_tokens.requires_grad else 0.0:.2f}%)"
    )

    lora_trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    )
    if lora_trainable:
        print(f"  lora_adapters: trainable={format_parameter_count(lora_trainable)}")


def load_model_and_optimizer(args, device):
    if args.resume:
        model, checkpoint = multitaskVLM.from_checkpoint(args.resume, map_location=device)
        if args.use_lora and not checkpoint.get("init_config", {}).get("use_lora"):
            raise ValueError(
                "--use-lora cannot be added while resuming a non-LoRA checkpoint. "
                "Start a fresh LoRA run, or resume a checkpoint that was created with --use-lora."
            )
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))
    else:
        model = multitaskVLM(
            vision_name=args.vision_name,
            llm_name=args.llm_name,
            freeze_vision=not args.train_vision,
            freeze_llm=not args.train_llm,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
        )
        checkpoint = None
        start_epoch = 0
        global_step = 0

    model.to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Enable LoRA or unfreeze part of the model.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if checkpoint and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    if checkpoint and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return model, optimizer, scheduler, checkpoint, start_epoch, global_step


def best_loss_from_checkpoint(checkpoint):
    if not checkpoint:
        return None
    extra = checkpoint.get("extra") or {}
    best_loss = extra.get("best_val_loss")
    if best_loss is None:
        return None
    return float(best_loss)


def best_loss_from_history(output_dir):
    history_path = Path(output_dir) / "history.csv"
    if not history_path.exists():
        return None

    losses_by_epoch = defaultdict(list)
    with history_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("split") != "val" or not row.get("loss"):
                continue
            losses_by_epoch[int(row["epoch"])].append(float(row["loss"]))

    if not losses_by_epoch:
        return None
    return min(
        sum(losses) / len(losses)
        for losses in losses_by_epoch.values()
    )


def output_metrics(outputs, task):
    if task != "image_seg":
        return {}

    return {
        "bce_loss": outputs["bce_loss"].item()
        if outputs["bce_loss"] is not None
        else None,
        "dice_loss": outputs["dice_loss"].item()
        if outputs["dice_loss"] is not None
        else None,
        "dice_score": outputs["dice_score"].item()
        if outputs["dice_score"] is not None
        else None,
        "positive_ratio": outputs["positive_ratio"].item()
        if outputs["positive_ratio"] is not None
        else None,
        "predicted_positive_ratio": outputs["predicted_positive_ratio"].item()
        if outputs["predicted_positive_ratio"] is not None
        else None,
    }


def save_split_rows(output_dir, train_rows, val_rows, test_rows):
    for name, rows in {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }.items():
        path = Path(output_dir) / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate(model, dataloader, device, history, epoch, global_step, lr, split):
    model.eval()
    totals_by_task = defaultdict(lambda: defaultdict(float))
    counts_by_task = defaultdict(int)

    for batch in tqdm(dataloader, desc=f"{split} epoch {epoch}", leave=False):
        batch = move_batch_to_device(batch, device)
        task, outputs, loss = batch_outputs_and_loss(model, batch)
        counts_by_task[task] += 1
        totals_by_task[task]["loss"] += loss.item()

        for key, value in output_metrics(outputs, task).items():
            if value is not None:
                totals_by_task[task][key] += value

    for task, count in counts_by_task.items():
        row = {
            "step": global_step,
            "epoch": epoch,
            "split": split,
            "task": task,
            "loss": totals_by_task[task]["loss"] / count,
            "lr": lr,
        }

        for key in [
            "bce_loss",
            "dice_loss",
            "dice_score",
            "positive_ratio",
            "predicted_positive_ratio",
        ]:
            if key in totals_by_task[task]:
                row[key] = totals_by_task[task][key] / count
            else:
                row[key] = None

        history.append(row)

    model.train()
    if not counts_by_task:
        return None
    total_loss = sum(totals_by_task[task]["loss"] for task in counts_by_task)
    total_count = sum(counts_by_task.values())
    return total_loss / total_count


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model, optimizer, scheduler, checkpoint, start_epoch, global_step = load_model_and_optimizer(args, device)
    print_parameter_report(model)
    image_processor = AutoImageProcessor.from_pretrained(model.vision_name)

    dataset = JsonlTaskDataset(args.dataset)
    train_rows, val_rows, test_rows = split_rows(
        dataset.rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    save_split_rows(output_dir, train_rows, val_rows, test_rows)

    collator = MultiTaskCollator(
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        max_length=args.max_length,
    )
    dataloader = make_dataloader(
        train_rows,
        batch_size=args.batch_size,
        collator=collator,
        device=device,
        shuffle=True,
        drop_last=args.drop_last,
        num_workers=args.num_workers,
    )
    val_dataloader = make_dataloader(
        val_rows,
        batch_size=args.batch_size,
        collator=collator,
        device=device,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_dataloader = make_dataloader(
        test_rows,
        batch_size=args.batch_size,
        collator=collator,
        device=device,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    history = MetricHistory(output_dir, resume=bool(args.resume))
    best_val_loss = (
        best_loss_from_checkpoint(checkpoint)
        if checkpoint is not None
        else None
    )
    history_best_val_loss = best_loss_from_history(output_dir) if args.resume else None
    if history_best_val_loss is not None:
        best_val_loss = (
            history_best_val_loss
            if best_val_loss is None
            else min(best_val_loss, history_best_val_loss)
        )
    if best_val_loss is not None:
        print(f"best val loss before training: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            task = batch["task"]
            batch = move_batch_to_device(batch, device)
            task, outputs, loss = batch_outputs_and_loss(model, batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            lr = optimizer.param_groups[0]["lr"]
            loss_value = loss.item()
            history.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "split": "train",
                    "task": task,
                    "loss": loss_value,
                    "lr": lr,
                    **output_metrics(outputs, task),
                }
            )
            progress.set_postfix(task=task, loss=f"{loss_value:.4f}")

            if args.save_every and global_step % args.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint-step-{global_step}.pt"
                model.save_checkpoint(
                    checkpoint_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                )

        if val_rows:
            val_loss = evaluate(
                model,
                val_dataloader,
                device,
                history,
                epoch=epoch + 1,
                global_step=global_step,
                lr=optimizer.param_groups[0]["lr"],
                split="val",
            )
            if val_loss is not None:
                scheduler.step(val_loss)
                print(f"epoch {epoch + 1} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g}")
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = output_dir / "checkpoint-best.pt"
                    model.save_checkpoint(
                        best_path,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        step=global_step,
                        extra={
                            "best_val_loss": best_val_loss,
                            "best_epoch": epoch + 1,
                            "train_rows": len(train_rows),
                            "val_rows": len(val_rows),
                            "test_rows": len(test_rows),
                        },
                    )
                    print(f"saved best checkpoint: {best_path} val_loss={best_val_loss:.4f}")

        checkpoint_path = output_dir / f"checkpoint-epoch-{epoch + 1}.pt"
        model.save_checkpoint(
            checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            extra={
                "best_val_loss": best_val_loss,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "test_rows": len(test_rows),
            },
        )

    final_path = output_dir / "checkpoint-final.pt"
    model.save_checkpoint(
        final_path,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        step=global_step,
        extra={"best_val_loss": best_val_loss},
    )
    if test_rows:
        evaluate(
            model,
            test_dataloader,
            device,
            history,
            epoch=args.epochs,
            global_step=global_step,
            lr=optimizer.param_groups[0]["lr"],
            split="test",
        )
    return final_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train multitask VLM with checkpoint resume.")
    parser.add_argument("--dataset", default="generated_data/multimodal_qa_200.jsonl")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--vision-name", default="facebook/dinov2-base")
    parser.add_argument("--llm-name", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--train-llm", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    checkpoint = train(parse_args())
    print(checkpoint)
