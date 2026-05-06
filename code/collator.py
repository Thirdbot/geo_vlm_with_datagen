import base64
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler


TEXT_TASKS = {"visual_qa", "text_qa"}
SEGMENTATION_TASKS = {"image_seg"}
SUPPORTED_TASKS = TEXT_TASKS | SEGMENTATION_TASKS


class JsonlTaskDataset(Dataset):
    def __init__(self, path, supported_tasks=SUPPORTED_TASKS):
        self.path = Path(path)
        self.rows = []

        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                if row.get("task") in supported_tasks:
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TaskGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices_by_task = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            self.indices_by_task[row["task"]].append(index)

    def __iter__(self):
        batches = []

        for indices in self.indices_by_task.values():
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        total = 0
        for indices in self.indices_by_task.values():
            full_batches, remainder = divmod(len(indices), self.batch_size)
            total += full_batches
            if remainder and not self.drop_last:
                total += 1
        return total


class MultiTaskCollator:
    def __init__(
        self,
        tokenizer,
        image_processor,
        max_length=512,
        mask_prompt_labels=True,
        resize_masks_to_image=True,
        mask_dilation=1,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.mask_prompt_labels = mask_prompt_labels
        self.resize_masks_to_image = resize_masks_to_image
        self.mask_dilation = mask_dilation

    def __call__(self, rows):
        tasks = {row["task"] for row in rows}
        if len(tasks) != 1:
            raise ValueError(
                "MultiTaskCollator expects one task per batch. "
                "Use TaskGroupedBatchSampler to avoid mixed-task batches."
            )

        task = rows[0]["task"]
        if task in TEXT_TASKS:
            return self._collate_text(rows, task)
        if task in SEGMENTATION_TASKS:
            return self._collate_segmentation(rows)

        raise ValueError(f"Unsupported task: {task}")

    def _collate_text(self, rows, task):
        prompts = []
        targets = []
        images = []

        for row in rows:
            if task == "visual_qa":
                prompts.append(f"Task: visual_qa\nQuestion: {row['question']}\nAnswer:")
                targets.append(row["answer"])
                images.append(self._load_rgb(row["reference_image_path"]))
            elif task == "text_qa":
                context = row.get("reference_text", "")
                context_2 = row.get("reference_text_2", "")
                prompts.append(
                    "Task: text_qa\n"
                    f"Context: {context}\n"
                    f"Additional context: {context_2}\n"
                    f"Question: {row['question']}\n"
                    "Answer:"
                )
                targets.append(row["answer"])

        full_texts = [f"{prompt} {target}" for prompt, target in zip(prompts, targets)]
        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        if self.mask_prompt_labels:
            prompt_tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            for row_index, prompt_mask in enumerate(prompt_tokens["attention_mask"]):
                prompt_length = int(prompt_mask.sum().item())
                labels[row_index, :prompt_length] = -100

        batch = {
            "task": task,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

        if task == "visual_qa":
            batch["pixel_values"] = self.image_processor(
                images,
                return_tensors="pt",
            )["pixel_values"]

        return batch

    def _collate_segmentation(self, rows):
        prompts = []
        images = []
        masks = []

        for row in rows:
            prompts.append(f"Task: image_seg\nInstruction: {row['instruction']}")
            images.append(self._load_rgb(row["reference_image_path"]))
            masks.append(self._load_mask(row))

        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        pixel_values = self.image_processor(images, return_tensors="pt")["pixel_values"]
        masks = torch.stack(masks).unsqueeze(1)
        if self.resize_masks_to_image and masks.shape[-2:] != pixel_values.shape[-2:]:
            masks = F.interpolate(
                masks,
                size=pixel_values.shape[-2:],
                mode="nearest",
            )
        if self.mask_dilation > 0:
            for _ in range(self.mask_dilation):
                masks = F.max_pool2d(masks, kernel_size=3, stride=1, padding=1)

        return {
            "task": "image_seg",
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "pixel_values": pixel_values,
            "masks": masks,
        }

    @staticmethod
    def _load_rgb(path):
        return Image.open(path).convert("RGB")

    def _load_mask(self, row):
        if row.get("result_image_base64"):
            image_bytes = base64.b64decode(row["result_image_base64"])
            mask = Image.open(BytesIO(image_bytes)).convert("L")
        else:
            mask = Image.open(row["target_image_path"]).convert("L")

        mask_array = np.array(mask, dtype=np.float32) / 255.0
        return torch.from_numpy(mask_array)
