import json
from collections import Counter
from pathlib import Path


def dataset_check():
    dataset_path = Path(__file__).parent.parent / "generated_data" / "multimodal_qa.jsonl"
    counts = Counter()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            counts[row.get("task", "unknown")] += 1

    for task, total in sorted(counts.items()):
        print(f"{task}: {total}")

    return counts


if __name__ == "__main__":
    dataset_check()
