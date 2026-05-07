import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from vlm import TEXT_TASKS, multitaskVLM


def load_image_processor(model):
    return AutoImageProcessor.from_pretrained(model.vision_name)


def load_model(checkpoint_path, device):
    model, _ = multitaskVLM.from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def build_text_prompt(args):
    if args.task == "visual_qa":
        return f"Task: visual_qa\nQuestion: {args.prompt}\nAnswer:"
    if args.task == "text_qa":
        return (
            "Task: text_qa\n"
            f"Context: {args.context}\n"
            f"Additional context: {args.context_2}\n"
            f"Question: {args.prompt}\n"
            "Answer:"
        )
    raise ValueError(f"Unsupported text task: {args.task}")


def infer_text(model, image_processor, args, device):
    prompt = build_text_prompt(args)
    tokens = model.tokenizer(prompt, return_tensors="pt").to(device)

    pixel_values = None
    if args.task == "visual_qa":
        if not args.image:
            raise ValueError("--image is required for visual_qa")
        image = Image.open(args.image).convert("RGB")
        pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)

    output_ids = model.generate_text(
        task=args.task,
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        pixel_values=pixel_values,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def infer_segmentation(model, image_processor, args, device):
    if not args.image:
        raise ValueError("--image is required for image_seg")

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
    output_size = image.size[::-1] if args.original_size else None

    probs = model.predict_mask(
        pixel_values=pixel_values,
        output_size=output_size,
        threshold=None,
    )
    mask = (probs[0, 0].detach().cpu().numpy() >= args.threshold).astype(np.uint8) * 255

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    return output_path.as_posix()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference from a multitask VLM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=["visual_qa", "text_qa", "image_seg"])
    parser.add_argument("--prompt", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--context-2", default="")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default="generated_data/predicted_mask.png")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--original-size", action="store_true")
    return parser.parse_args()


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)
    image_processor = load_image_processor(model)

    if args.task in TEXT_TASKS:
        result = infer_text(model, image_processor, args, device)
    else:
        result = infer_segmentation(model, image_processor, args, device)

    print(result)


if __name__ == "__main__":
    main(parse_args())
