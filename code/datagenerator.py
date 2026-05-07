import os
import base64
import json
import mimetypes
import re
from collections import Counter

from click import prompt
import anthropic
from distilabel.llms import AnthropicLLM
from openai.types.beta.threads import image_file
from sympy.strategies.core import switch

from rag import RAG
from setup import *
from dotenv import load_dotenv
from tqdm import tqdm
# from llama_index.core.llama_dataset.generator import RagDatasetGenerator


load_dotenv()

anthopic_api_key = os.getenv("ANTHROPIC_API_KEY")
class DataGenerator:
    def __init__(self, task_names=None,text_top_k=3,image_top_k=3):
        self.topic_query = "small fault interpretation"
        self.client = anthropic.Anthropic(api_key=anthopic_api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.generated_path = Path(__file__).parent.parent / "generated_data"
        self.generated_path.mkdir(parents=True, exist_ok=True)
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.task_names = task_names or {
            ("image", "text"): "visual_qa",
            ("text", "image"): "image_gen_from_text",
            ("text","text"): "text_qa",
            ("image","image_target"): "image_seg",
            ("image","image"):"image_gen_from_image"
        }
        # self.llm = AnthropicLLM(model="claude-opus-4-6", api_key=anthopic_api_key)
        # self.llm.load()

    def ranking_image_text(self,rotation=3):
        # 1. if both text and image show high score and in same page then group it if cant find high text score that less than image score then move to next lesser score text
        # 2. if image's neighbors text has highly same score then group it (different page)
        # 3. if there is no page (page is None) by pass happens (pair by rotate images around relevant context)
        # 4. every left-over text get pairs
        # 5. every left-over image get pairs
        texts,images = self.get_retrival()
        texts = sorted(texts, key=lambda x: x[1], reverse=True)
        images = sorted(images, key=lambda x: x[1], reverse=True)

        groups = []
        # used_text_ids = set()
        for image_idx, image in enumerate(images):
            image_node, image_score, image_page = image

            if image_page is None: # by pass non existed page in data
                candidates = texts
                rotate_text = True
            else:
                rotate_text = False
                same_page_texts = [
                    text for text in texts
                    if text[2] == image_page
                ]
                nearby_texts = [
                    text for text in texts
                    if (
                       text[2] is not None
                        and (abs(text[2] - image_page) <= 1 )
                    )
                ]
                candidates = same_page_texts or nearby_texts

            if not candidates: # can add check index if it still run but no candidate (in this case want to map relational data e.g. seismic and fault)
                # special for seismic's data
                if "fault_mask_path" in image_node["metadata"] or "fault_overlay_path" in image_node["metadata"]:
                    pack_data= {
                        "image": image_node,
                        "image_score":image_score,  # from seismic
                        "text": None,
                        "text_score": None,
                        "page_distance": 0,
                        "pair_score": image_score,
                    }
                    groups.append(pack_data)

            else:
                if rotate_text:
                    rotation_size = min(rotation, len(candidates))
                    best_text = candidates[image_idx % rotation_size]
                else:
                    best_text = max(candidates, key=lambda x: x[1])

                groups.append(
                    {
                        "image": image_node,
                        "image_score": image_score,
                        "text": best_text[0],
                        "text_score": best_text[1],
                        "page_distance": abs(best_text[2] - image_page if image_page is not None else 0),
                        "pair_score": image_score + best_text[1],
                    }
                )
        leftover_texts = [
            text for text in texts
        ]
        for first_text, second_text in zip(leftover_texts, leftover_texts[1:]):
            first_node, first_score, first_page = first_text
            second_node, second_score, second_page = second_text
            groups.append(
                {
                    "image": None,
                    "image_score": None,
                    "text": first_node,
                    "text_score": first_score,
                    "text_2": second_node,
                    "text_2_score": second_score,
                    "page_distance": abs(second_page - first_page)
                    if first_page is not None and second_page is not None
                    else 0,
                    "pair_score": first_score + second_score,
                }
            )

        groups.sort(key=lambda x: (x["page_distance"], -x["pair_score"])) # low pages distance and high page score
        # print("groups:", groups)
        return groups

    def packing(self):
        groups = self.ranking_image_text()
        cleaned_groups = []
        for group in groups:
            get_image = group["image"] # image metadata
            get_text = group["text"]  # text metadata

            if get_image is None:
                get_text_2 = group.get("text_2")
                if get_text and get_text_2:
                    cleaned_groups.append((
                        {
                            "type": get_text["type"],
                            "content": get_text["content"],
                        },
                        {
                            "type": get_text_2["type"],
                            "content": get_text_2["content"],
                        },
                    ))
                continue

            structure_imagedata = {
                "type": get_image["type"],
                "content": get_image["image_path"],
            }
            # in case of similarity is low for other images such as mask
            if "fault_mask_path" in get_image["metadata"]:
                structure_mask_data = {"type":"image_target",
                                      "content":get_image["metadata"]["fault_mask_path"]}
                cleaned_groups.append((structure_imagedata,structure_mask_data))

            if  "fault_overlay_path" in get_image["metadata"]:
                structure_overlay_data = {"type": "image",
                                       "content": get_image["metadata"]["fault_overlay_path"]}
                cleaned_groups.append((structure_imagedata, structure_overlay_data))

            if get_text:
                structure_textdata = {
                    "type": get_text["type"],
                    "content": get_text["content"],
                }
                cleaned_groups.append((structure_imagedata,structure_textdata)) # index like this for easier task determination
        return cleaned_groups

    def get_task_name(self, *items):
        types = tuple(item["type"] for item in items)
        return self.task_names.get(types, None)

    def get_retrival(self):
        images_types = ["image","image_target"] # in case similarity work it will find "image_target"
        with RAG(source_path=source_path, storage_path=storage_path) as rag:
            results = rag.retrieve(self.topic_query, self.text_top_k, self.image_top_k)

        text_nodes = []
        image_nodes = []

        for item in results:

            if item.get("type",None) in images_types:
                image_nodes.append((item, item.get("score",None), item.get("page",None))) # ranking use indexing ,so no change
            else:
                text_nodes.append((item, item.get("score",None), item.get("page",None))) # ranking use indexing ,so no change
        return text_nodes,image_nodes

    @staticmethod
    def _task_value(config, task_name, default=None):
        if config is None:
            return default
        value = config.get(task_name, default)
        return default if value is None else value

    @staticmethod
    def _remaining_task_capacity(task_counts, task_name, max_rows_per_task):
        cap = DataGenerator._task_value(max_rows_per_task, task_name)
        if cap is None:
            return None
        return max(0, cap - task_counts[task_name])

    @staticmethod
    def _append_allowed(task_counts, task_name, max_rows_per_task):
        remaining = DataGenerator._remaining_task_capacity(
            task_counts,
            task_name,
            max_rows_per_task,
        )
        return remaining is None or remaining > 0

    def generate(
        self,
        questions_per_reference=3,
        output_name="multimodal_qa.jsonl",
        questions_per_task=None,
        max_rows_per_task=None,
    ):
        package = self.packing()
        dataset = []
        task_counts = Counter()

        for first, second in tqdm(package, desc="Generating QA pairs"):
            first_content = first["content"] # might be any  type
            second_content= second["content"] # might be any type
            task_name = self.get_task_name(first, second)
            if task_name is None:
                continue

            remaining_capacity = self._remaining_task_capacity(
                task_counts,
                task_name,
                max_rows_per_task,
            )
            if remaining_capacity == 0:
                continue

            task_questions_per_reference = self._task_value(
                questions_per_task,
                task_name,
                questions_per_reference,
            )
            if remaining_capacity is not None:
                task_questions_per_reference = min(
                    task_questions_per_reference,
                    remaining_capacity,
                )
            if task_questions_per_reference <= 0:
                continue

            match task_name:
                case "visual_qa":
                    prompt = f"""
                            You are generating a multimodal QA dataset.
                            
                            Task: {task_name}
                            
                            Use ONLY the image and text context.
                            Generate {task_questions_per_reference} question-answer pairs.
                            
                            Rules:
                            - Each question must require visual understanding of the image.
                            - Use the text context as supporting evidence.
                            - Answers must be text.
                            - Do not include image bytes or base64 in the answer.
                            - Do not mention "provided image" or "provided text".
                            - If the image and text are not related, return an empty list.
                            - Return valid JSON only.
                            
                            Text context:
                            {second_content}
                            
                            Return format:
                            [
                              {{
                                "question": "...",
                                "answer": "..."
                              }}
                            ]
                            """
                    qa_pairs = self.call_vlm(prompt, image_paths=[first_content]) # image only

                    for qa in qa_pairs:
                        if not self._append_allowed(task_counts, task_name, max_rows_per_task):
                            break
                        dataset.append({
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "task": task_name,
                            "reference_text": second_content,
                            "reference_image_path": first_content,
                        })
                        task_counts[task_name] += 1
                case "image_seg":
                    prompt = f"""
                            You are generating an image segmentation dataset.
                            
                            Task: {task_name}
                            
                            Use the input image to write {task_questions_per_reference} segmentation instructions.
                            The expected segmentation output is stored separately by target_image_path.
                            
                            Rules:
                            - Ask for a segmentation mask or fault mask from the input image.
                            - The response target is an image, not text.
                            - Do not return image bytes yourself.
                            - Do not mention file paths.
                            - Return valid JSON only.
                            
                            Return format:
                            [
                              {{
                                "instruction": "...",
                                "output_type": "image_path"
                              }}
                            ]
                            """
                    qa_pairs = self.call_vlm(prompt, image_paths=[first_content,second_content])

                    for qa in qa_pairs:
                        if not self._append_allowed(task_counts, task_name, max_rows_per_task):
                            break
                        dataset.append({
                            "instruction": qa["instruction"],
                            "output_type": "image_path",
                            "task": task_name,
                            "reference_image_path": first_content,
                            "target_image_path": second_content,
                        })
                        task_counts[task_name] += 1
                case "image_gen_from_text":
                    prompt = f"""
                            You are generating a text-to-image dataset.
                            
                            Task: {task_name}
                            
                            Use the target image to write {task_questions_per_reference} text prompts that could generate it.
                            The expected image output is stored separately by target_image_path.
                            
                            Rules:
                            - Prompts must describe the target image clearly.
                            - The response target is an image, not text.
                            - Do not return image bytes yourself.
                            - Do not mention file paths.
                            - Return valid JSON only.
                            
                            Return format:
                            [
                              {{
                                "prompt": "...",
                                "output_type": "image_path"
                              }}
                            ]
                            """
                    qa_pairs = self.call_vlm(prompt, image_paths=[second_content])

                    for qa in qa_pairs:
                        if not self._append_allowed(task_counts, task_name, max_rows_per_task):
                            break
                        dataset.append({
                            "prompt": qa["prompt"],
                            "output_type": "image_path",
                            "task": task_name,
                            "reference_text": first_content,
                            "target_image_path": second_content,
                        })
                        task_counts[task_name] += 1
                case "image_gen_from_image":
                    prompt = f"""
                            You are generating an image-to-image dataset.
                            
                            Task: {task_name}
                            
                            Use the input image and target image relationship to write {task_questions_per_reference} image editing or image transformation instructions.
                            The expected output image is stored separately by target_image_path.
                            
                            Rules:
                            - Instructions must describe how to transform the input image into the target image.
                            - The response target is an image, not text.
                            - Do not return image bytes yourself.
                            - Do not mention file paths.
                            - Return valid JSON only.
                            
                            Return format:
                            [
                              {{
                                "instruction": "...",
                                "output_type": "image_path"
                              }}
                            ]
                            """
                    qa_pairs = self.call_vlm(prompt, image_paths=[first_content,second_content]) # both images input

                    for qa in qa_pairs:
                        if not self._append_allowed(task_counts, task_name, max_rows_per_task):
                            break
                        dataset.append({
                            "instruction": qa["instruction"],
                            "output_type": "image_path",
                            "task": task_name,
                            "reference_image_path": first_content,
                            "target_image_path": second_content,
                        })
                        task_counts[task_name] += 1
                case "text_qa":
                    prompt = f"""
                            You are generating a text QA dataset.
                            
                            Task: {task_name}
                            
                            Use ONLY the text context below.
                            Generate {task_questions_per_reference} question-answer pairs.
                            
                            Rules:
                            - Answers must be text.
                            - Do not invent facts beyond the context.
                            - Return valid JSON only.
                            
                            Text context:
                            {first_content}
                            
                            Additional text context:
                            {second_content}
                            
                            Return format:
                            [
                              {{
                                "question": "...",
                                "answer": "..."
                              }}
                            ]
                            """
                    qa_pairs = self.call_llm(prompt)

                    for qa in qa_pairs:
                        if not self._append_allowed(task_counts, task_name, max_rows_per_task):
                            break
                        dataset.append({
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "task": task_name,
                            "reference_text": first_content,
                            "reference_text_2": second_content,
                        })
                        task_counts[task_name] += 1
                case _:
                    continue # skip unsupported task pairs

        output_path = self.generated_path / output_name
        with output_path.open("w", encoding="utf-8") as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Generated rows by task: {dict(task_counts)}")
        return output_path

    def call_llm(self, prompt):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"LLM response was not valid JSON: {text}")

    def call_vlm(self, prompt, image_paths):
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        content = []
        for image_path in image_paths:
            media_type, _ = mimetypes.guess_type(image_path)
            if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                raise ValueError(f"Unsupported image type for Anthropic vision: {image_path}")

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"VLM response was not valid JSON: {text}")


# gen = DataGenerator()
# path = gen.generate(questions_per_reference=5)
# print(path)

if __name__ == "__main__":
    gen = DataGenerator(image_top_k=100,text_top_k=100,)
    print(gen.generate( questions_per_task={
          "visual_qa": 2,
          "image_seg": 1,
          "text_qa": 3,
          "image_gen_from_image": 1,
      },
      max_rows_per_task={
          "visual_qa": 500,
          "image_seg": 500,
          "text_qa": 500,
          "image_gen_from_image": 0,
          "image_gen_from_text": 0,
      },))
