import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2QFormerConfig,
    Blip2QFormerModel,
)


TEXT_TASKS = {"visual_qa", "text_qa"}
SEGMENTATION_TASKS = {"image_seg"}
SUPPORTED_TASKS = TEXT_TASKS | SEGMENTATION_TASKS


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class LightUNetDecoder(nn.Module):
    def __init__(self, in_channels, base_channels=128):
        super().__init__()
        self.input_block = ConvBlock(in_channels, base_channels)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=2, stride=2)
        self.block1 = ConvBlock(base_channels // 2, base_channels // 2)
        self.up2 = nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=2, stride=2)
        self.block2 = ConvBlock(base_channels // 4, base_channels // 4)
        self.up3 = nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=2, stride=2)
        self.block3 = ConvBlock(base_channels // 8, base_channels // 8)
        self.out_conv = nn.Conv2d(base_channels // 8, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_block(x)
        x = self.block1(self.up1(x))
        x = self.block2(self.up2(x))
        x = self.block3(self.up3(x))
        return self.out_conv(x)


class multitaskVLM(nn.Module):
    def __init__(
        self,
        vision_name="facebook/dinov2-base",
        llm_name="HuggingFaceTB/SmolLM-135M",
        num_query_tokens=32,
        qformer_hidden_size=768,
        freeze_vision=True,
        freeze_llm=True,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        llm_quantization_config=None,
        llm_device_map=None,
    ):
        super().__init__()
        self.vision_name = vision_name
        self.llm_name = llm_name
        self.num_query_tokens = num_query_tokens
        self.init_config = {
            "vision_name": vision_name,
            "llm_name": llm_name,
            "num_query_tokens": num_query_tokens,
            "qformer_hidden_size": qformer_hidden_size,
            "freeze_vision": freeze_vision,
            "freeze_llm": freeze_llm,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
        }

        self.vision_encoder = AutoModel.from_pretrained(vision_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if llm_quantization_config is not None:
            llm_kwargs["quantization_config"] = llm_quantization_config
        if llm_device_map is not None:
            llm_kwargs["device_map"] = llm_device_map

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, **llm_kwargs)

        vision_hidden = self.vision_encoder.config.hidden_size
        llm_hidden = self.llm.config.hidden_size

        self.qformer_config = Blip2QFormerConfig(
            hidden_size=qformer_hidden_size,
            encoder_hidden_size=vision_hidden,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=qformer_hidden_size * 4,
            cross_attention_frequency=2,
        )
        self.qformer = Blip2QFormerModel(self.qformer_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, qformer_hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.visual_projection = nn.Linear(qformer_hidden_size, llm_hidden)
        self.segmentation_decoder = LightUNetDecoder(vision_hidden)

        if freeze_vision:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False

        if freeze_llm:
            for parameter in self.llm.parameters():
                parameter.requires_grad = False

        if use_lora:
            self._apply_lora(
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )

    def _apply_lora(self, r, alpha, dropout, target_modules):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LoRA training requires peft. Install project dependencies again, "
                "for example: uv sync"
            ) from exc

        if isinstance(target_modules, str):
            target_modules = [
                module.strip()
                for module in target_modules.split(",")
                if module.strip()
            ]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def format_prompt(self, instruction, task):
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}")
        return f"Task: {task}\nInstruction: {instruction}"

    def _qformer_output(self, pixel_values):
        if pixel_values is None:
            raise ValueError("pixel_values is required for image tasks")

        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        return qformer_outputs.last_hidden_state

    def _visual_embeds(self, pixel_values, dtype=None):
        query_output = self._qformer_output(pixel_values)
        visual_embeds = self.visual_projection(query_output)
        if dtype is not None:
            visual_embeds = visual_embeds.to(dtype=dtype)
        return visual_embeds

    def _vision_feature_map(self, pixel_values):
        if pixel_values is None:
            raise ValueError("pixel_values is required for image_seg")

        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        tokens = vision_outputs.last_hidden_state
        patch_tokens = tokens[:, 1:, :]

        num_patches = patch_tokens.shape[1]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Cannot reshape {num_patches} vision patches to a square grid")

        return patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            grid_size,
            grid_size,
        )

    def forward_text(
        self,
        input_ids,
        attention_mask,
        labels=None,
        pixel_values=None,
        use_image=True,
    ):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = text_embeds
        combined_attention_mask = attention_mask

        if use_image:
            visual_embeds = self._visual_embeds(pixel_values, dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            visual_attention_mask = torch.ones(
                visual_embeds.shape[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

            if labels is not None:
                visual_labels = torch.full(
                    visual_attention_mask.shape,
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
        )

    def forward_segmentation(self, pixel_values, masks=None, output_size=None):
        feature_map = self._vision_feature_map(pixel_values)
        logits = self.segmentation_decoder(feature_map)

        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        elif masks is not None:
            target_size = masks.shape[-2:]
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        loss = None
        bce_loss = None
        dice_loss = None
        dice_score = None
        positive_ratio = None
        predicted_positive_ratio = None
        if masks is not None:
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = masks.to(device=logits.device, dtype=logits.dtype)
            positive_ratio = masks.mean()
            positive_pixels = masks.sum().clamp_min(1.0)
            negative_pixels = (masks.numel() - masks.sum()).clamp_min(1.0)
            pos_weight = (negative_pixels / positive_pixels).clamp(max=100.0)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits,
                masks,
                pos_weight=pos_weight,
            )
            probs = torch.sigmoid(logits)
            predicted_positive_ratio = (probs >= 0.5).to(dtype=probs.dtype).mean()
            smooth = 1.0
            intersection = (probs * masks).sum(dim=(1, 2, 3))
            denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice_score = ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
            dice_loss = 1.0 - dice_score
            loss = bce_loss + dice_loss

        return {
            "loss": loss,
            "logits": logits,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "dice_score": dice_score,
            "positive_ratio": positive_ratio,
            "predicted_positive_ratio": predicted_positive_ratio,
        }

    def forward(
        self,
        task,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        masks=None,
    ):
        if task == "visual_qa":
            return self.forward_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                use_image=True,
            )

        if task == "text_qa":
            return self.forward_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_image=False,
            )

        if task == "image_seg":
            return self.forward_segmentation(pixel_values=pixel_values, masks=masks)

        raise ValueError(f"Unsupported task: {task}")

    @torch.no_grad()
    def generate_text(self, task, input_ids, attention_mask, pixel_values=None, **generate_kwargs):
        if task not in TEXT_TASKS:
            raise ValueError(f"generate_text only supports text tasks, got: {task}")

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = text_embeds
        combined_attention_mask = attention_mask

        if task == "visual_qa":
            visual_embeds = self._visual_embeds(pixel_values, dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            visual_attention_mask = torch.ones(
                visual_embeds.shape[:-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

    @torch.no_grad()
    def predict_mask(self, pixel_values, output_size=None, threshold=None):
        result = self.forward_segmentation(pixel_values=pixel_values, output_size=output_size)
        probs = torch.sigmoid(result["logits"])
        if threshold is not None:
            return probs >= threshold
        return probs

    def checkpoint_payload(self, optimizer=None, scheduler=None, epoch=0, step=0, extra=None):
        payload = {
            "model_state": self.state_dict(),
            "init_config": self.init_config,
            "epoch": epoch,
            "step": step,
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state"] = scheduler.state_dict()
        if extra is not None:
            payload["extra"] = extra
        return payload

    def save_checkpoint(self, path, optimizer=None, scheduler=None, epoch=0, step=0, extra=None):
        torch.save(
            self.checkpoint_payload(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                extra=extra,
            ),
            path,
        )

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", **override_config):
        checkpoint = torch.load(path, map_location=map_location)
        config = dict(checkpoint.get("init_config", {}))
        config.update(override_config)
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state"])
        return model, checkpoint


VLM = multitaskVLM
