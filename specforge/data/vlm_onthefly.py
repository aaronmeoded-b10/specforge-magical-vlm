"""
On-the-fly VLM dataset for EAGLE-3 training.

Instead of pre-tokenizing all images into Arrow cache (which uses ~22MB per image
and blows up disk at scale), this processes images on-the-fly during training.

Each __getitem__ call:
1. Loads the raw conversation from JSONL
2. Processes images through the Qwen processor
3. Returns tokenized input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
4. Pixel values are discarded after the training step (no disk cache)
"""

import json
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

from .template import TEMPLATE_REGISTRY, ChatTemplate


def _apply_loss_mask(input_ids, tokenizer, chat_template: ChatTemplate):
    """Create loss mask: 1 for assistant tokens, 0 for everything else."""
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)

    # Find assistant response spans using chat template markers
    ast_start = chat_template.assistant_start
    ast_end = chat_template.assistant_end

    tokens_so_far = 0
    text = decoded
    while ast_start in text:
        start_idx = text.index(ast_start)
        # Find the end of this assistant turn
        after_start = text[start_idx + len(ast_start):]
        if ast_end in after_start:
            end_idx = start_idx + len(ast_start) + after_start.index(ast_end)
        else:
            end_idx = len(text)

        # Map character positions to token positions (approximate)
        # Use the tokenizer to get exact positions
        prefix = text[:start_idx + len(ast_start)]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        content = text[:end_idx]
        content_tokens = tokenizer.encode(content, add_special_tokens=False)

        # Mark assistant tokens
        start_token = len(prefix_tokens)
        end_token = len(content_tokens)
        if start_token < len(loss_mask) and end_token <= len(loss_mask):
            loss_mask[start_token:end_token] = 1.0

        text = text[end_idx + len(ast_end):]

    return loss_mask


class VLMOnTheFlyDataset(Dataset):
    """Dataset that tokenizes VLM examples on-the-fly, avoiding massive Arrow caches."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        chat_template_name: str,
        max_length: int = 8192,
        max_examples: Optional[int] = None,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.chat_template = TEMPLATE_REGISTRY.get(chat_template_name)
        self.max_length = max_length

        # Load JSONL into memory (just the text, not tokenized — small)
        self.examples = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                self.examples.append(json.loads(line))

        print(f"VLMOnTheFlyDataset: loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        conversations = example["conversations"]

        # Build messages for the processor
        system_prompt = self.chat_template.system_prompt

        # Use per-row system prompt if present
        if conversations[0]["role"] == "system":
            messages = [{"role": "system", "content": conversations[0]["content"]}]
            source = conversations[1:]
        else:
            messages = [{"role": "system", "content": system_prompt}]
            source = conversations

        # Pass through conversations, handling both text and image content blocks
        for sentence in source:
            role = sentence["role"]
            content = sentence.get("content", "")
            if isinstance(content, list):
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": content})

        # Apply chat template
        conversation = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Extract images using qwen_vl_utils
        if not HAS_QWEN_VL_UTILS:
            raise ImportError("qwen_vl_utils is required for VLM on-the-fly processing")

        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs is None:
            image_inputs = []

        # Tokenize through processor
        proc_kwargs = dict(
            text=[conversation],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs

        encoding = self.processor(**proc_kwargs)

        input_ids = encoding.input_ids[0]
        pixel_values = getattr(encoding, "pixel_values", None)
        image_grid_thw = getattr(encoding, "image_grid_thw", None)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Create loss mask (1 for assistant tokens, 0 for prompt)
        # Simple approach: mark everything after the last assistant start as loss
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        decoded = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        ast_end_token = self.chat_template.assistant_end
        ast_start_token = self.chat_template.assistant_start

        # Find all assistant spans and mark them
        pos = 0
        while True:
            start = decoded.find(ast_start_token, pos)
            if start == -1:
                break
            end = decoded.find(ast_end_token, start + len(ast_start_token))
            if end == -1:
                end = len(decoded)
            else:
                end = end  # don't include the end token

            # Map character positions to token positions
            prefix_len = len(self.tokenizer.encode(decoded[:start + len(ast_start_token)], add_special_tokens=False))
            content_len = len(self.tokenizer.encode(decoded[:end], add_special_tokens=False))

            if prefix_len < len(loss_mask):
                loss_mask[prefix_len:min(content_len, len(loss_mask))] = 1.0

            pos = end + len(ast_end_token)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


class VLMOnTheFlyCollator:
    """Collator for on-the-fly VLM dataset. Handles variable-length sequences and images."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # For batch_size=1 (which is what EAGLE training uses), just return the single example
        if len(features) == 1:
            f = features[0]
            return {
                "input_ids": f["input_ids"].unsqueeze(0),
                "attention_mask": f["attention_mask"].unsqueeze(0),
                "loss_mask": f["loss_mask"].unsqueeze(0),
                "pixel_values": f["pixel_values"],  # already has right shape from processor
                "image_grid_thw": f["image_grid_thw"],  # [num_images, 3]
            }

        # For batch_size > 1, pad sequences
        max_len = max(f["input_ids"].size(0) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_loss_mask = []
        all_pixel_values = []
        all_grid_thw = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            batch_input_ids.append(
                torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            batch_attention_mask.append(
                torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            batch_loss_mask.append(
                torch.cat([f["loss_mask"], torch.zeros(pad_len, dtype=torch.float32)])
            )

            if f["pixel_values"] is not None:
                all_pixel_values.append(f["pixel_values"])
            if f["image_grid_thw"] is not None:
                all_grid_thw.append(f["image_grid_thw"])

        batch = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "loss_mask": torch.stack(batch_loss_mask),
            "pixel_values": torch.cat(all_pixel_values, dim=0) if all_pixel_values else None,
            "image_grid_thw": torch.cat(all_grid_thw, dim=0) if all_grid_thw else None,
        }
        return batch
