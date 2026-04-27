import json
import os.path
import re
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration
from qwen_vl_inference.vision_process import process_vision_info

min_pixels=256*28*28
max_pixels=1280*28*28

image_prompt = """You are an image relevance judge.
Decide how relevant the image is to the news Title and Summary.
Use the image title and caption if helpful.

Output one word:
Good / Fair / Bad

INPUT
Title: {news_title}
Summary: {news_summary}
ImageTitle: {image_title}
ImageCaption: {image_caption}"""

class VLM_Inference:
    def __init__(self, model_path: str, model_base: str, device: str = "cuda"):
        self.device = device
        self.best_threshold = 0.41
        self.processor, self.gen_model = self._load_model(model_path, model_base, device)

        bad_id = self.processor.tokenizer.encode("Bad", add_special_tokens=False)[0]
        fair_id = self.processor.tokenizer.encode("Fair", add_special_tokens=False)[0]
        good_id = self.processor.tokenizer.encode("Good", add_special_tokens=False)[0]

        self.model = self.gen_model.model
        self.score_linear = self.get_trinary_linear(self.gen_model, good_id, fair_id, bad_id)
        self.score_linear.eval()
        self.score_linear.to(self.device).to(self.model.dtype)
        self.model.eval()

    def get_trinary_linear(self, model, good_id, fair_id, bad_id):
        lm_head_weights = model.lm_head.weight.data

        weight_good = lm_head_weights[good_id]
        weight_fair = lm_head_weights[fair_id]
        weight_bad = lm_head_weights[bad_id]

        D = weight_good.size()[0]
        linear_layer = torch.nn.Linear(D, 3, bias=False)

        with torch.no_grad():
            linear_layer.weight[0,:] = weight_good
            linear_layer.weight[1,:] = weight_fair
            linear_layer.weight[2,:] = weight_bad
        return linear_layer

    def _load_model(self, model_path: str, model_base: str, device: str):
        processor = AutoProcessor.from_pretrained(model_base, use_fast=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_base,
                                                                low_cpu_mem_usage=True,
                                                                device_map=device,
                                                                dtype=torch.bfloat16)
        if os.path.exists(model_path):
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            model.tie_weights()
            model = model.to(dtype=torch.bfloat16)
            print("✅ LoRA model loaded and merged successfully.")
        print("✅model loaded successfully.")
        return processor, model

    def _load_lora_model(self, model_path: str, model_base: str, device: str):
        """Load LoRA model and merge weights into base Qwen2.5-VL model."""
        print(f"🔹 Loading LoRA model from {model_path}")
        processor = AutoProcessor.from_pretrained(model_base, use_fast=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_base,
                                                                low_cpu_mem_usage=True,
                                                                device_map=device,
                                                                dtype=torch.bfloat16)
        # Merge LoRA
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.tie_weights()
        model = model.to(dtype=torch.bfloat16)
        print("✅ LoRA model loaded and merged successfully.")
        return processor, model

    def build_prompt(self, entry: dict) -> dict:
        prompt_filled = image_prompt.format(
            news_title=entry.get("gem_title", ""),
            news_summary=entry.get("gem_summary", ""),
            image_title=entry.get("image_title", ""),
            image_caption=entry.get("image_caption", "")
        )
        return {
            "image": entry.get("image", ""),
            "prompt": "<image>\n" + prompt_filled,
        }

    def build_manual_prompt(self, conversation):
        turn = conversation[0]
        assert turn["role"] == "user"
        content = ""
        for c in turn["content"]:
            if c["type"] == "text":
                content += c["text"]
        pattern = r'\n?' + re.escape("<image>") + r'\n?'
        replacement = "<|vision_start|>" + "<|image_pad|>" + "<|vision_end|>"
        content = re.sub(pattern, replacement, content)

        prompt = (
            "<|im_start|>user\n"
            f"{content}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        return prompt

    def infer_sample(self, raw_entry: dict):
        result = {
            "pred": 0,
            "prob_rel": 0,
            "response": "",
            "error": "",
        }
        try:
            # 1️⃣ Build the input prompt and retrieve image path
            sample = self.build_prompt(raw_entry)
            user_prompt = sample["prompt"]
            image_path = sample["image"]

            # 2️⃣ Compose the conversation (text + image)
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image_path, "min_pixels": min_pixels, "max_pixels": max_pixels},
                ],
            }]

            # # 3️⃣ Prepare model inputs
            prompt = self.build_manual_prompt(conversation)
            image_inputs = process_vision_info(conversation, image_patch_size=16)
            inputs = self.processor(text=[prompt], images=image_inputs, videos=None, padding=False, do_resize=False,
                                    return_tensors="pt").to(self.device)

            # 4️⃣ Run model generation
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                )

            last_hidden = outputs.last_hidden_state[:, -1]
            class_logits = self.score_linear(last_hidden)
            probs = torch.softmax(class_logits, dim=-1)

            prob_good = probs[:, 0]
            prob_fair = probs[:, 1]

            prob_rel = (prob_good + 0.5 * prob_fair).item()
            result["prob_rel"] = prob_rel
            result["pred"] = int(prob_rel >= self.best_threshold)

        except Exception as e_main:
            result["error"] = str(e_main)
            result["pred"] = 0
            result["prob_rel"] = 0
            result["response"] = ""

        result_json = json.dumps(result, ensure_ascii=False)
        return result_json
