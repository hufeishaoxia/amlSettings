"""Point-wise SFT dataset for generic (user, item) click prediction.

Input format: JSONL, one user impression per line:
    {
        "history":   [{"title": "...", "summary": "..."}, ...],   # past clicked items (chronological)
        "interests": "tech, sports, finance",                     # free-text interest profile (optional)
        "items":     [{"title": "...", "summary": "...", "clicked": 0/1}, ...]
    }

Each line is expanded into one (history, candidate, label) sample per item in `items`.
"""

import json
import random
import torch


def load_samples(jsonl_path: str, max_history: int = 30):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            history = row.get("history", [])
            if max_history > 0:
                history = history[-max_history:]
            interests = row.get("interests", "")
            for item in row.get("items", []):
                samples.append({
                    "history":   history,
                    "interests": interests,
                    "candidate": {"title": item.get("title", ""),
                                   "summary": item.get("summary", "")},
                    "label":     int(item.get("clicked", 0)),
                })
    return samples


def build_prompt(history, interests, candidate):
    parts = []
    if interests:
        parts.append(f"User interests: {interests}")
    if history:
        parts.append("User reading history:")
        for i, h in enumerate(history, 1):
            t = h.get("title", "")
            s = h.get("summary", "")
            parts.append(f"{i}. {t}" + (f" - {s}" if s else ""))
    else:
        parts.append("User reading history: (none)")

    parts.append("")
    parts.append("Candidate item:")
    parts.append(candidate.get("title", "") +
                 (f" - {candidate.get('summary', '')}" if candidate.get("summary") else ""))
    parts.append("")
    parts.append("Will the user click this item? Answer:")
    return "\n".join(parts)


class PointwiseSFTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=2048, max_history=30,
                 use_chat_template=True, sample=-1, seed=42):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_chat_template = use_chat_template

        self.samples = load_samples(jsonl_path, max_history=max_history)
        if sample > 0 and sample < len(self.samples):
            random.Random(seed).shuffle(self.samples)
            self.samples = self.samples[:sample]

        pos = sum(s["label"] for s in self.samples)
        neg = len(self.samples) - pos
        print(f"[{jsonl_path}] {len(self.samples)} samples (pos={pos}, neg={neg})")

        # Per-class weights (inverse class frequency, mean weight = 1.0)
        if pos > 0 and neg > 0:
            total = pos + neg
            self.weights = {1: total / (2.0 * pos), 0: total / (2.0 * neg)}
        else:
            self.weights = {1: 1.0, 0: 1.0}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = build_prompt(s["history"], s["interests"], s["candidate"])
        target = " Yes" if s["label"] == 1 else " No"

        if self.use_chat_template:
            messages = [
                {"role": "system", "content":
                    "You are a recommendation assistant. Predict whether the user will click "
                    "the candidate item based on their history and interests. Answer Yes or No."},
                {"role": "user", "content": prompt},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full = formatted + target
            input_ids  = self.tokenizer.encode(full,      max_length=self.max_len,
                                               truncation=True, add_special_tokens=False)
            prompt_ids = self.tokenizer.encode(formatted, max_length=self.max_len,
                                               truncation=True, add_special_tokens=False)
        else:
            full = prompt + target
            input_ids  = self.tokenizer.encode(full,   max_length=self.max_len,
                                               truncation=True, add_special_tokens=True)
            prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_len,
                                               truncation=True, add_special_tokens=True)

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        input_ids = input_ids[:self.max_len]
        labels    = labels[:self.max_len]

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "labels":         torch.tensor(labels,    dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "weight":         torch.tensor(self.weights[s["label"]], dtype=torch.float),
        }
