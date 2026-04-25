"""Pairwise dataset and collator."""

import json
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from prompt import build_prompt, build_prompt_budgeted, encode_prompt


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(path: str) -> List[dict]:
    """Load preprocessed JSONL (one feed per line, candidates embedded)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class PairwiseDataset(torch.utils.data.Dataset):
    """Pairwise dataset: each item = (pos_prompt, neg_1, ..., neg_K).

    Negatives:
      1. In-session: non-clicked from same feed
      2. Cross-user: other users' clicked items (patched into this user's context)
    Until `num_negatives` per positive.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_len: int = 2048,
        use_chat_template: bool = True,
        num_negatives: int = 20,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_chat_template = use_chat_template
        self.num_negatives = num_negatives
        # Reserve tokens for chat template overhead
        self._body_budget = max(256, max_len - 150)

        records = load_jsonl(jsonl_path)
        print(f"[{_ts()}] Loaded {len(records)} feed records from {jsonl_path}")

        # Flatten into per-feed groups
        # Each record has .candidates = [{..., is_clicked: bool}, ...]
        rng = random.Random(seed)

        # Collect all positives across feeds for cross-user negatives
        all_positives = []  # (record_context, candidate)
        feed_groups = []    # (record_context, positives, negatives)

        for rec in records:
            candidates = rec.get("candidates", [])
            ctx = {k: v for k, v in rec.items() if k != "candidates"}
            pos = [c for c in candidates if c.get("is_clicked")]
            neg = [c for c in candidates if not c.get("is_clicked")]
            if pos:
                feed_groups.append((ctx, pos, neg))
                for c in pos:
                    all_positives.append((ctx, c))

        # Build pairs
        self.pairs = []  # each: (pos_body, [neg_body_1, ...])
        n_insession = 0
        n_crossuser = 0

        for ctx, positives, in_session_negs in feed_groups:
            for pos_cand in positives:
                neg_cands = []

                # Stage 1: in-session negatives
                if in_session_negs:
                    pool = list(in_session_negs)
                    rng.shuffle(pool)
                    neg_cands.extend([(ctx, c) for c in pool[:num_negatives]])

                n_is = len(neg_cands)
                n_insession += n_is

                # Stage 2: cross-user patch
                if len(neg_cands) < num_negatives:
                    need = num_negatives - len(neg_cands)
                    # Other users' clicks, using THIS user's context
                    pool = [(cx, c) for cx, c in all_positives
                            if cx.get("feed_id") != ctx.get("feed_id")]
                    if pool:
                        rng.shuffle(pool)
                        for _, cross_cand in pool[:need]:
                            # Patch: this user's context + other user's candidate
                            neg_cands.append((ctx, cross_cand))
                        n_crossuser += len(neg_cands) - n_is

                if not neg_cands:
                    continue

                neg_cands = neg_cands[:num_negatives]

                # Build prompts with budget
                pos_body = build_prompt_budgeted(
                    ctx, pos_cand, tokenizer, self._body_budget)
                neg_bodies = [
                    build_prompt_budgeted(nc_ctx, nc_cand, tokenizer, self._body_budget)
                    for nc_ctx, nc_cand in neg_cands
                ]
                self.pairs.append((pos_body, neg_bodies))

        avg_neg = np.mean([len(negs) for _, negs in self.pairs]) if self.pairs else 0
        print(f"[{_ts()}] PairwiseDataset: {len(self.pairs)} pairs, "
              f"avg {avg_neg:.1f} neg/pos "
              f"(in-session={n_insession}, cross-user={n_crossuser})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_body, neg_bodies = self.pairs[idx]

        pos_ids = encode_prompt(pos_body, self.tokenizer, self.max_len,
                                self.use_chat_template)
        neg_ids_list = [
            encode_prompt(nb, self.tokenizer, self.max_len, self.use_chat_template)
            for nb in neg_bodies
        ]

        # Left-pad all to same length within this group
        all_ids = [pos_ids] + neg_ids_list
        max_l = max(len(x) for x in all_ids)

        input_ids = []
        attention_masks = []
        pad_id = self.tokenizer.pad_token_id or 0

        for ids in all_ids:
            pad_len = max_l - len(ids)
            input_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


class PairwiseCollator:
    """Collate variable-sized pairwise groups into a flat batch."""

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, features):
        all_ids = []
        all_masks = []
        group_sizes = []

        for f in features:
            all_ids.append(f["input_ids"])
            all_masks.append(f["attention_mask"])
            group_sizes.append(f["input_ids"].shape[0])

        # Pad to max seq len across batch
        max_l = max(ids.shape[1] for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_l = max_l - ids.shape[1]
            if pad_l > 0:
                ids = F.pad(ids, (pad_l, 0), value=self.pad_id)
                mask = F.pad(mask, (pad_l, 0), value=0)
            padded_ids.append(ids)
            padded_masks.append(mask)

        return {
            "input_ids": torch.cat(padded_ids, dim=0),
            "attention_mask": torch.cat(padded_masks, dim=0),
            "group_sizes": group_sizes,
        }
