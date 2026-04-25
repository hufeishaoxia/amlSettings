"""Pairwise trainer with BCE + InfoNCE loss.

Bug fix 1: For left-padded sequences, last_pos = seq_len - 1 (the rightmost
position is always a real token since left-padding pushes pads to the left).
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from transformers import Trainer


@dataclass
class PairwiseCollator:
    """Collate variable-sized groups into a flat batch with group tracking.

    Each sample from PairwiseDataset has shape (1+K_i, L_i).
    We flatten all groups into (sum(1+K_i), max_L) and track group_sizes.
    """
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        group_sizes = []
        all_ids = []
        all_masks = []

        for f in features:
            gs = f["group_size"]
            group_sizes.append(gs)
            all_ids.append(f["input_ids"])      # (gs, L_i)
            all_masks.append(f["attention_mask"])

        # Pad to max seq length across all groups
        max_len = max(ids.shape[1] for ids in all_ids)
        padded_ids = []
        padded_masks = []

        for ids, masks in zip(all_ids, all_masks):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                # Left-pad
                padded_ids.append(torch.cat([
                    torch.full((ids.shape[0], pad_len), self.pad_token_id, dtype=torch.long),
                    ids,
                ], dim=1))
                padded_masks.append(torch.cat([
                    torch.zeros(ids.shape[0], pad_len, dtype=torch.long),
                    masks,
                ], dim=1))
            else:
                padded_ids.append(ids)
                padded_masks.append(masks)

        return {
            "input_ids": torch.cat(padded_ids, dim=0),          # (N_total, max_L)
            "attention_mask": torch.cat(padded_masks, dim=0),    # (N_total, max_L)
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
        }


class PairwiseTrainer(Trainer):
    """Trainer with pairwise BCE + InfoNCE loss.

    Args:
        yes_token_id: token id for "Yes"
        no_token_id: token id for "No"
        bce_weight: weight for BCE loss component
        infonce_weight: weight for InfoNCE loss component
        temperature: temperature for InfoNCE softmax
    """

    def __init__(self, *args, yes_token_id: int = None, no_token_id: int = None,
                 bce_weight: float = 1.0, infonce_weight: float = 1.0,
                 temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.bce_weight = bce_weight
        self.infonce_weight = infonce_weight
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]           # (N_total, L)
        attention_mask = inputs["attention_mask"]  # (N_total, L)
        group_sizes = inputs["group_sizes"]        # (B,)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (N_total, L, V)

        # Bug fix 1: For left-padded sequences, last real token is always at
        # position seq_len - 1 (rightmost), because padding is on the left.
        last_pos = logits.shape[1] - 1
        last_logits = logits[:, last_pos, :]  # (N_total, V)

        # Score = logit(Yes) - logit(No)
        yes_logits = last_logits[:, self.yes_token_id]  # (N_total,)
        no_logits = last_logits[:, self.no_token_id]    # (N_total,)
        scores = yes_logits - no_logits                  # (N_total,)

        # Split by groups
        total_bce = 0.0
        total_infonce = 0.0
        offset = 0
        n_groups = group_sizes.shape[0]

        for i in range(n_groups):
            gs = group_sizes[i].item()
            group_scores = scores[offset:offset + gs]  # (1+K,)
            pos_score = group_scores[0]                 # positive
            neg_scores = group_scores[1:]               # negatives

            # BCE loss: BCE(σ(pos), 1) + mean(BCE(σ(neg), 0))
            bce_pos = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score))
            bce_neg = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores))
            total_bce += bce_pos + bce_neg.mean()

            # InfoNCE: CE(softmax([pos, neg_1..K] / τ), target=0)
            all_scores = group_scores / self.temperature  # (1+K,)
            target = torch.zeros(1, dtype=torch.long, device=all_scores.device)
            total_infonce += F.cross_entropy(all_scores.unsqueeze(0), target)

            offset += gs

        loss = (self.bce_weight * total_bce + self.infonce_weight * total_infonce) / n_groups

        return (loss, outputs) if return_outputs else loss
