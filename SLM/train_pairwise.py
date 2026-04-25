"""Pairwise SFT trainer for click prediction.

For each positive (clicked) impression, constructs a list of negatives:
  1. In-session negatives: non-clicked impressions from the same feed
  2. Cross-user negatives: clicked items from OTHER feeds (patched as negatives)
Until we have `num_negatives` (default 20) negatives per positive.

Loss: BCE or InfoNCE over (pos, neg_1, ..., neg_K) using P(Yes) as the score.

After training, automatically evaluates each epoch checkpoint via eval_auc.py.

Usage:
    torchrun --nproc_per_node N train_pairwise.py \
        --base_model Qwen/Qwen3-0.6B \
        --data_path data \
        --output_dir output/pairwise
"""

import os
import random
import glob
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)
import fire

from data import (
    PointwiseSFTDataset, load_samples, load_samples_jsonl,
    build_prompt, build_prompt_budgeted,
)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Pairwise dataset
# ---------------------------------------------------------------------------

class PairwiseSFTDataset(torch.utils.data.Dataset):
    """Each item = (positive_prompt, [neg_prompt_1, ..., neg_prompt_K]).

    Negatives are sourced in two stages:
      1. In-session: non-clicked impressions from the SAME feed
      2. Cross-user patch: clicked items from OTHER feeds, treated as negatives
         (the intuition: another user's click is topically interesting but
          not aligned with THIS user's interests — a hard negative)
    Until we reach `num_negatives` per positive.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_len: int = 2048,
        max_history: int = 30,
        include_conv: bool = True,
        use_chat_template: bool = True,
        num_negatives: int = 20,
        max_rows: int = -1,
        flight_filter: str = "",
        require_features: bool = False,
        bizdate_min: str = "",
        bizdate_max: str = "",
        seed: int = 42,
        sample: int = -1,
        max_interests: int = 0,
        max_conv_groups: int = 0,
        max_msgs_per_group: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_chat_template = use_chat_template
        self.num_negatives = num_negatives
        self._body_budget = max(256, self.max_len - 120)
        self._fast_char_limit = int(2.5 * self._body_budget)

        # Load all samples
        if path.endswith(".jsonl"):
            all_samples = load_samples_jsonl(path)
        else:
            all_samples = load_samples(
                path,
                max_history=max_history,
                max_interests=max_interests,
                max_conv_groups=max_conv_groups,
                max_msgs_per_group=max_msgs_per_group,
                include_conv=include_conv,
                max_rows=max_rows,
                flight_filter=flight_filter,
                require_features=require_features,
                bizdate_min=bizdate_min,
                bizdate_max=bizdate_max,
            )

        # Group by feed_id
        feeds: dict = {}
        for s in all_samples:
            feeds.setdefault(s["feed_id"], []).append(s)

        # Build cross-user negative pool: all positives from all feeds
        cross_user_positives = [s for s in all_samples if s["label"] == 1]

        rng = random.Random(seed)

        # Build pairwise training instances
        self.pairs: List[dict] = []  # each: {pos_sample, neg_samples: [...]}
        n_insession_total = 0
        n_crossuser_total = 0

        for feed_id, feed_samples in feeds.items():
            positives = [s for s in feed_samples if s["label"] == 1]
            in_session_negs = [s for s in feed_samples if s["label"] == 0]

            for pos in positives:
                neg_list = []

                # Stage 1: in-session negatives (random subset)
                if in_session_negs:
                    insess = list(in_session_negs)
                    rng.shuffle(insess)
                    take = min(len(insess), num_negatives)
                    neg_list.extend(insess[:take])

                n_insession = len(neg_list)
                n_insession_total += n_insession

                # Stage 2: cross-user patch (other users' clicks as hard negatives)
                if len(neg_list) < num_negatives:
                    need = num_negatives - len(neg_list)
                    # Sample from cross-user positives, excluding same feed
                    candidates = [s for s in cross_user_positives
                                  if s["feed_id"] != feed_id]
                    if candidates:
                        rng.shuffle(candidates)
                        # Create patched negatives: keep pos's user context,
                        # but swap in the cross-user candidate
                        for cu in candidates[:need]:
                            patched = {
                                "feed_id": pos["feed_id"],
                                "user_id": pos["user_id"],
                                "bizdate": pos["bizdate"],
                                "flight_ids": pos["flight_ids"],
                                "history": pos["history"],
                                "interests": pos["interests"],
                                "candidate": cu["candidate"],  # other user's clicked item
                                "features": cu.get("features"),
                                "label": 0,  # treat as negative for THIS user
                            }
                            neg_list.append(patched)

                n_crossuser_total += len(neg_list) - n_insession

                if neg_list:
                    # Truncate to exactly num_negatives
                    neg_list = neg_list[:num_negatives]
                    self.pairs.append({"pos": pos, "negs": neg_list})

        if sample > 0 and sample < len(self.pairs):
            rng.shuffle(self.pairs)
            self.pairs = self.pairs[:sample]

        n_pos = len(self.pairs)
        avg_neg = np.mean([len(p["negs"]) for p in self.pairs]) if self.pairs else 0
        print(f"[{_ts()}] PairwiseDataset: {n_pos} positive anchors, "
              f"avg {avg_neg:.1f} negatives/pos "
              f"(in-session: {n_insession_total}, cross-user: {n_crossuser_total})")

        # Pre-build prompts
        for pair in self.pairs:
            pair["pos"]["_prompt"] = self._build_body(pair["pos"])
            for neg in pair["negs"]:
                neg["_prompt"] = self._build_body(neg)

    def _build_body(self, s: dict) -> str:
        """Build the prompt body (without chat template wrapping)."""
        full_text = build_prompt(s["history"], s["interests"], s["candidate"])
        if len(full_text) <= self._fast_char_limit:
            return full_text
        text, _, _, _ = build_prompt_budgeted(
            s["history"], s["interests"], s["candidate"],
            self.tokenizer, self._body_budget,
        )
        return text

    def _encode_prompt(self, body: str) -> List[int]:
        """Encode prompt body → input_ids (with chat template if enabled)."""
        if self.use_chat_template:
            messages = [
                {"role": "system", "content":
                    "I am a recommendation assistant. I read the user's interests, recent "
                    "conversations, and shown cards, then predict whether they will click "
                    "the candidate item. I answer Yes or No."},
                {"role": "user", "content": body},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = body
        return self.tokenizer.encode(formatted, max_length=self.max_len,
                                     truncation=True, add_special_tokens=False)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        pos_ids = self._encode_prompt(pair["pos"]["_prompt"])
        neg_ids_list = [self._encode_prompt(n["_prompt"]) for n in pair["negs"]]

        # Pad all to same length
        all_ids = [pos_ids] + neg_ids_list
        max_l = min(self.max_len, max(len(x) for x in all_ids))

        input_ids = []
        attention_masks = []
        for ids in all_ids:
            ids = ids[-max_l:]  # truncate from left (keep recent context)
            pad_len = max_l - len(ids)
            input_ids.append([self.tokenizer.pad_token_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),          # (1+K, L)
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),  # (1+K, L)
            "num_negatives": len(neg_ids_list),
        }


# ---------------------------------------------------------------------------
# Pairwise collator
# ---------------------------------------------------------------------------

class PairwiseCollator:
    """Collates variable-length pairwise batches.

    Each sample has shape (1+K_i, L_i) — we flatten into a single (N_total, L_max)
    batch and return offsets so the trainer can reconstruct groups.
    """
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or 0

    def __call__(self, features):
        all_ids = []
        all_masks = []
        group_sizes = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            all_ids.append(ids)
            all_masks.append(mask)
            group_sizes.append(ids.shape[0])

        # Pad to max seq len across entire batch
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


# ---------------------------------------------------------------------------
# Pairwise Trainer
# ---------------------------------------------------------------------------

class PairwiseTrainer(Trainer):
    """Trainer with BCE or InfoNCE loss over (pos, neg_1, ..., neg_K) groups."""

    def __init__(self, *args, loss_type: str = "bce", temperature: float = 1.0,
                 yes_id: int = 0, no_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.temperature = temperature
        self.yes_id = yes_id
        self.no_id = no_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_sizes = inputs.pop("group_sizes")
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits

        # Get logit at last non-pad position for Yes and No tokens
        last_pos = attn_mask.sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.size(0)), last_pos]

        # Score = logit(Yes) - logit(No)
        scores = last_logits[:, self.yes_id] - last_logits[:, self.no_id]

        # Split scores by group
        losses = []
        offset = 0
        for gs in group_sizes:
            group_scores = scores[offset:offset + gs]
            pos_score = group_scores[0]
            neg_scores = group_scores[1:]

            if self.loss_type == "infonce":
                all_scores = group_scores / self.temperature
                loss = F.cross_entropy(all_scores.unsqueeze(0),
                                       torch.zeros(1, dtype=torch.long,
                                                   device=all_scores.device))
            else:
                # BCE: pos → 1, each neg → 0
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_score.unsqueeze(0),
                    torch.ones(1, device=pos_score.device))
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_scores,
                    torch.zeros_like(neg_scores))
                loss = pos_loss + neg_loss.mean()

            losses.append(loss)
            offset += gs

        total_loss = torch.stack(losses).mean()
        return (total_loss, outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# Post-training eval helper
# ---------------------------------------------------------------------------

def _run_eval_on_checkpoints(
    output_dir: str,
    data_path: str,
    eval_from: str,
    ura_flight: str,
    max_history: int,
    cutoff_len: int,
    use_chat_template: bool,
    include_conv: int,
    eval_max_rows: int,
    eval_ura_jsonl: str,
    eval_all_jsonl: str,
    ura_only: int,
):
    """After training, find all epoch checkpoints and run eval_auc.py on each."""
    # Find checkpoint dirs: checkpoint-<step> or epoch_<n>
    ckpt_dirs = []

    # HF Trainer saves as checkpoint-<global_step>
    for d in sorted(glob.glob(os.path.join(output_dir, "checkpoint-*"))):
        if os.path.isdir(d):
            ckpt_dirs.append(d)

    # Also check final_checkpoint
    final = os.path.join(output_dir, "final_checkpoint")
    if os.path.isdir(final):
        ckpt_dirs.append(final)

    if not ckpt_dirs:
        print(f"[{_ts()}] No checkpoints found in {output_dir}")
        return

    # Determine number of GPUs
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1

    for ckpt in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt)
        out_json = os.path.join(output_dir, f"eval_{ckpt_name}.json")
        print(f"\n[{_ts()}] === Evaluating {ckpt_name} ===")

        cmd = []
        if ngpu > 1:
            cmd += ["torchrun", "--nproc_per_node", str(ngpu)]
        else:
            cmd += [sys.executable]
        cmd += [
            "eval_auc.py",
            "--ckpt", ckpt,
            "--data_path", data_path,
            "--eval_from", str(eval_from),
            "--ura_flight", ura_flight,
            "--max_history", str(max_history),
            "--max_len", str(cutoff_len),
            "--batch_size", "8",
            "--use_chat_template", str(use_chat_template),
            "--include_conv", str(include_conv),
            "--out_json", out_json,
        ]
        if eval_max_rows > 0:
            cmd += ["--eval_max_rows", str(eval_max_rows)]
        if eval_ura_jsonl:
            cmd += ["--eval_ura_jsonl", eval_ura_jsonl]
        if eval_all_jsonl:
            cmd += ["--eval_all_jsonl", eval_all_jsonl]
        if ura_only:
            cmd += ["--ura_only", "1"]

        print(f"[{_ts()}] CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"[{_ts()}] WARNING: eval failed for {ckpt_name} (exit={result.returncode})")
        else:
            print(f"[{_ts()}] Eval done for {ckpt_name} → {out_json}")

    # Print summary table
    print(f"\n[{_ts()}] === Eval Summary ===")
    print(f"{'checkpoint':<30} {'split':>5} {'AUC':>8} {'n':>8} {'ctr':>7}")
    import json as _json
    for ckpt in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt)
        out_json = os.path.join(output_dir, f"eval_{ckpt_name}.json")
        if not os.path.exists(out_json):
            print(f"{ckpt_name:<30} (no results)")
            continue
        try:
            with open(out_json) as f:
                results = _json.load(f)
            for r in results:
                print(f"{ckpt_name:<30} {r['split']:>5} {r['auc']:>8.4f} {r['n']:>8d} {r['ctr']:>7.4f}")
        except Exception as e:
            print(f"{ckpt_name:<30} (error reading: {e})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train(
    base_model: str,
    data_path: str,
    output_dir: str,
    train_until: str = "20260416",
    train_bizdate_min: str = "",
    eval_from: str = "20260417",
    ura_flight: str = "discover-rk-ura",
    train_ura_only: int = 0,
    max_history: int = 30,
    cutoff_len: int = 2048,
    batch_size: int = 32,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    num_negatives: int = 20,
    loss_type: str = "bce",       # "bce" or "infonce"
    temperature: float = 1.0,     # for infonce
    sample: int = -1,
    max_rows: int = -1,
    eval_max_rows: int = -1,
    eval_batch_size: int = 8,
    optim: str = "adamw_torch",
    seed: int = 42,
    use_chat_template: bool = True,
    include_conv: int = 1,
    train_jsonl: str = "",
    eval_ura_jsonl: str = "",
    eval_all_jsonl: str = "",
    eval_ura_only: int = 1,       # 1 = only URA eval; 0 = URA + ALL
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    grad_accum = max(1, batch_size // micro_batch_size // world_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    # Resolve Yes/No token IDs
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    if not yes_ids or not no_ids:
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = tokenizer.encode("No", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]
    if rank == 0:
        print(f"[{_ts()}] Yes token id={yes_id}, No token id={no_id}")

    _include_conv = int(include_conv) > 0
    _train_path = train_jsonl if train_jsonl else data_path

    # Training dataset
    train_data = PairwiseSFTDataset(
        _train_path, tokenizer,
        max_len=cutoff_len, max_history=max_history,
        include_conv=_include_conv,
        use_chat_template=use_chat_template,
        num_negatives=num_negatives,
        max_rows=max_rows,
        flight_filter=ura_flight if int(train_ura_only) else "",
        bizdate_min=train_bizdate_min,
        bizdate_max=train_until,
        seed=seed, sample=sample,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        optim=optim,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=int(num_epochs) + 1,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name or None,
        remove_unused_columns=False,
    )

    trainer = PairwiseTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        data_collator=PairwiseCollator(tokenizer),
        loss_type=loss_type,
        temperature=temperature,
        yes_id=yes_id,
        no_id=no_id,
    )

    if rank == 0:
        print(f"[{_ts()}] Starting pairwise training: {len(train_data)} pairs, "
              f"loss={loss_type}, epochs={num_epochs}, batch={batch_size}, "
              f"micro_bs={micro_batch_size}, grad_accum={grad_accum}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final checkpoint
    out = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if rank == 0:
        print(f"[{_ts()}] Saved final checkpoint to {out}")

    # Distributed cleanup before eval
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    # Only rank 0 runs post-training eval
    if rank == 0:
        print(f"\n[{_ts()}] === Post-training evaluation ===")
        _run_eval_on_checkpoints(
            output_dir=output_dir,
            data_path=data_path,
            eval_from=eval_from,
            ura_flight=ura_flight,
            max_history=max_history,
            cutoff_len=cutoff_len,
            use_chat_template=use_chat_template,
            include_conv=include_conv,
            eval_max_rows=eval_max_rows,
            eval_ura_jsonl=eval_ura_jsonl,
            eval_all_jsonl=eval_all_jsonl,
            ura_only=int(eval_ura_only),
        )


if __name__ == "__main__":
    fire.Fire(train)
