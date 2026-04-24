"""Point-wise SFT v2 — replace the full-vocab LM head with a 2-class head.

Why v2:
    The base model's lm_head projects hidden_size -> |vocab| (~152k for Qwen3),
    but the only labels we ever supervise are " Yes" / " No" (one token each).
    The full-vocab matmul + softmax + gradient is ~5-10 GB of useless work.

What v2 does:
    1. Drop `lm_head`. Keep only the transformer backbone.
    2. Add a 2-class linear head, warm-started from the rows of the original
       lm_head at (no_token_id, yes_token_id) so logits match the SFT init.
    3. Forward returns ONLY hidden states; we index active positions
       (label != -100), apply the tiny head, and CE over 2 classes.

Drop-in replacement for train.py — same CLI, same dataset/data.py.

Saved checkpoint layout:
    <ckpt>/                ← HF backbone (use AutoModel.from_pretrained)
    <ckpt>/binary_head.pt  ← {"weight": (2, H), "no_id": int, "yes_id": int}
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
import fire

from data import PointwiseSFTDataset


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Binary-head wrapper: backbone -> last_hidden_state -> 2-class linear
# ---------------------------------------------------------------------------

class BinaryHeadModel(nn.Module):
    """Wraps a HF causal LM, dropping its lm_head in favor of a 2-class head."""

    def __init__(self, base_model, no_token_id: int, yes_token_id: int):
        super().__init__()
        # Keep only the transformer backbone (Qwen / Llama: `model`; Mistral too).
        # GPT-NeoX uses `gpt_neox`; fall back gracefully.
        backbone = getattr(base_model, "model", None) or getattr(base_model, "transformer", None)
        if backbone is None:
            raise AttributeError(
                f"Cannot find transformer backbone on {type(base_model).__name__}; "
                "expected `.model` or `.transformer`."
            )
        self.backbone = backbone
        self.config = base_model.config
        self.no_token_id = int(no_token_id)
        self.yes_token_id = int(yes_token_id)

        hidden = base_model.config.hidden_size
        self.head = nn.Linear(hidden, 2, bias=False)
        # Warm-start head rows from the original LM head (row i -> class i).
        with torch.no_grad():
            w = base_model.get_output_embeddings().weight  # (V, H)
            init = torch.stack([w[self.no_token_id], w[self.yes_token_id]], dim=0)
            self.head.weight.copy_(init.to(self.head.weight.dtype))

        # Free the original lm_head (now unreferenced through `base_model`).
        del base_model

    # Trainer / gradient_checkpointing_enable() compatibility shims --------
    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kw)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.backbone, "is_gradient_checkpointing", False)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        # Strip kwargs the backbone doesn't expect from our pipeline.
        kw.pop("labels", None)
        kw.pop("weight", None)
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **kw,
        )
        # `out.last_hidden_state` for HF backbones.
        return out  # caller reads .last_hidden_state


# ---------------------------------------------------------------------------
# Collator + Trainer (binary head, weighted, memory-efficient)
# ---------------------------------------------------------------------------

class WeightedCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        weights = [f.pop("weight") for f in features]
        batch = super().__call__(features)
        batch["weight"] = torch.stack(weights)
        return batch


class BinaryHeadTrainer(Trainer):
    """Compute loss on a 2-class head, only at active (label != -100) positions."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)
        labels = inputs.pop("labels")  # (B, T) token-ids; -100 elsewhere

        # Forward through backbone only — no full-vocab projection.
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (B, T, H)

        # Next-token shift: predict labels[:, 1:] from hidden[:, :-1, :].
        shift_labels = labels[..., 1:].contiguous()        # (B, T-1)
        shift_hidden = hidden[:, :-1, :].contiguous()      # (B, T-1, H)
        active_mask = shift_labels != -100

        batch_idx, seq_idx = active_mask.nonzero(as_tuple=True)
        if batch_idx.numel() == 0:
            # No supervised positions — degenerate batch; return 0 grad.
            zero = hidden.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        active_hidden = shift_hidden[batch_idx, seq_idx]   # (N_active, H)

        # Resolve head: under DDP, model is wrapped — strip to inner module.
        head_owner = getattr(model, "module", model)
        active_logits = head_owner.head(active_hidden)     # (N_active, 2)

        # Map token-id labels {no_id, yes_id} -> {0, 1}.
        active_token = shift_labels[batch_idx, seq_idx]
        no_id = head_owner.no_token_id
        yes_id = head_owner.yes_token_id
        bin_labels = torch.where(
            active_token == yes_id,
            torch.ones_like(active_token),
            torch.zeros_like(active_token),
        )
        # Sanity: any active label that is neither yes nor no would be miscounted.
        # In our dataset it's always one of the two (single-token answer).

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        active_loss = loss_fct(active_logits, bin_labels)  # (N_active,)

        # Reduce per-sample then apply class weights.
        B = shift_labels.size(0)
        per_sample = torch.zeros(B, device=hidden.device, dtype=active_loss.dtype)
        counts = torch.zeros(B, device=hidden.device, dtype=active_loss.dtype)
        per_sample.scatter_add_(0, batch_idx, active_loss)
        counts.scatter_add_(0, batch_idx, torch.ones_like(active_loss))
        per_sample = per_sample / counts.clamp(min=1)

        if weights is not None:
            per_sample = per_sample * weights.to(per_sample.device)
        loss = per_sample.mean()
        return (loss, outputs) if return_outputs else loss

    # Override save: HF backbone via save_pretrained + binary head as a .pt file.
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        m = self.model.module if hasattr(self.model, "module") else self.model
        m.backbone.save_pretrained(output_dir, safe_serialization=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(
            {
                "weight": m.head.weight.detach().cpu(),
                "no_token_id": m.no_token_id,
                "yes_token_id": m.yes_token_id,
            },
            os.path.join(output_dir, "binary_head.pt"),
        )
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


# ---------------------------------------------------------------------------
# Entry point — same CLI as train.py
# ---------------------------------------------------------------------------

def train(
    base_model: str,
    data_path: str,
    output_dir: str,
    train_until: str = "20260416",
    eval_from: str = "20260417",
    ura_flight: str = "discover-rk-ura",
    train_ura_only: int = 0,
    max_history: int = 30,
    cutoff_len: int = 2048,
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    eval_steps: int = 256,
    save_steps: int = 512,
    early_stopping_patience: int = 8,
    sample: int = -1,
    eval_sample: int = -1,
    max_rows: int = -1,
    eval_max_rows: int = -1,
    optim: str = "adamw_torch",
    seed: int = 42,
    use_chat_template: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
):
    set_seed(seed)
    train_ura_only = int(train_ura_only) > 0
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, batch_size // micro_batch_size // world_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # The dataset emits target " Yes" / " No" (with a leading space) so we
    # must look up the SAME token IDs the dataset produces.
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(
            f"Tokenizer splits ' Yes'/' No' into multiple tokens: "
            f"yes={yes_ids} no={no_ids}. Pick a tokenizer where each is one token."
        )
    yes_token_id, no_token_id = yes_ids[0], no_ids[0]
    print(f"[v2] binary head: no_id={no_token_id} ({tokenizer.decode([no_token_id])!r}) "
          f"yes_id={yes_token_id} ({tokenizer.decode([yes_token_id])!r})")

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    model = BinaryHeadModel(base, no_token_id=no_token_id, yes_token_id=yes_token_id)
    model.gradient_checkpointing_enable()

    train_data = PointwiseSFTDataset(
        data_path, tokenizer,
        max_len=cutoff_len, max_history=max_history,
        use_chat_template=use_chat_template,
        sample=sample, seed=seed,
        max_rows=max_rows,
        bizdate_max=train_until,
        flight_filter=ura_flight if train_ura_only else "",
    )
    val_ura = PointwiseSFTDataset(
        data_path, tokenizer,
        max_len=cutoff_len, max_history=max_history,
        use_chat_template=use_chat_template,
        sample=eval_sample, seed=seed,
        max_rows=eval_max_rows,
        bizdate_min=eval_from,
        flight_filter=ura_flight,
        require_features=True,
    )
    val_all = PointwiseSFTDataset(
        data_path, tokenizer,
        max_len=cutoff_len, max_history=max_history,
        use_chat_template=use_chat_template,
        sample=eval_sample, seed=seed,
        max_rows=eval_max_rows,
        bizdate_min=eval_from,
        require_features=True,
    )
    eval_dataset = {"ura": val_ura, "all": val_all}

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        optim=optim,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(num_epochs) + 1,
        # NOTE: load_best_model_at_end is OFF for v2.
        # Trainer's reloader expects an HF model on disk; our checkpoint is
        # backbone + binary_head.pt. Pick the best ckpt manually after training.
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        gradient_checkpointing=True,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name or None,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    trainer = BinaryHeadTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_dataset,
        data_collator=WeightedCollator(tokenizer, pad_to_multiple_of=8,
                                       return_tensors="pt", padding=True),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final = os.path.join(output_dir, "final_checkpoint")
    trainer._save(final)
    print(f"Saved to {final}")


if __name__ == "__main__":
    fire.Fire(train)
