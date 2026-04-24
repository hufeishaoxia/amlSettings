"""Point-wise SFT trainer (Yes/No click classification).

Usage:
    torchrun --nproc_per_node N pointwise_sft/train.py \
        --base_model Qwen/Qwen3-8B-Instruct \
        --train_path data/train.jsonl \
        --eval_path  data/dev.jsonl \
        --output_dir output/pointwise
"""

import os
import random
import numpy as np
import torch
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments,
)
import fire

from data import PointwiseSFTDataset


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedCollator(DataCollatorForSeq2Seq):
    """DataCollatorForSeq2Seq that also stacks per-sample `weight`."""
    def __call__(self, features):
        weights = [f.pop("weight") for f in features]
        batch = super().__call__(features)
        batch["weight"] = torch.stack(weights)
        return batch


class WeightedTrainer(Trainer):
    """Trainer that scales per-sample LM loss by `batch["weight"]`."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)
        labels  = inputs["labels"]
        outputs = model(**inputs)
        logits  = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_tok = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        mask = (shift_labels != -100).float()
        per_sample = (per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if weights is not None:
            per_sample = per_sample * weights.to(per_sample.device)
        loss = per_sample.mean()
        return (loss, outputs) if return_outputs else loss


def train(
    base_model: str,
    train_path: str,
    eval_path: str,
    output_dir: str,
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
    seed: int = 42,
    use_chat_template: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
):
    set_seed(seed)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = max(1, batch_size // micro_batch_size // world_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    train_data = PointwiseSFTDataset(train_path, tokenizer, max_len=cutoff_len,
                                     max_history=max_history, use_chat_template=use_chat_template,
                                     sample=sample, seed=seed)
    val_data   = PointwiseSFTDataset(eval_path,  tokenizer, max_len=cutoff_len,
                                     max_history=max_history, use_chat_template=use_chat_template,
                                     sample=min(5000, max(1, len(train_data) // 10)),
                                     seed=seed)

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
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_run_name or None,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=WeightedCollator(tokenizer, pad_to_multiple_of=8,
                                       return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    out = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    fire.Fire(train)
