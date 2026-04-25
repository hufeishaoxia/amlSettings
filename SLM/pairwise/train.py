"""Pairwise SFT training entry point.

No eval during training, no early stopping.
Saves checkpoint every epoch, then auto-evals each checkpoint.

Usage:
    torchrun --nproc_per_node 8 train.py \
        --base_model Qwen/Qwen3-0.6B \
        --train_jsonl pairwise_train.jsonl \
        --eval_jsonl pairwise_eval.jsonl \
        --output_dir output/pairwise_bce
"""

import glob
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
)
import fire

from dataset import PairwiseDataset, PairwiseCollator
from trainer import PairwiseTrainer
from prompt import get_yes_no_ids


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_eval_on_checkpoints(output_dir, eval_jsonl, max_len, use_chat_template):
    """After training, eval each checkpoint via eval_auc.py."""
    ckpt_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    final = os.path.join(output_dir, "final_checkpoint")
    if os.path.isdir(final):
        ckpt_dirs.append(final)

    if not ckpt_dirs:
        print(f"[{_ts()}] No checkpoints found in {output_dir}")
        return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for ckpt in ckpt_dirs:
        name = os.path.basename(ckpt)
        out_json = os.path.join(output_dir, f"eval_{name}.json")
        print(f"\n[{_ts()}] === Evaluating {name} ===")

        cmd = []
        if ngpu > 1:
            cmd += ["torchrun", "--nproc_per_node", str(ngpu)]
        else:
            cmd += [sys.executable]
        cmd += [
            os.path.join(script_dir, "eval_auc.py"),
            "--ckpt", ckpt,
            "--eval_jsonl", eval_jsonl,
            "--max_len", str(max_len),
            "--use_chat_template", str(use_chat_template),
            "--out_json", out_json,
        ]

        print(f"[{_ts()}] CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=script_dir)
        if result.returncode != 0:
            print(f"[{_ts()}] WARNING: eval failed for {name} (exit={result.returncode})")

    # Summary table
    print(f"\n[{_ts()}] === Eval Summary ===")
    print(f"{'checkpoint':<30} {'split':>5} {'AUC':>8} {'n':>8} {'ctr':>7}")
    for ckpt in ckpt_dirs:
        name = os.path.basename(ckpt)
        out_json = os.path.join(output_dir, f"eval_{name}.json")
        if not os.path.exists(out_json):
            print(f"{name:<30} (no results)")
            continue
        try:
            with open(out_json) as f:
                results = json.load(f)
            for r in results:
                print(f"{name:<30} {r['split']:>5} {r['auc']:>8.4f} "
                      f"{r['n']:>8d} {r['ctr']:>7.4f}")
        except Exception as e:
            print(f"{name:<30} (error: {e})")


def train(
    base_model: str,
    train_jsonl: str,
    eval_jsonl: str,
    output_dir: str,
    max_len: int = 2048,
    batch_size: int = 32,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    num_negatives: int = 20,
    loss_type: str = "bce",
    temperature: float = 1.0,
    optim: str = "adamw_torch",
    seed: int = 42,
    use_chat_template: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    grad_accum = max(1, batch_size // micro_batch_size // world_size)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    yes_id, no_id = get_yes_no_ids(tokenizer)
    if rank == 0:
        print(f"[{_ts()}] Yes={yes_id} No={no_id}")
        print(f"[{_ts()}] loss={loss_type} neg={num_negatives} epochs={num_epochs} "
              f"batch={batch_size} micro={micro_batch_size} grad_accum={grad_accum}")

    train_data = PairwiseDataset(
        train_jsonl, tokenizer, max_len=max_len,
        use_chat_template=use_chat_template,
        num_negatives=num_negatives, seed=seed,
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
        save_total_limit=num_epochs + 1,
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
        data_collator=PairwiseCollator(pad_id=tokenizer.pad_token_id or 0),
        loss_type=loss_type,
        temperature=temperature,
        yes_id=yes_id,
        no_id=no_id,
    )

    if rank == 0:
        print(f"[{_ts()}] Starting training: {len(train_data)} pairs")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final
    out = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if rank == 0:
        print(f"[{_ts()}] Saved final → {out}")

    # Cleanup distributed
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    # Post-training eval (rank 0 only)
    if rank == 0:
        print(f"\n[{_ts()}] === Post-training evaluation ===")
        _run_eval_on_checkpoints(output_dir, eval_jsonl, max_len, use_chat_template)


if __name__ == "__main__":
    fire.Fire(train)
