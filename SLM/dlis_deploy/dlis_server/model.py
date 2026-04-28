"""
DLIS ModelImp for Qwen3-0.6B pointwise ranker.
Uses vLLM for accelerated inference with prefix KV cache.

Produces prompts that are byte-identical to the offline training/eval pipeline
(``SLM/data.py`` + ``SLM/eval_auc.py``). See ``prompt.py`` (vendored copy).

Scoring is mathematically identical to the offline scorer: a vLLM logits
processor masks every vocab id except ' Yes' / ' No' to -inf before sampling,
so the returned logprobs are exactly ``softmax(logits[[yes_id, no_id]])``.
"""
import os
import json
import math
import time
import logging
import utils

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompt import (
    SYSTEM_MSG,
    build_prompt,
    build_prompt_budgeted,
    normalize_request,
)

MODEL_VERSION = "v26-dlis"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_yesno_mask_processor(yes_id: int, no_id: int):
    """Return a vLLM logits processor that masks every vocab id except
    ``yes_id`` and ``no_id`` to ``-inf``. After this mask, the next-token
    softmax is computed over a 2-element distribution -- mathematically
    identical to the offline scorer's ``softmax(logits[[yes_id, no_id]])``.

    The processor is stateless and safe to share across requests / threads.
    Signature matches vLLM's ``LogitsProcessor`` ABC: ``(token_ids, logits) ->
    logits`` where ``logits`` is a 1-D float tensor of shape ``(vocab_size,)``.
    """
    import torch  # local import: vLLM already pulls torch into the image

    def _proc(_token_ids, logits):
        # In-place mask is fine -- vLLM passes a fresh tensor per step.
        keep = torch.full_like(logits, float("-inf"))
        keep[yes_id] = logits[yes_id]
        keep[no_id] = logits[no_id]
        return keep

    return _proc


class ModelImp:
    def __init__(self):
        self.model_path = utils.get_model_path()
        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(self.model_path)), "model")
        self.data_path = utils.get_data_path()
        self.data_dir = None
        if self.data_path is not None:
            self.data_dir = os.path.dirname(os.path.realpath(self.data_path))
        self.initial_dyanmic_data_paths = utils.get_initial_dynamic_data_paths()
        self.initial_dynamic_data_dirs = utils.get_named_directories(self.initial_dyanmic_data_paths)

        print(f"Model Path: {self.model_path}")
        print(f"Model Dir: {self.model_dir}")
        print(f"Data Path: {self.data_path}")
        print(f"Data Dir: {self.data_dir}")

        # Model weights path - check env override or default to /qwen3_model
        ckpt_path = os.getenv("QWEN3_MODEL_PATH", "/qwen3_model")
        tp = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
        gpu_mem = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
        dtype = os.getenv("VLLM_DTYPE", "bfloat16")
        # Token budget for the prompt body (excludes ~120 tokens reserved for
        # the chat-template wrapper). Keep this aligned with the offline
        # eval setting so AUC matches; SLM/eval_auc.py defaults to max_len=2048.
        self.eval_max_len = int(os.getenv("EVAL_MAX_LEN", "2048"))
        self.body_budget = max(256, self.eval_max_len - 120)

        print(f"=== Qwen3-0.6B Ranker {MODEL_VERSION} ===")
        print(f"Loading vLLM engine from {ckpt_path} (tp={tp}, max_len={max_model_len}, dtype={dtype}, eager=True)")
        self.llm = LLM(
            model=ckpt_path,
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=tp,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        yes_ids = self.tokenizer.encode(" Yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" No", add_special_tokens=False)
        if not yes_ids or not no_ids:
            yes_ids = self.tokenizer.encode("Yes", add_special_tokens=False)
            no_ids = self.tokenizer.encode("No", add_special_tokens=False)
        self.yes_id = yes_ids[0]
        self.no_id = no_ids[0]

        # Restrict the next-token distribution to {Yes, No} via a logits
        # processor that sets every other vocab entry to -inf. After this mask,
        # vLLM's softmax produces exactly P(Yes) / (P(Yes) + P(No)) -- the same
        # quantity the offline eval (eval_auc.py) computes via
        #     softmax(logits[:, -1, [yes_id, no_id]])[:, 0]
        # Because the post-mask distribution has only 2 non-zero entries, the
        # top-2 logprobs vLLM returns are guaranteed to be Yes and No, so we
        # only need logprobs=2 (no top-K truncation hazard, no fallback).
        self._yesno_processor = _make_yesno_mask_processor(self.yes_id, self.no_id)
        self.sampling_params = SamplingParams(
            max_tokens=1, temperature=0, logprobs=2,
            logits_processors=[self._yesno_processor],
        )

        print(f"vLLM engine ready. Yes={self.yes_id}, No={self.no_id}")
        print("model loaded.")

    def _score_prompts(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [self._extract_yes_prob(o) for o in outputs]

    def _extract_yes_prob(self, out) -> float:
        # After the yes/no mask, vLLM's logprobs at this position are computed
        # over a 2-element distribution -> Yes is guaranteed present. Both lp
        # values are normalized so exp(yes_lp) + exp(no_lp) == 1, but we keep
        # the explicit two-way softmax for safety / numerical hygiene.
        logprobs_dict = out.outputs[0].logprobs[0]
        yes_lp = logprobs_dict[self.yes_id].logprob
        no_lp = logprobs_dict[self.no_id].logprob
        max_lp = max(yes_lp, no_lp)
        return math.exp(yes_lp - max_lp) / (math.exp(yes_lp - max_lp) + math.exp(no_lp - max_lp))

    def Eval(self, data):
        """DLIS string eval interface. JSON in, JSON out.

        Accepts both the new rich schema (mirrors training JSONL):
            {"interests": {"positive":[...], "negative":[...],
                            "interactions": {...}, "conversations":[...]},
             "history":   [{"title": str, "summary": str}],
             "candidates":[{"id": str, "title": str, "summary": str}]}

        and the legacy flat schema (interests=list[dict], shownTitles=...).
        Prompt format is identical to ``SLM/data.py:build_prompt`` so the
        deployed AUC matches the offline eval AUC for the same checkpoint.

        Optional flags:
          "max_len":      int     -> per-request prompt body budget.
        """
        try:
            req = json.loads(data)
        except Exception as e:
            logger.error(f"Invalid JSON: {e}")
            return json.dumps({"error": f"Invalid JSON: {e}"})

        try:
            t0 = time.time()
            history, interests, candidates = normalize_request(req)
            req_max_len = int(req.get("max_len") or self.eval_max_len)
            body_budget = max(256, req_max_len - 120)

            prompts = []
            for card in candidates:
                cand = {
                    "title": card.get("title", ""),
                    "summary": card.get("summary", ""),
                }
                body, _truncated, _dh, _dc = build_prompt_budgeted(
                    history, interests, cand, self.tokenizer, body_budget,
                )
                msgs = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": body},
                ]
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                prompts.append(text)

            t1 = time.time()
            all_scores = self._score_prompts(prompts)
            t2 = time.time()

            prompt_ms = (t1 - t0) * 1000
            infer_ms = (t2 - t1) * 1000
            total_ms = (t2 - t0) * 1000

            results = []
            for card, score in zip(candidates, all_scores):
                results.append({
                    "id": card.get("id", ""),
                    "score": round(score, 6),
                })

            response = {
                "scores": results,
                "latency_ms": round(total_ms, 1),
                "prompt_build_ms": round(prompt_ms, 1),
                "inference_ms": round(infer_ms, 1),
                "model_version": MODEL_VERSION,
            }
            return json.dumps(response)

        except Exception as e:
            logger.exception(f"Eval error: {e}")
            return json.dumps({"error": str(e)})

    def EvalBatch(self, data_list):
        return [self.Eval(d) for d in data_list]

    def EvalBinary(self, data):
        return data

    def EvalBatchBinary(self, data_list):
        return data_list

    def OnDataUpdate(self, updated_paths):
        print("Got a fresh set of updated data")
        updated_dirs = utils.get_named_directories(updated_paths)
        if updated_dirs:
            for namedpath in updated_dirs:
                print(f"Updated data labeled {namedpath.name} is in {namedpath.path}")
