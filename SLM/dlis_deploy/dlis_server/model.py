"""
DLIS ModelImp for Qwen3-0.6B pointwise ranker.
Uses vLLM for accelerated inference with prefix KV cache.

Produces prompts that are byte-identical to the offline training/eval pipeline
(``SLM/data.py`` + ``SLM/eval_auc.py``). See ``prompt.py`` (vendored copy).

Determinism:
  vLLM with bf16 + dynamic batching + prefix cache + chunked prefill is
  NOT bit-reproducible: the same prompt can produce slightly different
  logits depending on (a) which other prompts share the prefill batch,
  (b) whether the prefix is cached, and (c) bf16 reduce ordering. For URA
  test we observed |Δscore| up to ~0.06 across runs (Pearson 0.998, AUC
  swing < 0.001). Enough for ranking, but not deterministic.

  Clients can request a deterministic-but-slower path by sending
  ``"deterministic": true`` in the JSON request. Default is fast (batched).
"""
import os
import json
import math
import time
import uuid
import logging
import threading
import utils

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompt import (
    SYSTEM_MSG,
    build_prompt,
    build_prompt_budgeted,
    normalize_request,
)

MODEL_VERSION = "v25-dlis-logprobs200"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global lock used only for deterministic-mode requests to keep them out of
# concurrent batches with regular requests. Regular requests do NOT take it.
_DET_LOCK = threading.Lock()


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

        # Number of top-K logprobs vLLM returns per generated token. We need
        # both ' Yes' (id=7414) and ' No' (id=2308) to ALWAYS be present, otherwise
        # the score code falls back to lp=-100 and softmax((-100,-100))=(0.5,0.5),
        # producing a tied score for ~4% of URA samples and a measurable AUC drop.
        # Vocab is ~151k tokens; 200 is comfortably enough for a binary head.
        self.logprobs_k = int(os.getenv("LOGPROBS_K", "200"))

        print(f"=== Qwen3-0.6B Ranker {MODEL_VERSION} ===")
        print(f"Loading vLLM engine from {ckpt_path} (tp={tp}, max_len={max_model_len}, dtype={dtype}, eager=True, max_logprobs={self.logprobs_k})")
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
            max_logprobs=self.logprobs_k,
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
        # Default fast sampling params (used by the optimized path).
        self.sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=self.logprobs_k)
        # Probe whether SamplingParams supports cache_salt (vLLM >= 0.6.6).
        self._sp_supports_cache_salt = self._probe_cache_salt_support()

        print(f"vLLM engine ready. Yes={self.yes_id}, No={self.no_id}, "
              f"cache_salt_supported={self._sp_supports_cache_salt}")
        print("model loaded.")

    def _probe_cache_salt_support(self) -> bool:
        try:
            SamplingParams(max_tokens=1, temperature=0, logprobs=self.logprobs_k, cache_salt="probe")
            return True
        except TypeError:
            return False

    def _det_sampling_params(self) -> SamplingParams:
        """SamplingParams with a unique cache_salt so the prefix cache
        ALWAYS misses. Falls back to the regular params on older vLLM."""
        if self._sp_supports_cache_salt:
            return SamplingParams(
                max_tokens=1, temperature=0, logprobs=self.logprobs_k,
                cache_salt=uuid.uuid4().hex,
            )
        return self.sampling_params

    def _score_prompts(self, prompts, sampling_params=None, one_at_a_time: bool = False):
        sp = sampling_params or self.sampling_params
        if one_at_a_time:
            scores = []
            for p in prompts:
                # One prompt per generate() so the per-call vLLM batch shape
                # is always (1,) -> reduce order for matmul/attention is
                # constant across runs -> bit-reproducible logits.
                outputs = self.llm.generate([p], sp)
                scores.extend(self._extract_yes_prob(o) for o in outputs)
            return scores
        outputs = self.llm.generate(prompts, sp)
        return [self._extract_yes_prob(o) for o in outputs]

    def _extract_yes_prob(self, out) -> float:
        logprobs_dict = out.outputs[0].logprobs[0]
        # Fallback for the (now rare) case where Yes/No fall outside top-K:
        # use the smallest observed logprob in the returned set as a tight
        # upper bound on the missing token's true logprob. This avoids the
        # old -100 sentinel that mapped both-missing -> exact 0.5.
        floor_lp = min(lp.logprob for lp in logprobs_dict.values())
        yes_lp = logprobs_dict[self.yes_id].logprob if self.yes_id in logprobs_dict else floor_lp
        no_lp = logprobs_dict[self.no_id].logprob if self.no_id in logprobs_dict else floor_lp
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
          "deterministic": true   -> bit-reproducible scoring (slower).
                                     Bypasses the prefix KV cache (unique
                                     cache_salt), processes one prompt per
                                     vLLM batch, and serializes against
                                     other deterministic requests with a
                                     global lock. Default is false.
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
            deterministic = bool(req.get("deterministic", False))

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
            if deterministic:
                with _DET_LOCK:
                    all_scores = self._score_prompts(
                        prompts,
                        sampling_params=self._det_sampling_params(),
                        one_at_a_time=True,
                    )
            else:
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
                "deterministic": deterministic,
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
