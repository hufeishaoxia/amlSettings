"""
DLIS ModelImp for Qwen3-0.6B pointwise ranker.
Uses vLLM for accelerated inference with prefix KV cache.
"""
import os
import json
import math
import time
import logging
import utils

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_VERSION = "v19-dlis"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_MSG = (
    "I am a recommendation assistant. I read the user's interests, recent "
    "conversations, and shown cards, then predict whether they will click "
    "the candidate item. I answer Yes or No."
)


def _format_interest(it: dict) -> str:
    name = (it.get("name") or "").strip()
    if not name:
        return ""
    parts = [name]
    meta = []
    for key in ("domain", "classification", "status", "intent"):
        v = it.get(key)
        if v not in (None, "", []):
            meta.append(f"{key}={v}")
    s = it.get("strength")
    if s is not None:
        try:
            meta.append(f"strength={float(s):.2f}")
        except Exception:
            pass
    if meta:
        parts[0] += "  [" + "; ".join(meta) + "]"
    return "\n".join(parts)


def build_inference_prompt(
    interests, shown_titles, conversations, candidate_title,
    candidate_summary="", candidate_matched_interest="",
    max_interests=30, max_shown=30, max_conv_groups=5, max_msgs_per_group=6,
):
    sections = []
    if interests:
        sorted_ints = sorted(interests, key=lambda x: -float(x.get("strength") or 0))
        formatted = [_format_interest(it) for it in sorted_ints[:max_interests]]
        formatted = [f for f in formatted if f]
        if formatted:
            sections.append("USER_INTERESTS:\n" + "\n".join(f"- {f}" for f in formatted))
    if shown_titles:
        sections.append("SHOWN_CARDS:\n" + "\n".join(f"- {t}" for t in shown_titles[:max_shown]))
    if conversations:
        conv_lines = []
        for g in conversations[:max_conv_groups]:
            msgs = g.get("messages", [])[-max_msgs_per_group:]
            for m in msgs:
                author = m.get("author", "?")
                text = m.get("text", "").strip().replace("\n", " ")
                if len(text) > 220:
                    text = text[:219] + "..."
                conv_lines.append(f"  [{author}] {text}")
        if conv_lines:
            sections.append("RECENT_CONVERSATIONS:\n" + "\n".join(conv_lines))
    cand_parts = [f"Title: {candidate_title}"]
    if candidate_summary:
        cand_parts.append(f"Summary: {candidate_summary}")
    if candidate_matched_interest:
        cand_parts.append(f"Matched Interest: {candidate_matched_interest}")
    sections.append("CANDIDATE:\n" + "\n".join(cand_parts))
    sections.append("Will the user click this candidate? Answer Yes or No.")
    return "\n\n".join(sections)


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
        self.sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

        print(f"vLLM engine ready. Yes={self.yes_id}, No={self.no_id}")
        print("model loaded.")

    def _score_prompts(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        scores = []
        for out in outputs:
            logprobs_dict = out.outputs[0].logprobs[0]
            yes_lp = logprobs_dict[self.yes_id].logprob if self.yes_id in logprobs_dict else -100.0
            no_lp = logprobs_dict[self.no_id].logprob if self.no_id in logprobs_dict else -100.0
            max_lp = max(yes_lp, no_lp)
            p_yes = math.exp(yes_lp - max_lp) / (math.exp(yes_lp - max_lp) + math.exp(no_lp - max_lp))
            scores.append(p_yes)
        return scores

    def Eval(self, data):
        """DLIS string eval interface. JSON in, JSON out."""
        try:
            req = json.loads(data)
        except Exception as e:
            logger.error(f"Invalid JSON: {e}")
            return json.dumps({"error": f"Invalid JSON: {e}"})

        try:
            t0 = time.time()
            candidates = req.get("candidates", [])
            if not candidates:
                # Single candidate mode (flat request)
                candidates = [req]

            interests = req.get("interests", [])
            shown_titles = req.get("shownTitles", req.get("shown_titles", []))
            conversations = req.get("conversations", [])

            prompts = []
            for card in candidates:
                body = build_inference_prompt(
                    interests=interests,
                    shown_titles=shown_titles,
                    conversations=conversations,
                    candidate_title=card.get("title", ""),
                    candidate_summary=card.get("summary", ""),
                    candidate_matched_interest=card.get("matchedInterest", card.get("matched_interest", "")),
                )
                msgs = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": body},
                ]
                text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
            }
            return json.dumps(response)

        except Exception as e:
            logger.error(f"Eval error: {e}")
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
