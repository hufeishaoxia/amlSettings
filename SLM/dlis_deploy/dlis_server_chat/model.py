"""
DLIS ModelImp for Qwen3-0.6B pointwise ranker (Papyrus-compatible).

Same vLLM scoring logic as `dlis_server/model.py`, but `Eval()` dual-mode:

  * RAW mode (legacy):
      Request:  {"interests": ..., "history": ..., "candidates": [...]}
      Response: {"scores": [{"id","score"}, ...], "latency_ms", ...}

  * OpenAI chat-completions mode (Papyrus pass-through):
      Request:  {"model": "docarankqwen06b", "messages": [{"role":"user","content":"<json string of raw req>"}], ...}
      Response: OpenAI ChatCompletion shape with `choices[0].message.content`
                = JSON string of the raw response (so callers can `json.loads()` it).

Detection: if request top-level has "messages" -> chat mode, else raw.
DLIS framework only exposes a single POST `/`; Papyrus strips the `/chat/completions`
path before forwarding the body, so this single endpoint covers both clients.
"""
import os
import copy
import json
import math
import time
import uuid
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

MODEL_VERSION = "v31-dlis-chat"
MODEL_NAME_PUBLIC = "docarankqwen06b"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence Tornado access-log warnings for unrelated probes (e.g. /metrics 404).
logging.getLogger("tornado.access").setLevel(logging.ERROR)


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

        ckpt_path = os.getenv("QWEN3_MODEL_PATH", "/qwen3_model")
        tp = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
        gpu_mem = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
        dtype = os.getenv("VLLM_DTYPE", "bfloat16")
        self.eval_max_len = int(os.getenv("EVAL_MAX_LEN", "4096"))
        self.body_budget = max(256, self.eval_max_len - 120)

        # Engine-side hard cap on prompt tokens. vLLM rejects any prompt that,
        # together with `max_tokens=1`, exceeds `max_model_len`. We reserve 1
        # token for the answer + a small safety margin to absorb tokenizer
        # quirks (e.g. apply_chat_template wrapper drift).
        self.engine_max_tokens = max_model_len
        self.prompt_token_limit = max(256, max_model_len - 1 - 16)

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

        self.sampling_params = SamplingParams(
            max_tokens=1, temperature=0, logprobs=2,
            allowed_token_ids=[self.yes_id, self.no_id],
        )

        print(f"vLLM engine ready. Yes={self.yes_id}, No={self.no_id}")
        print("model loaded.")

    # --------------------------- vLLM scoring helpers -----------------------

    def _score_prompts(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return [self._extract_yes_prob(o) for o in outputs]

    def _extract_yes_prob(self, out) -> float:
        logprobs_dict = out.outputs[0].logprobs[0]
        yes_lp = logprobs_dict[self.yes_id].logprob
        no_lp = logprobs_dict[self.no_id].logprob
        max_lp = max(yes_lp, no_lp)
        return math.exp(yes_lp - max_lp) / (math.exp(yes_lp - max_lp) + math.exp(no_lp - max_lp))

    # --------------------------- core ranking -------------------------------

    def _render_full_prompt(self, body: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": body},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    def _ntok(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _build_and_fit_prompt(self, history, interests, candidate, body_budget):
        """Build a chat-templated prompt that fits ``self.prompt_token_limit``.

        Pipeline:
          1. ``build_prompt_budgeted`` (halves SHOWN_CARDS, then drops oldest
             conversations from the tail) to fit ``body_budget`` tokens.
          2. Render through ``apply_chat_template`` and count *real* tokens.
          3. If still over the engine cap (wrapper overhead bigger than the
             body_budget reservation), iteratively shrink **interests**:
               a. Drop newest -> oldest CONVERSATIONS until empty.
               b. Then drop trailing entries from USER_INTERACTIONS lists
                  (clicks / thumbsUp / thumbsDown, round-robin).
          4. Last-resort tail-truncate at the token level (keeps the
             question + CANDIDATE_ITEM at the end of the prompt).

        USER_INTERESTS positive/negative lists and CANDIDATE_ITEM are never
        dropped here -- those are the highest-signal sections.
        """
        body, _trunc, _dh, _dc = build_prompt_budgeted(
            history, interests, candidate, self.tokenizer, body_budget,
        )
        text = self._render_full_prompt(body)
        n_tok = self._ntok(text)
        if n_tok <= self.prompt_token_limit:
            return text, n_tok, {}

        # Need to shrink past what build_prompt_budgeted already did.
        drops = {"conversations": 0, "clicks": 0, "thumbsUp": 0, "thumbsDown": 0,
                 "hard_truncated_tokens": 0}
        interests = copy.deepcopy(interests) if isinstance(interests, dict) else interests

        # 1) conversations: pop from tail (matches build_prompt_budgeted order)
        if isinstance(interests, dict):
            convs = list(interests.get("conversations") or [])
            while convs and n_tok > self.prompt_token_limit:
                convs.pop()
                drops["conversations"] += 1
                interests["conversations"] = convs
                body, *_ = build_prompt_budgeted(
                    history, interests, candidate, self.tokenizer, body_budget,
                )
                text = self._render_full_prompt(body)
                n_tok = self._ntok(text)

        # 2) interaction titles (impressions): round-robin pop from tail
        if isinstance(interests, dict) and n_tok > self.prompt_token_limit:
            inter = dict(interests.get("interactions") or {})
            for k in ("clicks", "thumbsUp", "thumbsDown"):
                inter[k] = list(inter.get(k) or [])
            interests["interactions"] = inter
            keys = ("clicks", "thumbsDown", "thumbsUp")  # drop clicks first, keep thumbsUp longest
            i = 0
            while n_tok > self.prompt_token_limit and any(inter[k] for k in keys):
                k = keys[i % len(keys)]
                i += 1
                if not inter[k]:
                    continue
                inter[k].pop()
                drops[k] += 1
                body, *_ = build_prompt_budgeted(
                    history, interests, candidate, self.tokenizer, body_budget,
                )
                text = self._render_full_prompt(body)
                n_tok = self._ntok(text)

        # 3) hard fallback: tail-truncate ids. Keep the tail because the
        #    question + CANDIDATE_ITEM sit at the end of the prompt and matter
        #    most for scoring. (vLLM will be re-fed text -- decode round trip.)
        if n_tok > self.prompt_token_limit:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            removed = len(ids) - self.prompt_token_limit
            ids = ids[removed:]
            text = self.tokenizer.decode(ids)
            n_tok = len(ids)
            drops["hard_truncated_tokens"] = removed

        return text, n_tok, drops

    def _rank(self, req: dict) -> dict:
        """Run the ranker on a parsed raw-schema request dict.
        Returns the same dict that legacy `Eval()` returned (pre-json.dumps).
        """
        t0 = time.time()
        history, interests, candidates = normalize_request(req)
        req_max_len = int(req.get("max_len") or self.eval_max_len)
        body_budget = max(256, req_max_len - 120)

        prompts = []
        shrink_events = []
        for card in candidates:
            cand = {
                "title": card.get("title", ""),
                "summary": card.get("summary", ""),
            }
            text, n_tok, drops = self._build_and_fit_prompt(
                history, interests, cand, body_budget,
            )
            if any(v for v in drops.values()):
                shrink_events.append({"id": card.get("id", ""), "n_tok": n_tok, **drops})
            prompts.append(text)

        if shrink_events:
            logger.warning(
                "[rank] shrink_applied n=%d limit=%d events=%s",
                len(shrink_events), self.prompt_token_limit, shrink_events,
            )

        t1 = time.time()
        all_scores = self._score_prompts(prompts)
        t2 = time.time()

        results = [
            {"id": card.get("id", ""), "score": round(score, 6)}
            for card, score in zip(candidates, all_scores)
        ]

        response = {
            "scores": results,
            "latency_ms": round((t2 - t0) * 1000, 1),
            "prompt_build_ms": round((t1 - t0) * 1000, 1),
            "inference_ms": round((t2 - t1) * 1000, 1),
            "model_version": MODEL_VERSION,
        }

        try:
            score_detail = ", ".join(f"{r['id']}={r['score']:.4f}" for r in results)
        except Exception:
            score_detail = str(results)
        logger.info(
            "[rank] model=%s n=%d total_ms=%.1f build_ms=%.1f infer_ms=%.1f scores=[%s]",
            MODEL_VERSION,
            len(results),
            response["latency_ms"],
            response["prompt_build_ms"],
            response["inference_ms"],
            score_detail,
        )

        return response

    # --------------------------- chat-completions wrap ----------------------

    @staticmethod
    def _is_chat_request(req: dict) -> bool:
        # OpenAI chat-completions: must have non-empty `messages` list.
        msgs = req.get("messages")
        return isinstance(msgs, list) and len(msgs) > 0

    @staticmethod
    def _extract_raw_payload_from_chat(req: dict) -> dict:
        """Pull the raw ranker payload out of a chat-completions request.

        Convention: callers JSON-serialize the raw payload and put the string
        as `messages[-1].content` (role=user). We accept either the last user
        message or the last message overall to be liberal.
        """
        messages = req["messages"]
        # Prefer last user message; fall back to last message.
        candidate = None
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                candidate = m
                break
        if candidate is None:
            candidate = messages[-1]

        if not isinstance(candidate, dict):
            raise ValueError("messages[-1] must be an object")

        content = candidate.get("content")
        if isinstance(content, list):
            # OpenAI vision-style content: list of {"type":"text","text":"..."}
            text_parts = [
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = "".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise ValueError("messages[-1].content must be a non-empty string")

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"messages[-1].content must be a JSON string of the raw ranker payload: {e}")

    def _wrap_chat_response(self, raw_response: dict, req_model: str) -> dict:
        """Wrap raw ranker response into OpenAI ChatCompletion shape."""
        content_str = json.dumps(raw_response, ensure_ascii=False)
        # Rough token accounting (Papyrus quota uses model-seconds, not tokens,
        # so exact values do not affect billing for our setup).
        approx_tokens = max(1, len(content_str) // 4)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req_model or MODEL_NAME_PUBLIC,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content_str},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": approx_tokens,
                "total_tokens": approx_tokens,
            },
            # Custom passthrough fields useful for monitoring / debugging.
            "x_ranker_latency_ms": raw_response.get("latency_ms"),
            "x_ranker_inference_ms": raw_response.get("inference_ms"),
            "x_ranker_prompt_build_ms": raw_response.get("prompt_build_ms"),
            "x_model_version": raw_response.get("model_version"),
        }

    # --------------------------- DLIS Eval entrypoint -----------------------

    # Cap how many chars of response body we dump to log per request so we
    # don't blow up DLIS log ingestion on huge candidate batches. Override via
    # env if needed.
    _RESP_LOG_MAX_CHARS = int(os.getenv("RESP_LOG_MAX_CHARS", "4000"))

    def _log_response(self, response_str: str, status: str = "ok") -> None:
        body = response_str
        n = len(body)
        if n > self._RESP_LOG_MAX_CHARS:
            body = body[: self._RESP_LOG_MAX_CHARS] + f"...<truncated, total_len={n}>"
        logger.info("[resp] status=%s len=%d body=%s", status, n, body)

    def Eval(self, data):
        """DLIS string eval interface. JSON in, JSON out.

        Two-mode dispatch:
          * If body has top-level `messages` (OpenAI chat-completions),
            unwrap -> rank -> wrap into ChatCompletion response.
          * Otherwise, treat body as the raw ranker schema.
        """
        try:
            req = json.loads(data)
        except Exception as e:
            logger.error(f"Invalid JSON: {e}")
            resp = json.dumps({"error": f"Invalid JSON: {e}"})
            self._log_response(resp, status="invalid_json")
            return resp

        try:
            if self._is_chat_request(req):
                # ----- chat-completions mode (Papyrus) -----
                raw_payload = self._extract_raw_payload_from_chat(req)
                raw_response = self._rank(raw_payload)
                chat_response = self._wrap_chat_response(raw_response, req.get("model", ""))
                resp = json.dumps(chat_response, ensure_ascii=False)
                self._log_response(resp, status="ok_chat")
                return resp
            else:
                # ----- raw mode (legacy) -----
                raw_response = self._rank(req)
                resp = json.dumps(raw_response, ensure_ascii=False)
                self._log_response(resp, status="ok_raw")
                return resp
        except ValueError as e:
            logger.error(f"Bad chat request: {e}")
            resp = json.dumps({"error": str(e)})
            self._log_response(resp, status="bad_request")
            return resp
        except Exception as e:
            logger.exception(f"Eval error: {e}")
            resp = json.dumps({"error": str(e)})
            self._log_response(resp, status="internal_error")
            return resp

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
