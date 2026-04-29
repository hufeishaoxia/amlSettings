# dlis_server_chat — Papyrus-compatible DLIS deploy

A drop-in replacement for `dlis_server/`. Adds **OpenAI chat-completions**
support so the model can be onboarded behind Papyrus, while keeping the
legacy raw schema fully working for existing callers.

## What changed

Only `model.py`. Everything else (`http_server.py`, `main.py`, `utils.py`,
`prompt.py`, protobuf helpers, `run.sh`, requirements) is byte-identical
to the original `dlis_server/`.

DLIS framework only exposes a single POST `/` (see `http_server.py` —
"DO NOT EDIT"). So we do schema dispatch inside `Eval()`:

```
POST /  body = {"interests":..., "candidates":[...]}        -> raw mode (legacy)
POST /  body = {"messages":[{"role":"user","content":"..."}]} -> chat mode (Papyrus)
```

Detection: presence of top-level non-empty `messages` list.

### Chat mode contract

**Request** (Papyrus strips `/chat/completions` and forwards body to DLIS `/`):

```json
{
  "model": "docarankqwen06b",
  "messages": [
    {"role": "user", "content": "<JSON string of the raw ranker payload>"}
  ],
  "max_tokens": 1,
  "temperature": 0.0
}
```

`messages[-1].content` MUST be a JSON-serialized string of the raw ranker
payload (`{"interests":..., "history":..., "candidates":[...]}`). Vision-style
content arrays are also accepted (text parts are concatenated).

**Response**:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1714435200,
  "model": "docarankqwen06b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "{\"scores\":[{\"id\":\"card-001\",\"score\":0.622}, ...], \"latency_ms\":..., \"model_version\":\"v28-dlis-chat\"}"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": N, "total_tokens": N},
  "x_ranker_latency_ms": ...,
  "x_ranker_inference_ms": ...,
  "x_model_version": "v28-dlis-chat"
}
```

The caller does `json.loads(resp["choices"][0]["message"]["content"])` to get
the original `{"scores":[...]}`. The `x_ranker_*` fields are duplicated at top
level for easier monitoring without parsing the inner JSON.

## Deploy

Same as the original. Package this folder into the DLIS image:

```bash
# in SLM/dlis_deploy/
# (same Dockerfile / deploy_dlis.sh, just point to dlis_server_chat instead of dlis_server)
```

Suggested approach: copy `Dockerfile` and replace `COPY dlis_server` with
`COPY dlis_server_chat`. Or symlink during build. Either way the deployed
container behaves identically for raw callers and additionally accepts chat.

## Verify locally

```bash
cd dlis_server_chat
python main.py http        # listens on 8888
```

```bash
# raw mode (existing path)
curl -s http://localhost:8888 --data @../test_request.json | jq

# chat mode (new path)
RAW=$(cat ../test_request.json | jq -c .)
jq -n --arg c "$RAW" '{model:"docarankqwen06b", messages:[{role:"user",content:$c}], max_tokens:1, temperature:0}' \
  | curl -s http://localhost:8888 --data-binary @- | jq
```

Both should return scores. The chat response wraps them in `choices[0].message.content`.

## Benchmark

Use the new `../benchmark_chat_completions.py` (keep-alive enabled):

```bash
# A) Direct DLIS, legacy raw
python ../benchmark_chat_completions.py --mode raw --url http://localhost:8888 \
  --requests 100 --concurrency 8 --warmup 5 --vary-request

# B) Direct DLIS, chat wrapper (validates new code)
python ../benchmark_chat_completions.py --mode chat --url http://localhost:8888 \
  --requests 100 --concurrency 8 --warmup 5 --vary-request

# C) Papyrus GLB, chat (after AppIds PR merged)
python ../benchmark_chat_completions.py --mode chat \
  --url https://westus2.papyrus.binginternal.com/chat/completions \
  --papyrus-model-name docarankqwen06b-Picasso \
  --papyrus-quota-id picasso/discover \
  --aad-resource api://5fe538a8-15d5-4a84-961e-be66cd036687 \
  --requests 200 --concurrency 16 --warmup 5 --vary-request
```

A and B should produce near-identical latency (chat wrap overhead is
~0.1ms — JSON serialize). If they differ noticeably, look at
`prompt_build_ms` to confirm.

## Migration plan

1. Build + deploy `dlis_server_chat` to a NEW DLIS endpoint
   (e.g. `dlis-coreranker.docarankqwen06b-chat`) — keep the existing one running.
2. Run benchmarks A vs B to confirm parity.
3. Update the Papyrus PR to point `EndpointConfigs[0].URL` at the new endpoint
   (or onboard the new endpoint as a second endpoint with weight 1, old with weight 0).
4. Once Papyrus path validated, sunset the raw-only endpoint.

Alternatively (less risky): redeploy the *existing* endpoint with this
`dlis_server_chat`, since it is fully backward compatible for raw callers.
