# Qwen3-0.6B Pointwise Ranker â€” DLIS Deployment

## Overview

This folder contains everything needed to deploy the `v9_Qwen3-0.6B_all_ep5/checkpoint-1128` pointwise ranking model as a DLIS Falcon service.

The model takes user context (interests, shown history, conversations) + a candidate card and outputs P(click) âˆˆ [0,1].

## Files

| File | Description |
|------|-------------|
| `inference_server.py` | FastAPI inference server with `/score` and `/health` endpoints |
| `Dockerfile` | CUDA-based container image for DLIS deployment |
| `requirements.txt` | Python dependencies |
| `deploy.sh` | End-to-end deployment script (build â†’ push â†’ DLIS pipeline) |
| `test_request.json` | Example request payload for testing |

## API

### `POST /score`

Score candidate cards given user context.

**Request:**
```json
{
  "interests": [{"name": "AI", "strength": 0.95, "classification": "topic"}],
  "shownTitles": ["Article A", "Article B"],
  "conversations": [],
  "candidates": [
    {"id": "card-1", "title": "New AI Breakthrough", "summary": "...", "matchedInterest": "AI"},
    {"id": "card-2", "title": "Weather Today", "summary": "..."}
  ]
}
```

**Response:**
```json
{
  "scores": [
    {"id": "card-1", "score": 0.87},
    {"id": "card-2", "score": 0.23}
  ],
  "latency_ms": 45.2
}
```

### `GET /health`
Returns `{"status": "healthy", "model": "...", "device": "..."}`.

## Deployment Steps (DLIS)

Following the Qwen-Omni DLIS deployment flow:

### Step 1: Copy checkpoint into this folder
```bash
# From AML compute
cp -r /path/to/output/v9_Qwen3-0.6B_all_ep5/checkpoint-1128 ./checkpoint-1128/
```

### Step 2: Build & push Docker image
```bash
az acr login --name f9309c3acdd842848c88032e1ec736d2
docker build -t f9309c3acdd842848c88032e1ec736d2.azurecr.io/qwen3-06b-ranker:v1 .
docker push f9309c3acdd842848c88032e1ec736d2.azurecr.io/qwen3-06b-ranker:v1
```

### Step 3: AML â†’ DLIS Docker conversion
Run the ADO pipeline: `Pipelines - Run 64277546` (same as Qwen-Omni, update image tag).

### Step 4: DLIS Deploy
Update the DLIS deploy pipeline with:
- Application name: e.g. `qwen3-06b-ranker`
- Image: `dlisfalconprodcontainerregistry.azurecr.io/qwen3-06b-ranker:v1`
- GPU: 1 GPU is sufficient for 0.6B model (vs 4-8 GPU for Qwen-Omni 7B)

### Step 5: Test
```bash
curl -X POST https://fabricrouter-external.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.qwen3-06b-ranker/score \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## Local Testing

```bash
pip install -r requirements.txt
python inference_server.py --model_path ./checkpoint-1128 --port 8080
# In another terminal:
curl -X POST http://localhost:8080/score -H "Content-Type: application/json" -d @test_request.json
```

## Notes

- 0.6B model fits in a single GPU with cl=4096, micro_batch=16
- Model uses chat template (Qwen3 format) with system message
- Scoring: `P(Yes) / (P(Yes) + P(No))` at next-token position (same as eval_auc.py)
- Left-padding + left-truncation for consistent batch inference
