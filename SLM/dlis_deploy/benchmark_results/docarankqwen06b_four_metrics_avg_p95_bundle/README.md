# docarankqwen06b e2e latency bundle

This bundle packages the data and scripts used for `docarankqwen06b_four_metrics_avg_p95` so it can be copied to another machine for e2e latency investigation.

## Contents

- `data/docarankqwen06b_curve_20260428_093952.csv`: raw concurrency sweep data used by the avg/p95 chart.
- `outputs/docarankqwen06b_four_metrics_avg_p95.svg`: existing chart output from the original run.
- `outputs/docarankqwen06b_four_metrics_avg_p95.md`: existing table output from the original run.
- `scripts/plot_four_metrics_avg_p95.py`: standalone standard-library plotting script to regenerate the avg/p95 chart from CSV.
- `scripts/benchmark_docarankqwen06b_curve.py`: benchmark script for rerunning the endpoint sweep.
- `test_request.json`: request payload used by the benchmark.

## Regenerate the chart from included data

```bash
cd docarankqwen06b_four_metrics_avg_p95_bundle
python3 scripts/plot_four_metrics_avg_p95.py \
  --input data/docarankqwen06b_curve_20260428_093952.csv \
  --output-prefix outputs/regenerated_four_metrics_avg_p95
```

Outputs:

- `outputs/regenerated_four_metrics_avg_p95.svg`
- `outputs/regenerated_four_metrics_avg_p95.md`

## Rerun the e2e latency benchmark

```bash
cd docarankqwen06b_four_metrics_avg_p95_bundle
python3 scripts/benchmark_docarankqwen06b_curve.py \
  --request-file test_request.json \
  --levels 1,2,4,8,16,32,64,128,256 \
  --requests-per-level 128 \
  --warmup 2 \
  --timeout 180 \
  --output-dir outputs
```

The script measures client-observed `end2end_*` using `urllib.request.urlopen`, and also parses server-returned `latency_ms`, `prompt_build_ms`, and `inference_ms` from the response. In the original run, internal service latency stayed around 35 ms while e2e latency grew with concurrency, which points to network/router/client connection overhead rather than model inference.

Default endpoint in the benchmark script:

```text
https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.docarankqwen06b/
```

Use `--url` to test a different route.

## Important note for e2e analysis

The benchmark script opens HTTP connections via Python stdlib `urllib`; it does not use persistent connection pooling. That is useful for exposing DNS/TCP/TLS/router cost, but it can overstate client e2e latency compared with a pooled production client. If the new machine is meant to isolate network path cost, compare:

- the included script as-is,
- a pooled HTTP client such as `requests.Session` or `httpx.Client`,
- `curl -w` timings for DNS/connect/TLS/TTFB breakdown.