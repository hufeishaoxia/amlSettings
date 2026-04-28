# DocaRank Qwen 0.6B four p95 metrics

Source: `benchmark_results/docarankqwen06b_curve_20260428_093952.csv`
Plot: `benchmark_results/docarankqwen06b_four_metrics_together.svg`

| concurrency | end2end_p95_ms | latency_p95_ms | prompt_build_p95_ms | inference_p95_ms |
| --- | --- | --- | --- | --- |
| 1 | 1,038 | 37.1 | 0.40 | 36.7 |
| 2 | 1,034 | 36.9 | 0.40 | 36.6 |
| 4 | 1,050 | 37.2 | 0.40 | 36.9 |
| 8 | 1,167 | 37.2 | 0.40 | 36.8 |
| 16 | 1,540 | 36.1 | 0.30 | 35.9 |
| 32 | 1,989 | 35.8 | 0.30 | 35.5 |
| 64 | 2,565 | 35.4 | 0.30 | 35.1 |
| 128 | 3,594 | 35.4 | 0.30 | 35.1 |
| 256 | 3,573 | 35.3 | 0.30 | 35 |
