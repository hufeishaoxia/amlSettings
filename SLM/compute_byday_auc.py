"""Compute per-day URA AUC from per-row score JSONLs produced by eval_auc.py
(--save_scores_ura) and xgb_byday.py.

Each input file contains one JSON per line: {idx, label, score, bizdate, is_ura, ...}
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def per_day_auc(rows, only_ura: bool = True):
    by_day = defaultdict(lambda: ([], []))   # day -> (labels, scores)
    for r in rows:
        if only_ura and not int(r.get("is_ura", 0)):
            continue
        d = (r.get("bizdate") or "")[:8]
        by_day[d][0].append(int(r["label"]))
        by_day[d][1].append(float(r["score"]))
    out = {}
    for d, (ys, ss) in by_day.items():
        y = np.asarray(ys); s = np.asarray(ss)
        n = len(y); pos = int(y.sum()); neg = n - pos
        ctr = pos / max(1, n)
        auc = roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")
        out[d] = {"n": n, "pos": pos, "neg": neg, "ctr": ctr, "auc": auc}
    return out


def main(*paths, only_ura: int = 1):
    """Usage: python compute_byday_auc.py path1 path2 ...

    Args after `--` flags. Pass score JSONL files (one per model).
    """
    if not paths:
        print(__doc__)
        sys.exit(1)
    tables = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        rows = load_scores(p)
        tables[name] = per_day_auc(rows, only_ura=bool(only_ura))

    days = sorted({d for t in tables.values() for d in t})
    names = list(tables.keys())

    # Header
    print()
    print(f"{'day':<10} {'n':>6} {'pos':>5} {'ctr':>7}  " + "  ".join(f"{n:>14}" for n in names))
    for d in days:
        first = tables[names[0]].get(d) or {}
        n = first.get("n", 0); pos = first.get("pos", 0); ctr = first.get("ctr", 0.0)
        cells = []
        for nm in names:
            t = tables[nm].get(d)
            cells.append(f"AUC={t['auc']:.4f}" if t else "(missing)")
        print(f"{d:<10} {n:>6} {pos:>5} {ctr:>7.4f}  " + "  ".join(f"{c:>14}" for c in cells))

    # Overall
    print()
    print(f"{'OVERALL':<10} {'':>6} {'':>5} {'':>7}  ", end="")
    for nm in names:
        all_y, all_s = [], []
        for r in load_scores([p for p in paths if os.path.splitext(os.path.basename(p))[0] == nm][0]):
            if int(only_ura) and not int(r.get("is_ura", 0)):
                continue
            all_y.append(int(r["label"])); all_s.append(float(r["score"]))
        a = roc_auc_score(all_y, all_s) if len(set(all_y)) > 1 else float("nan")
        print(f"{'AUC=%.4f' % a:>14}  ", end="")
    print()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
