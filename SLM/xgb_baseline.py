"""XGBoost CTR baseline on the same parquet split as the SLM.

Train: impressions with `features` dict, bizdate <= TRAIN_UNTIL, URA flight only.
Eval : impressions with `features` dict, bizdate >= EVAL_FROM
       (reports AUC on URA slice and on ALL flights).

Usage:
    python xgb_baseline.py [data_dir]
"""
import glob
import json
import os
import sys
from typing import List

import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "data"
TRAIN_UNTIL = os.environ.get("TRAIN_UNTIL", "20260416")
EVAL_FROM   = os.environ.get("EVAL_FROM",   "20260417")
URA_FLIGHT  = os.environ.get("URA_FLIGHT",  "discover-rk-ura")
TRAIN_URA_ONLY = os.environ.get("TRAIN_URA_ONLY", "true").lower() in ("1", "true", "yes")

SCORE_FEATURES = [
    "quality", "worthiness", "longTermRelevance", "shortTermRelevance",
    "clickLikelihood", "clickHistoryRelevance", "novelty",
    "interestStrength", "interestRecency",
]

NEEDED_COLS = ["bizdate", "user_flight_ids", "candidate_cards"]


def _safe_json(s, default):
    if s is None: return default
    if isinstance(s, (list, dict)): return s
    try: return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError): return default


def _is_impression(c: dict) -> bool:
    si = c.get("sectionIndex")
    return si is not None and str(si) != ""


def load_xy(data_dir: str, bizdate_min: str = "", bizdate_max: str = "",
            ura_only: bool = False):
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    X, y, flights, sec_idx = [], [], [], []
    for p in paths:
        pf = pq.ParquetFile(p)
        cols = [c for c in NEEDED_COLS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            for row in batch.to_pylist():
                bd = (row.get("bizdate") or "").strip()
                if bizdate_min and bd and bd < bizdate_min: continue
                if bizdate_max and bd and bd > bizdate_max: continue
                fl = row.get("user_flight_ids") or ""
                if ura_only and URA_FLIGHT not in fl: continue
                cands = _safe_json(row.get("candidate_cards"), [])
                for c in cands:
                    if not isinstance(c, dict) or not _is_impression(c): continue
                    feats = c.get("features")
                    if not isinstance(feats, dict): continue
                    vec = []
                    ok = True
                    for f in SCORE_FEATURES:
                        v = feats.get(f)
                        if v is None: v = 0.0
                        try: vec.append(float(v))
                        except (TypeError, ValueError): ok = False; break
                    if not ok: continue
                    X.append(vec)
                    y.append(1 if bool(c.get("is_clicked")) else 0)
                    flights.append(fl)
                    try: sec_idx.append(int(c.get("sectionIndex", 0)))
                    except Exception: sec_idx.append(0)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int32),
            np.asarray(flights, dtype=object),
            np.asarray(sec_idx, dtype=np.int32))


def main():
    print(f"DATA_DIR={DATA_DIR}  TRAIN<={TRAIN_UNTIL}  EVAL>={EVAL_FROM}  URA={URA_FLIGHT}  TRAIN_URA_ONLY={TRAIN_URA_ONLY}")
    print(f"Loading train ({'URA only' if TRAIN_URA_ONLY else 'all flights'}) ...")
    Xtr, ytr, _, _ = load_xy(DATA_DIR, bizdate_max=TRAIN_UNTIL, ura_only=TRAIN_URA_ONLY)
    print(f"  train: {Xtr.shape}  pos={int(ytr.sum())}  neg={len(ytr)-int(ytr.sum())}  ctr={ytr.mean():.4f}")

    print("Loading eval (all flights) ...")
    Xev, yev, fev, _ = load_xy(DATA_DIR, bizdate_min=EVAL_FROM, ura_only=False)
    print(f"  eval : {Xev.shape}  pos={int(yev.sum())}  neg={len(yev)-int(yev.sum())}  ctr={yev.mean():.4f}")

    # XGBoost
    pos = max(1, int(ytr.sum())); neg = max(1, len(ytr) - pos)
    spw = neg / pos
    params = dict(
        n_estimators=500, learning_rate=0.05,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", n_jobs=-1, random_state=42,
        scale_pos_weight=spw,
    )
    print(f"Training XGBoost (scale_pos_weight={spw:.2f}) ...")
    clf = xgb.XGBClassifier(**params)
    clf.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)

    pred = clf.predict_proba(Xev)[:, 1]

    ura_mask = np.array([URA_FLIGHT in (f or "") for f in fev])
    print("\n=== Results ===")
    print(f"{'split':<12} {'n':>8} {'pos':>6} {'ctr':>7} {'AUC':>8} {'LogLoss':>9}")

    def report(name: str, mask: np.ndarray):
        if mask.sum() == 0:
            print(f"{name:<12} (empty)"); return
        yy = yev[mask]; pp = pred[mask]
        if len(np.unique(yy)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(yy, pp)
        ll = log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))
        print(f"{name:<12} {len(yy):>8d} {int(yy.sum()):>6d} {yy.mean():>7.4f} {auc:>8.4f} {ll:>9.4f}")

    report("ALL",  np.ones_like(ura_mask, dtype=bool))
    report("URA",  ura_mask)
    report("non-URA", ~ura_mask)

    # Per-feature single-feature AUC for sanity
    print("\nSingle-feature AUC on EVAL (ALL):")
    for i, f in enumerate(SCORE_FEATURES):
        vals = Xev[:, i]
        if len(np.unique(vals)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(yev, vals)
        print(f"  {f:<22} {auc:.4f}")

    # Feature importance
    print("\nXGB feature importance (gain):")
    booster = clf.get_booster()
    imp = booster.get_score(importance_type="gain")
    # XGB names features as f0..fN by default
    for i, f in enumerate(SCORE_FEATURES):
        print(f"  {f:<22} {imp.get(f'f{i}', 0.0):.2f}")


if __name__ == "__main__":
    main()
