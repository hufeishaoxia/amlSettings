"""XGBoost baseline — train on last 7 days (20260410-20260416) only.

Compares 4 AUCs (train {Full,URA-7d} × test {Full,URA}) against the
3-week LightGBM baseline.
"""
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = "data_v10"
FEATS = [
    "quality", "worthiness", "longTermRelevance", "shortTermRelevance",
    "clickLikelihood", "clickHistoryRelevance", "novelty",
    "interestStrength", "interestRecency",
]
DATE_LO, DATE_HI = "20260410", "20260416"

XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
    reg_lambda=1.0, objective="binary:logistic", eval_metric="auc",
    tree_method="hist", n_jobs=-1, random_state=42,
    early_stopping_rounds=50,
)


def load(path, date_lo=None, date_hi=None):
    X, y = [], []
    n_total = n_skip = n_outdate = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            bd = r.get("bizdate", "")
            if date_lo and (bd < date_lo or bd > date_hi):
                n_outdate += 1
                continue
            feats = r.get("features") or {}
            if not feats:
                n_skip += 1
                continue
            try:
                X.append([float(feats.get(k, 0.0) or 0.0) for k in FEATS])
            except (TypeError, ValueError):
                n_skip += 1
                continue
            y.append(int(r["label"]))
    print(f"  [load] {path} [{date_lo or '-'}..{date_hi or '-'}]: "
          f"kept {len(y)}/{n_total} (out_of_range={n_outdate} no_feat={n_skip})")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def fit_eval(name, Xtr, ytr, test_sets):
    print(f"\n=== Train: {name} (n={len(ytr)} pos={int(ytr.sum())}) ===")
    # tiny holdout for early stopping
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(ytr))
    n_val = max(1, int(len(ytr) * 0.1))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(Xtr[tr_idx], ytr[tr_idx],
              eval_set=[(Xtr[val_idx], ytr[val_idx])], verbose=False)
    print(f"  best_iter={model.best_iteration}")
    rows = []
    for te_name, Xte, yte in test_sets:
        pred = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pred)
        ll = log_loss(yte, np.clip(pred, 1e-6, 1 - 1e-6))
        print(f"  test={te_name:<5} n={len(yte):>6}  AUC={auc:.4f}  LogLoss={ll:.4f}")
        rows.append({"trained_on": name, "tested_on": te_name,
                     "AUC": auc, "LogLoss": ll, "n_test": len(yte)})
    return rows


def main():
    print(f"Loading splits, train window = [{DATE_LO}..{DATE_HI}] (7 days)")
    Xtr_full, ytr_full = load(os.path.join(DATA_DIR, "train_all.jsonl"), DATE_LO, DATE_HI)
    Xtr_ura,  ytr_ura  = load(os.path.join(DATA_DIR, "train_ura.jsonl"), DATE_LO, DATE_HI)
    Xte_full, yte_full = load(os.path.join(DATA_DIR, "eval_all.jsonl"))
    Xte_ura,  yte_ura  = load(os.path.join(DATA_DIR, "eval_ura.jsonl"))
    test_sets = [("Full", Xte_full, yte_full), ("URA", Xte_ura, yte_ura)]

    rows = []
    rows += fit_eval("Full-7d", Xtr_full, ytr_full, test_sets)
    rows += fit_eval("URA-7d",  Xtr_ura,  ytr_ura,  test_sets)

    print("\n=== Summary (XGB, 7-day train) ===")
    print(f"{'Trained':<10} {'Tested':<6} {'AUC':>8} {'LogLoss':>8}")
    for r in rows:
        print(f"{r['trained_on']:<10} {r['tested_on']:<6} {r['AUC']:>8.4f} {r['LogLoss']:>8.4f}")

    os.makedirs("eval_results", exist_ok=True)
    out = "eval_results/eval_xgb_v10_7day_matrix.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
