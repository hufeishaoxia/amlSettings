"""Train XGBoost on data_v9/train_all.jsonl features and score data_v9/eval_ura.jsonl.

Saves per-row scores to scores/xgb_v9_ura.jsonl with same schema as eval_auc.py
score files: {idx, label, score, bizdate, is_ura, feed_id, user_id}.
"""
import json
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

SCORE_FEATURES = [
    "quality", "worthiness", "longTermRelevance", "shortTermRelevance",
    "clickLikelihood", "clickHistoryRelevance", "novelty",
    "interestStrength", "interestRecency",
]


def load_jsonl_xy(path: str):
    X, y, meta = [], [], []
    n_skip = 0
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            feats = s.get("features")
            if not isinstance(feats, dict):
                n_skip += 1
                continue
            vec = []
            ok = True
            for k in SCORE_FEATURES:
                v = feats.get(k)
                if v is None:
                    v = 0.0
                try:
                    vec.append(float(v))
                except (TypeError, ValueError):
                    ok = False
                    break
            if not ok:
                n_skip += 1
                continue
            X.append(vec)
            y.append(int(s.get("label", 0)))
            meta.append({
                "label": int(s.get("label", 0)),
                "bizdate": s.get("bizdate", ""),
                "is_ura": int(s.get("is_ura", 0)),
                "feed_id": s.get("feed_id", ""),
                "user_id": s.get("user_id", ""),
            })
    print(f"  loaded {len(X)} rows from {path} (skipped {n_skip} without features)")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), meta


def main(
    train_jsonl: str = "data_v9/train_all.jsonl",
    eval_jsonl: str = "data_v9/eval_ura.jsonl",
    out_scores: str = "scores/xgb_v9_ura.jsonl",
):
    print(f"Loading train: {train_jsonl}")
    Xtr, ytr, _ = load_jsonl_xy(train_jsonl)
    print(f"  train shape={Xtr.shape}  pos={int(ytr.sum())}  neg={len(ytr)-int(ytr.sum())}  ctr={ytr.mean():.4f}")

    print(f"Loading eval: {eval_jsonl}")
    Xev, yev, meta = load_jsonl_xy(eval_jsonl)
    print(f"  eval shape={Xev.shape}  pos={int(yev.sum())}  neg={len(yev)-int(yev.sum())}  ctr={yev.mean():.4f}")

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

    auc_all = roc_auc_score(yev, pred) if len(set(yev.tolist())) > 1 else float("nan")
    ll = log_loss(yev, np.clip(pred, 1e-6, 1 - 1e-6))
    print(f"\nEval URA: n={len(yev)}  pos={int(yev.sum())}  ctr={yev.mean():.4f}  AUC={auc_all:.4f}  LogLoss={ll:.4f}")

    os.makedirs(os.path.dirname(out_scores) or ".", exist_ok=True)
    with open(out_scores, "w") as f:
        for i, (m, p) in enumerate(zip(meta, pred)):
            f.write(json.dumps({"idx": i, "label": m["label"], "score": float(p),
                                "bizdate": m["bizdate"], "is_ura": m["is_ura"],
                                "feed_id": m["feed_id"], "user_id": m["user_id"]}) + "\n")
    print(f"wrote per-row scores to {out_scores}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
