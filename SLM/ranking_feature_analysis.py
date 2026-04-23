# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC # Ranking Feature Analysis & Position Bias Evaluation
# MAGIC
# MAGIC 1. Feature-label correlation analysis (10 features vs is_clicked)
# MAGIC 2. Position bias evaluation: Baseline / Position-aware+drop / IPW-weighted

# COMMAND ----------

# MAGIC %pip install lightgbm scikit-learn matplotlib seaborn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load data
df_spark = spark.table("mai_ws_discover.analytics.ods_doca_ranking_ofe_wide_table") \
    .filter(F.col("RunId") == "20260420")

print(f"Total rows: {df_spark.count()}")
df_spark.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation

# COMMAND ----------

# Define features
SCORE_FEATURES = [
    "quality", "worthiness", "longTermRelevance", "shortTermRelevance",
    "clickLikelihood", "clickHistoryRelevance", "novelty",
    "interestStrength", "interestRecency"
]

# Parse features JSON → extract score features as columns
from pyspark.sql.types import MapType, StringType, DoubleType

@F.udf(MapType(StringType(), DoubleType()))
def extract_double_features(features_json):
    """Parse features JSON, keep only numeric (double) fields."""
    import json as _json
    if not features_json:
        return {}
    try:
        data = _json.loads(features_json)
    except (_json.JSONDecodeError, TypeError):
        return {}
    result = {}
    for k, v in data.items():
        if k in ["ParentId", "SpanId", "contentId"]:
            continue
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif isinstance(v, str):
            try:
                result[k] = float(v)
            except (ValueError, TypeError):
                pass
    return result

df_prep = df_spark.withColumn("_fmap", extract_double_features(F.col("features"))) \
    .select(
        F.col("picasso_user_id"),
        F.col("feedId"),
        F.col("sectionIndex").cast("int"),
        F.col("cardIndex").cast("int"),
        F.col("cardType"),
        F.col("clicks").cast("int"),
        F.col("impressions").cast("int"),
        F.coalesce(F.col("user_flight_ids"), F.lit("")).alias("user_flight_ids"),
        *[F.coalesce(F.col("_fmap").getItem(f), F.lit(0.0)).alias(f) for f in SCORE_FEATURES]
    ).withColumn(
        "is_clicked", F.when(F.col("clicks") > 0, 1).otherwise(0).cast("int")
    )

# Derive global position rank within each request
# Sort by sectionIndex asc, cardIndex asc, rank from 0
w = Window.partitionBy("feedId").orderBy("sectionIndex", "cardIndex")
df_prep = df_prep.withColumn("position_rank", F.row_number().over(w) - 1)

# Convert to pandas
pdf = df_prep.toPandas()
print(f"Pandas shape: {pdf.shape}")
print(f"Click rate: {pdf['is_clicked'].mean():.4f}")
print(f"Position rank range: {pdf['position_rank'].min()} - {pdf['position_rank'].max()}")
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Feature-Label Correlation Analysis

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats

LABEL = "is_clicked"

# One-hot encode cardType
card_type_dummies = pd.get_dummies(pdf["cardType"], prefix="cardType")
feature_cols = SCORE_FEATURES + list(card_type_dummies.columns)
pdf_features = pd.concat([pdf[SCORE_FEATURES], card_type_dummies], axis=1)

# Single-feature AUC: how well does each feature alone rank clicks?
# AUC = 0.5 random; >0.5 positive predictor; <0.5 negative predictor
# lift = AUC - 0.5 (signed strength), abs_lift for ranking
labels = pdf[LABEL].values
auc_results = []
for col in feature_cols:
    vals = pdf_features[col].fillna(0).values
    if len(np.unique(vals)) < 2:
        auc = 0.5
    else:
        try:
            auc = roc_auc_score(labels, vals)
        except Exception:
            auc = 0.5
    lift = auc - 0.5
    auc_results.append({"feature": col, "auc": auc, "lift": lift, "abs_lift": abs(lift)})

corr_df = pd.DataFrame(auc_results).sort_values("abs_lift", ascending=False)
print(corr_df.to_string(index=False))

# Same AUC on URA slice only (unbiased traffic)
URA_FLIGHT_TAG = "discover-rk-ura"
_ura_mask_for_auc = pdf["user_flight_ids"].fillna("").str.contains(URA_FLIGHT_TAG, regex=False).values
_pdf_features_ura = pdf_features[_ura_mask_for_auc]
_labels_ura = pdf[LABEL].values[_ura_mask_for_auc]
auc_results_ura = []
for col in feature_cols:
    vals = _pdf_features_ura[col].fillna(0).values
    if len(np.unique(vals)) < 2 or len(np.unique(_labels_ura)) < 2:
        auc_u = 0.5
    else:
        try:
            auc_u = roc_auc_score(_labels_ura, vals)
        except Exception:
            auc_u = 0.5
    lift_u = auc_u - 0.5
    auc_results_ura.append({"feature": col, "auc": auc_u, "lift": lift_u, "abs_lift": abs(lift_u)})
corr_df_ura = pd.DataFrame(auc_results_ura)
# Order to match Full ranking for side-by-side comparison
order = corr_df["feature"].tolist()
corr_df_ura = corr_df_ura.set_index("feature").loc[order].reset_index()
print("\nURA-only single-feature AUC:")
print(corr_df_ura.to_string(index=False))

# Plot lift = AUC - 0.5
fig, ax = plt.subplots(figsize=(10, max(6, len(corr_df) * 0.35)))
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in corr_df["lift"]]
ax.barh(corr_df["feature"], corr_df["lift"], color=colors)
ax.set_xlabel("Single-Feature AUC Lift (AUC - 0.5)")
ax.set_title("Feature-Label Correlation Analysis (Single-Feature AUC)")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Position Bias Analysis

# COMMAND ----------

# CTR by position
pos_ctr = pdf.groupby("position_rank").agg(
    impressions=("is_clicked", "count"),
    clicks=("is_clicked", "sum")
).reset_index()
pos_ctr["ctr"] = pos_ctr["clicks"] / pos_ctr["impressions"]

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(pos_ctr["position_rank"], pos_ctr["impressions"], alpha=0.3, label="Sample count")
ax1.set_ylabel("Sample Count")
ax2 = ax1.twinx()
ax2.plot(pos_ctr["position_rank"], pos_ctr["ctr"], "r-o", markersize=3, label="CTR")
ax2.set_ylabel("CTR")
ax1.set_xlabel("Position Rank")
ax1.set_title("CTR by Position Rank (Position Bias Visualization)")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Training & Comparison
# MAGIC
# MAGIC - **Baseline**: 10 features, no position, no IPW
# MAGIC - **Position-aware + drop**: 10 features + position_rank in training, drop position at inference
# MAGIC - **IPW-weighted**: 10 features, no position, IPW-weighted loss

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.metrics import log_loss, ndcg_score
import lightgbm as lgb

# Prepare feature matrix
X_base = pdf_features.fillna(0).values
X_with_pos = np.column_stack([X_base, pdf["position_rank"].values])
y = pdf[LABEL].values

base_feature_names = feature_cols
pos_feature_names = feature_cols + ["position_rank"]

# Train/test split (stratified, by requestId hash for no leakage)
request_ids = pdf["feedId"].unique()
train_req, test_req = train_test_split(request_ids, test_size=0.2, random_state=42)
train_mask = pdf["feedId"].isin(train_req).values
test_mask = pdf["feedId"].isin(test_req).values

X_train_base, X_test_base = X_base[train_mask], X_base[test_mask]
X_train_pos, X_test_pos = X_with_pos[train_mask], X_with_pos[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
pos_train = pdf["position_rank"].values[train_mask]
pos_test = pdf["position_rank"].values[test_mask]

print(f"Train: {len(y_train)}, Test: {len(y_test)}")
print(f"Train click rate: {y_train.mean():.4f}, Test click rate: {y_test.mean():.4f}")

# COMMAND ----------

# IPW propensity weights
# Estimate P(examine | position) as normalized CTR per position
pos_ctr_map = pos_ctr.set_index("position_rank")["ctr"]
max_ctr = pos_ctr_map.max()
# Propensity = CTR(pos) / CTR(pos_0), clipped to [0.05, 1.0]
propensity_train = np.array([pos_ctr_map.get(p, max_ctr) for p in pos_train]) / max_ctr
propensity_train = np.clip(propensity_train, 0.05, 1.0)

# IPW weight: for clicked samples, weight = 1/propensity; for non-clicked, weight = 1
# This upweights clicks at low-exposure positions
ipw_weights = np.where(y_train == 1, 1.0 / propensity_train, 1.0)
print(f"IPW weight range: {ipw_weights.min():.2f} - {ipw_weights.max():.2f}")
print(f"IPW weight mean (clicked): {ipw_weights[y_train==1].mean():.2f}")
print(f"IPW weight mean (not clicked): {ipw_weights[y_train==0].mean():.2f}")

# COMMAND ----------

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 300,
}

results = {}

# --- Model 1: Baseline (no position, no IPW) ---
model_base = lgb.LGBMClassifier(**LGB_PARAMS)
model_base.fit(X_train_base, y_train, feature_name=base_feature_names)
pred_base = model_base.predict_proba(X_test_base)[:, 1]
results["Baseline"] = {
    "AUC": roc_auc_score(y_test, pred_base),
    "LogLoss": log_loss(y_test, pred_base),
}

# --- Model 2: Position-aware, drop position at inference ---
model_pos = lgb.LGBMClassifier(**LGB_PARAMS)
model_pos.fit(X_train_pos, y_train, feature_name=pos_feature_names)

# Inference WITHOUT position (set position_rank = 0 for all, simulating "no bias")
X_test_pos_dropped = X_test_pos.copy()
X_test_pos_dropped[:, -1] = 0  # position_rank = 0 for all
pred_pos = model_pos.predict_proba(X_test_pos_dropped)[:, 1]
results["Pos-Aware+Drop"] = {
    "AUC": roc_auc_score(y_test, pred_pos),
    "LogLoss": log_loss(y_test, pred_pos),
}

# Also evaluate WITH position (upper bound / data leakage reference)
pred_pos_leak = model_pos.predict_proba(X_test_pos)[:, 1]
results["Pos-Aware (leak ref)"] = {
    "AUC": roc_auc_score(y_test, pred_pos_leak),
    "LogLoss": log_loss(y_test, pred_pos_leak),
}

# --- Model 3: IPW-weighted (no position feature) ---
model_ipw = lgb.LGBMClassifier(**LGB_PARAMS)
model_ipw.fit(X_train_base, y_train, sample_weight=ipw_weights, feature_name=base_feature_names)
pred_ipw = model_ipw.predict_proba(X_test_base)[:, 1]
results["IPW-Weighted"] = {
    "AUC": roc_auc_score(y_test, pred_ipw),
    "LogLoss": log_loss(y_test, pred_ipw),
}

# Summary
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Results Visualization

# COMMAND ----------

# Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = list(results.keys())
aucs = [results[m]["AUC"] for m in models]
losses = [results[m]["LogLoss"] for m in models]
colors = ["#3498db", "#e67e22", "#95a5a6", "#2ecc71"]

axes[0].bar(models, aucs, color=colors)
axes[0].set_title("AUC Comparison")
axes[0].set_ylabel("AUC")
axes[0].set_ylim(min(aucs) - 0.02, max(aucs) + 0.02)
for i, v in enumerate(aucs):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

axes[1].bar(models, losses, color=colors)
axes[1].set_title("LogLoss Comparison (lower is better)")
axes[1].set_ylabel("LogLoss")
axes[1].set_ylim(min(losses) - 0.02, max(losses) + 0.02)
for i, v in enumerate(losses):
    axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Feature importance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for idx, (name, model, fnames) in enumerate([
    ("Baseline", model_base, base_feature_names),
    ("Pos-Aware+Drop", model_pos, pos_feature_names),
    ("IPW-Weighted", model_ipw, base_feature_names),
]):
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": fnames, "importance": imp}) \
        .sort_values("importance", ascending=True).tail(15)
    axes[idx].barh(imp_df["feature"], imp_df["importance"])
    axes[idx].set_title(f"{name} - Top 15 Feature Importance")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Per-Position Fairness Check
# MAGIC
# MAGIC Compare predicted scores across positions — a fair model should give similar scores to same content regardless of position.

# COMMAND ----------

# Debias diagnostics: 4 metrics per model
#   spearman_r        : pred vs position rank correlation (|r|→0 is fair)
#   ipw_auc_delta     : IPW-weighted AUC − raw AUC (|Δ|→0 is fair)
#   cf_ndcg_delta     : NDCG@10 of model re-rank − NDCG@10 of original display order (>0 = adds value beyond position)
#   pal_residual_r2   : R² of OLS(pred ~ position_rank) (→0 means model doesn't encode position)

from sklearn.metrics import ndcg_score as _ndcg

def _ipw_auc(y_true, preds, propensity):
    # weighted AUC via Mann-Whitney with sample weights = 1/propensity
    w = 1.0 / np.clip(propensity, 0.05, 1.0)
    pos_mask = (y_true == 1)
    if pos_mask.sum() == 0 or (~pos_mask).sum() == 0:
        return float("nan")
    pos_p, neg_p = preds[pos_mask], preds[~pos_mask]
    pos_w, neg_w = w[pos_mask], w[~pos_mask]
    # pairwise (positives ranked above negatives) weighted by pos_w*neg_w
    diff = pos_p[:, None] - neg_p[None, :]
    win = (diff > 0).astype(float) + 0.5 * (diff == 0)
    weights = pos_w[:, None] * neg_w[None, :]
    return float((win * weights).sum() / weights.sum())

def _cf_ndcg_delta(feed_ids, y_true, preds, position_rank, k=10):
    df_ = pd.DataFrame({"feed": feed_ids, "y": y_true, "pred": preds, "pos": position_rank})
    orig_scores, model_scores = [], []
    for _, g in df_.groupby("feed"):
        if len(g) < 2 or g["y"].sum() == 0:
            continue
        y_arr = g["y"].values.reshape(1, -1)
        # original display order: lower position rank = shown earlier = higher "score"
        orig_score = -g["pos"].values.reshape(1, -1).astype(float)
        pred_score = g["pred"].values.reshape(1, -1)
        kk = min(k, len(g))
        orig_scores.append(_ndcg(y_arr, orig_score, k=kk))
        model_scores.append(_ndcg(y_arr, pred_score, k=kk))
    if not orig_scores:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(model_scores) - np.mean(orig_scores)), float(np.mean(model_scores)), float(np.mean(orig_scores))

def _pal_residual_r2(preds, position_rank):
    # R² of linear fit pred = a + b * position_rank  → how much variance in pred explained by position alone
    p = position_rank.astype(float)
    if p.std() == 0:
        return 0.0
    corr = np.corrcoef(p, preds)[0, 1]
    return float(corr ** 2) if not np.isnan(corr) else float("nan")

# Build test-set propensity from pos_ctr_map (same as training)
propensity_test = np.array([pos_ctr_map.get(p, max_ctr) for p in pos_test]) / max_ctr
propensity_test = np.clip(propensity_test, 0.05, 1.0)
test_feeds = pdf["feedId"].values[test_mask]

fairness = {}
for name, preds in [("Baseline", pred_base), ("Pos-Aware+Drop", pred_pos), ("IPW-Weighted", pred_ipw)]:
    r, p = stats.spearmanr(pos_test, preds)
    raw_auc = roc_auc_score(y_test, preds)
    ipw_auc = _ipw_auc(y_test, preds, propensity_test)
    cf_delta, cf_model, cf_orig = _cf_ndcg_delta(test_feeds, y_test, preds, pos_test, k=10)
    pal_r2 = _pal_residual_r2(preds, pos_test)
    fairness[name] = {
        "spearman_r": float(r), "p_value": float(p),
        "raw_auc": float(raw_auc), "ipw_auc": ipw_auc, "ipw_auc_delta": ipw_auc - float(raw_auc),
        "cf_ndcg_model": cf_model, "cf_ndcg_orig": cf_orig, "cf_ndcg_delta": cf_delta,
        "pal_residual_r2": pal_r2,
    }

fairness_df = pd.DataFrame(fairness).T
print("Prediction-Position Correlation (lower |r| = more fair):")
print(fairness_df.to_string())

# Scatter: predicted score vs position for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, preds) in enumerate([("Baseline", pred_base), ("Pos-Aware+Drop", pred_pos), ("IPW-Weighted", pred_ipw)]):
    # Bin by position and show mean predicted score
    tmp = pd.DataFrame({"position": pos_test, "pred": preds})
    pos_avg = tmp.groupby("position")["pred"].mean().reset_index()
    axes[idx].plot(pos_avg["position"], pos_avg["pred"], "o-", markersize=3)
    axes[idx].set_title(f"{name}\nSpearman r={fairness[name]['spearman_r']:.4f}")
    axes[idx].set_xlabel("Position Rank")
    axes[idx].set_ylabel("Mean Predicted Score")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Model | Description | Position at Train | Position at Inference | IPW |
# MAGIC |-------|-------------|:-:|:-:|:-:|
# MAGIC | Baseline | Standard model | ❌ | ❌ | ❌ |
# MAGIC | Pos-Aware+Drop | Learn position, remove at inference | ✅ | ❌ (set to 0) | ❌ |
# MAGIC | Pos-Aware (leak ref) | Upper bound with position leak | ✅ | ✅ | ❌ |
# MAGIC | IPW-Weighted | Reweight by inverse propensity | ❌ | ❌ | ✅ |
# MAGIC
# MAGIC **Key metrics**: AUC (higher=better), LogLoss (lower=better), Position-Prediction Spearman r (closer to 0 = more fair)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5b. Discover-RK-URA Slice — Train & Compare
# MAGIC
# MAGIC Re-train the same 4 models on the subset where `user_flight_ids` contains `discover-rk-ura`,
# MAGIC then compare AUC / LogLoss / Fairness against the full-traffic baseline above.

# COMMAND ----------

URA_FLIGHT = "discover-rk-ura"

ura_mask = pdf["user_flight_ids"].fillna("").str.contains(URA_FLIGHT, regex=False).values
pdf_ura = pdf[ura_mask].reset_index(drop=True)
print(f"URA slice: {len(pdf_ura):,} rows ({len(pdf_ura)/max(len(pdf),1):.2%} of full)")
print(f"URA click rate: {pdf_ura['is_clicked'].mean():.4f}  vs full: {pdf['is_clicked'].mean():.4f}")
print(f"URA feeds: {pdf_ura['feedId'].nunique():,}, users: {pdf_ura['picasso_user_id'].nunique():,}")

assert len(pdf_ura) > 0, f"No samples with user_flight_ids contains '{URA_FLIGHT}'"

# Rebuild feature matrix on the URA slice (reuse same feature_cols, including same cardType dummy columns)
ura_card_dummies = pd.get_dummies(pdf_ura["cardType"], prefix="cardType")
# Align columns with the full-traffic feature_cols (missing dummies → 0)
for c in card_type_dummies.columns:
    if c not in ura_card_dummies.columns:
        ura_card_dummies[c] = 0
ura_card_dummies = ura_card_dummies[card_type_dummies.columns]
pdf_ura_features = pd.concat([pdf_ura[SCORE_FEATURES].reset_index(drop=True),
                              ura_card_dummies.reset_index(drop=True)], axis=1)

X_ura_base = pdf_ura_features.fillna(0).values
X_ura_pos = np.column_stack([X_ura_base, pdf_ura["position_rank"].values])
y_ura = pdf_ura["is_clicked"].values

# Train/test split by feedId on URA slice
ura_req_ids = pdf_ura["feedId"].unique()
ura_train_req, ura_test_req = train_test_split(ura_req_ids, test_size=0.2, random_state=42)
ura_train_mask = pdf_ura["feedId"].isin(ura_train_req).values
ura_test_mask = pdf_ura["feedId"].isin(ura_test_req).values

X_ura_train_base, X_ura_test_base = X_ura_base[ura_train_mask], X_ura_base[ura_test_mask]
X_ura_train_pos, X_ura_test_pos = X_ura_pos[ura_train_mask], X_ura_pos[ura_test_mask]
y_ura_train, y_ura_test = y_ura[ura_train_mask], y_ura[ura_test_mask]
ura_pos_train = pdf_ura["position_rank"].values[ura_train_mask]
ura_pos_test = pdf_ura["position_rank"].values[ura_test_mask]
print(f"URA train: {len(y_ura_train)}, test: {len(y_ura_test)}, "
      f"train ctr={y_ura_train.mean():.4f}, test ctr={y_ura_test.mean():.4f}")

# IPW weights on URA slice (recompute propensity from URA position CTR)
ura_pos_ctr = pdf_ura.groupby("position_rank").agg(
    impressions=("is_clicked", "count"), clicks=("is_clicked", "sum")).reset_index()
ura_pos_ctr["ctr"] = ura_pos_ctr["clicks"] / ura_pos_ctr["impressions"]
ura_pos_ctr_map = ura_pos_ctr.set_index("position_rank")["ctr"]
ura_max_ctr = max(ura_pos_ctr_map.max(), 1e-6)
ura_propensity_train = np.array([ura_pos_ctr_map.get(p, ura_max_ctr) for p in ura_pos_train]) / ura_max_ctr
ura_propensity_train = np.clip(ura_propensity_train, 0.05, 1.0)
ura_ipw_weights = np.where(y_ura_train == 1, 1.0 / ura_propensity_train, 1.0)

# Train 4 models on URA slice
ura_results = {}

m_ura_base = lgb.LGBMClassifier(**LGB_PARAMS)
m_ura_base.fit(X_ura_train_base, y_ura_train, feature_name=base_feature_names)
p_ura_base = m_ura_base.predict_proba(X_ura_test_base)[:, 1]
ura_results["URA-Baseline"] = {"AUC": roc_auc_score(y_ura_test, p_ura_base),
                               "LogLoss": log_loss(y_ura_test, p_ura_base)}

m_ura_pos = lgb.LGBMClassifier(**LGB_PARAMS)
m_ura_pos.fit(X_ura_train_pos, y_ura_train, feature_name=pos_feature_names)
X_ura_test_pos_dropped = X_ura_test_pos.copy()
X_ura_test_pos_dropped[:, -1] = 0
p_ura_pos = m_ura_pos.predict_proba(X_ura_test_pos_dropped)[:, 1]
ura_results["URA-PosAware+Drop"] = {"AUC": roc_auc_score(y_ura_test, p_ura_pos),
                                    "LogLoss": log_loss(y_ura_test, p_ura_pos)}

p_ura_pos_leak = m_ura_pos.predict_proba(X_ura_test_pos)[:, 1]
ura_results["URA-PosAware (leak ref)"] = {"AUC": roc_auc_score(y_ura_test, p_ura_pos_leak),
                                          "LogLoss": log_loss(y_ura_test, p_ura_pos_leak)}

m_ura_ipw = lgb.LGBMClassifier(**LGB_PARAMS)
m_ura_ipw.fit(X_ura_train_base, y_ura_train, sample_weight=ura_ipw_weights, feature_name=base_feature_names)
p_ura_ipw = m_ura_ipw.predict_proba(X_ura_test_base)[:, 1]
ura_results["URA-IPW-Weighted"] = {"AUC": roc_auc_score(y_ura_test, p_ura_ipw),
                                   "LogLoss": log_loss(y_ura_test, p_ura_ipw)}

# Side-by-side comparison: URA slice vs full traffic
combined = {}
combined.update({f"Full · {k}": v for k, v in results.items()})
combined.update({f"URA · {k.replace('URA-', '')}": v for k, v in ura_results.items()})
combined_df = pd.DataFrame(combined).T
print("\n=== Full vs URA-only training comparison ===")
print(combined_df.to_string())

# URA fairness with full debias suite
ura_propensity_test = np.array([ura_pos_ctr_map.get(p, ura_max_ctr) for p in ura_pos_test]) / ura_max_ctr
ura_propensity_test = np.clip(ura_propensity_test, 0.05, 1.0)
ura_test_feeds = pdf_ura["feedId"].values[ura_test_mask]

ura_fairness = {}
for name, preds in [("URA-Baseline", p_ura_base),
                    ("URA-PosAware+Drop", p_ura_pos),
                    ("URA-IPW-Weighted", p_ura_ipw)]:
    r, p = stats.spearmanr(ura_pos_test, preds)
    raw_auc = roc_auc_score(y_ura_test, preds)
    ipw_auc = _ipw_auc(y_ura_test, preds, ura_propensity_test)
    cf_delta, cf_model, cf_orig = _cf_ndcg_delta(ura_test_feeds, y_ura_test, preds, ura_pos_test, k=10)
    pal_r2 = _pal_residual_r2(preds, ura_pos_test)
    ura_fairness[name] = {
        "spearman_r": float(r), "p_value": float(p),
        "raw_auc": float(raw_auc), "ipw_auc": ipw_auc, "ipw_auc_delta": ipw_auc - float(raw_auc),
        "cf_ndcg_model": cf_model, "cf_ndcg_orig": cf_orig, "cf_ndcg_delta": cf_delta,
        "pal_residual_r2": pal_r2,
    }
print("\nURA fairness (4-metric debias suite):")
print(pd.DataFrame(ura_fairness).T.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5c. Single-Feature Ranker on URA test set
# MAGIC
# MAGIC Use each score feature directly as a ranker (no model training) on the URA test split.
# MAGIC Compares NDCG@10 of:
# MAGIC   - feature-as-ranker (descending)
# MAGIC   - original display order
# MAGIC   - random shuffle baseline (avg over 5 seeds)
# MAGIC
# MAGIC Tests whether production score features carry standalone ranking signal on unbiased data.

# COMMAND ----------

from sklearn.metrics import ndcg_score as _ndcg_score
import pandas as pd

ura_test_df = pdf_ura.iloc[ura_test_mask][["feedId", "is_clicked", "position_rank"] + SCORE_FEATURES].copy()
ura_test_df["__orig_pos__"] = ura_test_df["position_rank"]

def _ndcg_per_feed(df, score_col, k=10):
    vals = []
    for fid, g in df.groupby("feedId"):
        if len(g) < 2 or g["is_clicked"].sum() == 0:
            continue
        y = g["is_clicked"].values.reshape(1, -1)
        s = g[score_col].values.reshape(1, -1)
        try:
            vals.append(_ndcg_score(y, s, k=k))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0

# Original display order: lower position_rank = higher score
ura_test_df["__neg_pos__"] = -ura_test_df["position_rank"].astype(float)
ndcg_orig = _ndcg_per_feed(ura_test_df, "__neg_pos__", k=10)

# Random baseline (5 seeds avg)
rng = np.random.default_rng(42)
random_scores = []
for seed in range(5):
    ura_test_df[f"__rand_{seed}__"] = rng.random(len(ura_test_df))
    random_scores.append(_ndcg_per_feed(ura_test_df, f"__rand_{seed}__", k=10))
ndcg_random = float(np.mean(random_scores))

single_feature_ndcg = []
for f in SCORE_FEATURES:
    if ura_test_df[f].nunique() < 2:
        continue
    nd = _ndcg_per_feed(ura_test_df, f, k=10)
    single_feature_ndcg.append({
        "feature": f,
        "ndcg@10": round(nd, 6),
        "delta_vs_orig": round(nd - ndcg_orig, 6),
        "delta_vs_random": round(nd - ndcg_random, 6),
    })

single_feature_ndcg_sorted = sorted(single_feature_ndcg, key=lambda x: -x["ndcg@10"])
print(f"\nNDCG@10 baselines on URA test set:")
print(f"  original display order : {ndcg_orig:.6f}")
print(f"  random shuffle (5-seed): {ndcg_random:.6f}")
print(f"\nSingle-feature ranker NDCG@10 (descending sort):")
print(pd.DataFrame(single_feature_ndcg_sorted).to_string(index=False))

ura_single_feature_ranker = {
    "ndcg_orig_display_order": round(ndcg_orig, 6),
    "ndcg_random_baseline": round(ndcg_random, 6),
    "per_feature": single_feature_ndcg_sorted,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5d. Cross-Eval — Full models on URA test set
# MAGIC
# MAGIC Apply Full-trained models (model_base / model_pos / model_ipw) on the SAME URA test set
# MAGIC used by URA models. This is the only fair AUC comparison — same labels, same features, same rows.
# MAGIC
# MAGIC Answers: does Full model's higher AUC reflect real ranking value, or is it selection-bias mimicry?

# COMMAND ----------

cross_eval = {}
ura_test_feed_ids = pdf_ura["feedId"].values[ura_test_mask]

def _ndcg_for_preds(feed_ids, y_true, preds, k=10):
    df_ = pd.DataFrame({"feedId": feed_ids, "y": y_true, "p": preds})
    vals = []
    for fid, g in df_.groupby("feedId"):
        if len(g) < 2 or g["y"].sum() == 0:
            continue
        try:
            vals.append(_ndcg_score(g["y"].values.reshape(1, -1),
                                    g["p"].values.reshape(1, -1), k=k))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0

# Full Baseline on URA test (uses base feature schema = same as URA-Baseline)
p_full_base_on_ura = model_base.predict_proba(X_ura_test_base)[:, 1]
cross_eval["Full-Baseline @URA-test"] = {
    "AUC": round(roc_auc_score(y_ura_test, p_full_base_on_ura), 6),
    "LogLoss": round(log_loss(y_ura_test, p_full_base_on_ura), 6),
    "NDCG@10": round(_ndcg_for_preds(ura_test_feed_ids, y_ura_test, p_full_base_on_ura), 6),
}

# Full Pos-Aware+Drop on URA test (zero-out position at inference, same as training-time drop)
X_ura_test_pos_drop = X_ura_test_pos.copy()
X_ura_test_pos_drop[:, -1] = 0
p_full_pos_on_ura = model_pos.predict_proba(X_ura_test_pos_drop)[:, 1]
cross_eval["Full-PosAware+Drop @URA-test"] = {
    "AUC": round(roc_auc_score(y_ura_test, p_full_pos_on_ura), 6),
    "LogLoss": round(log_loss(y_ura_test, p_full_pos_on_ura), 6),
    "NDCG@10": round(_ndcg_for_preds(ura_test_feed_ids, y_ura_test, p_full_pos_on_ura), 6),
}

# Full IPW on URA test
p_full_ipw_on_ura = model_ipw.predict_proba(X_ura_test_base)[:, 1]
cross_eval["Full-IPW-Weighted @URA-test"] = {
    "AUC": round(roc_auc_score(y_ura_test, p_full_ipw_on_ura), 6),
    "LogLoss": round(log_loss(y_ura_test, p_full_ipw_on_ura), 6),
    "NDCG@10": round(_ndcg_for_preds(ura_test_feed_ids, y_ura_test, p_full_ipw_on_ura), 6),
}

# URA-trained models on the same test for direct comparison (recompute NDCG with same fn)
cross_eval["URA-Baseline @URA-test"] = {
    "AUC": round(roc_auc_score(y_ura_test, p_ura_base), 6),
    "LogLoss": round(log_loss(y_ura_test, p_ura_base), 6),
    "NDCG@10": round(_ndcg_for_preds(ura_test_feed_ids, y_ura_test, p_ura_base), 6),
}
cross_eval["URA-PosAware+Drop @URA-test"] = {
    "AUC": round(roc_auc_score(y_ura_test, p_ura_pos), 6),
    "LogLoss": round(log_loss(y_ura_test, p_ura_pos), 6),
    "NDCG@10": round(_ndcg_for_preds(ura_test_feed_ids, y_ura_test, p_ura_pos), 6),
}
cross_eval["URA-IPW-Weighted @URA-test"] = {
    "AUC": round(roc_auc_score(y_ura_test, p_ura_ipw), 6),
    "LogLoss": round(log_loss(y_ura_test, p_ura_ipw), 6),
    "NDCG@10": round(_ndcg_for_preds(ura_test_feed_ids, y_ura_test, p_ura_ipw), 6),
}

# Reference: URA original display order NDCG (from 5c) and random baseline
cross_eval["__ref_ndcg_orig_display"] = round(ndcg_orig, 6)
cross_eval["__ref_ndcg_random"] = round(ndcg_random, 6)
cross_eval["__ura_test_size"] = int(len(y_ura_test))
cross_eval["__ura_test_click_rate"] = round(float(y_ura_test.mean()), 6)

print("\n=== Cross-Eval: All models on the SAME URA test set ===")
print(pd.DataFrame(cross_eval).T.to_string())
print(f"\nReference: orig display NDCG@10 = {ndcg_orig:.6f}, random NDCG@10 = {ndcg_random:.6f}")

# COMMAND ----------

# Side-by-side AUC bar chart: Full vs URA
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
model_kinds = ["Baseline", "Pos-Aware+Drop", "Pos-Aware (leak ref)", "IPW-Weighted"]
ura_kinds = ["URA-Baseline", "URA-PosAware+Drop", "URA-PosAware (leak ref)", "URA-IPW-Weighted"]
full_aucs = [results[m]["AUC"] for m in model_kinds]
ura_aucs = [ura_results[m]["AUC"] for m in ura_kinds]
full_losses = [results[m]["LogLoss"] for m in model_kinds]
ura_losses = [ura_results[m]["LogLoss"] for m in ura_kinds]

x = np.arange(len(model_kinds))
w_ = 0.38
axes[0].bar(x - w_/2, full_aucs, w_, label="Full traffic", color="#3498db")
axes[0].bar(x + w_/2, ura_aucs,  w_, label="URA only",     color="#e67e22")
axes[0].set_xticks(x); axes[0].set_xticklabels(model_kinds, rotation=15, ha="right")
axes[0].set_title("AUC: Full vs URA-only training"); axes[0].set_ylabel("AUC"); axes[0].legend()
for i, (a, b) in enumerate(zip(full_aucs, ura_aucs)):
    axes[0].text(i - w_/2, a + 0.001, f"{a:.3f}", ha="center", fontsize=8)
    axes[0].text(i + w_/2, b + 0.001, f"{b:.3f}", ha="center", fontsize=8)

axes[1].bar(x - w_/2, full_losses, w_, label="Full traffic", color="#3498db")
axes[1].bar(x + w_/2, ura_losses,  w_, label="URA only",     color="#e67e22")
axes[1].set_xticks(x); axes[1].set_xticklabels(model_kinds, rotation=15, ha="right")
axes[1].set_title("LogLoss: Full vs URA-only training (lower=better)")
axes[1].set_ylabel("LogLoss"); axes[1].legend()
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5e. URA Unbiasedness Audit (model-free)
# MAGIC
# MAGIC Statistical tests that URA traffic actually achieves item ⊥ position independence.
# MAGIC Compares URA slice vs Production slice (non-URA discover-* traffic) on the SAME pdf.

# COMMAND ----------

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
import json as _json_audit

audit = {}

URA_TAG = "discover-rk-ura"
ura_m  = pdf["user_flight_ids"].fillna("").str.contains(URA_TAG, regex=False).values
prod_m = ~ura_m  # all non-URA discover-* rows

audit["meta"] = {
    "ura_rows":  int(ura_m.sum()),
    "prod_rows": int(prod_m.sum()),
    "ura_unique_feeds":  int(pdf.loc[ura_m,  "feedId"].nunique()),
    "prod_unique_feeds": int(pdf.loc[prod_m, "feedId"].nunique()),
}

def perm_mi(x, y, n_perm=80, rng_seed=0):
    """MI with permutation null. Returns (obs, mean_null, std_null, z, p_value)."""
    rng = np.random.default_rng(rng_seed)
    obs = mutual_info_score(x, y)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        nulls[i] = mutual_info_score(x, rng.permutation(y))
    mu, sd = float(nulls.mean()), float(nulls.std() + 1e-12)
    z = (obs - mu) / sd
    p = float((np.sum(nulls >= obs) + 1) / (n_perm + 1))
    return float(obs), mu, sd, float(z), p

# --- Test A: MI(cardType ; position_rank) — proxy for "item × position" independence ---
def cardType_position_mi(mask, label):
    if mask.sum() < 200:
        return None
    sub = pdf.loc[mask, ["cardType", "position_rank"]].copy()
    # Cap at first 6 positions to keep contingency dense
    sub = sub[sub["position_rank"] < 6]
    ct  = sub["cardType"].astype(str).values
    pos = sub["position_rank"].astype(int).values
    obs, mu, sd, z, p = perm_mi(ct, pos)
    # Also χ² for cross-check
    ct_codes  = pd.factorize(ct)[0]
    ctab = pd.crosstab(ct_codes, pos)
    chi2, chi_p, dof, _ = chi2_contingency(ctab)
    return {
        "mi_obs": round(obs, 6), "mi_null_mean": round(mu, 6),
        "mi_null_std": round(sd, 6), "mi_z": round(z, 3), "mi_perm_p": round(p, 4),
        "chi2": round(float(chi2), 3), "chi2_p": float(f"{chi_p:.3e}"), "chi2_dof": int(dof),
        "n": int(len(sub)), "n_card_types": int(len(set(ct)))
    }

audit["cardType_x_position"] = {
    "URA":        cardType_position_mi(ura_m,  "URA"),
    "Production": cardType_position_mi(prod_m, "Production"),
    "interpretation": "Lower MI / lower mi_z / chi2_p > 0.05 = item ⊥ position. URA should be ≈ random; Production should show strong dependence."
}

# --- Test B: MI(score_feature_binned ; position_rank) per feature ---
def score_position_mi(mask):
    if mask.sum() < 200:
        return None
    sub = pdf.loc[mask, SCORE_FEATURES + ["position_rank"]].copy()
    sub = sub[sub["position_rank"] < 6]
    pos = sub["position_rank"].astype(int).values
    out = {}
    for f in SCORE_FEATURES:
        v = sub[f].values
        if np.std(v) < 1e-9 or len(np.unique(v)) < 4:
            continue
        try:
            v_bin = pd.qcut(v, q=8, duplicates="drop", labels=False).astype(int).values
        except Exception:
            continue
        if len(np.unique(v_bin)) < 2:
            continue
        obs, mu, sd, z, p = perm_mi(v_bin, pos, n_perm=50)
        out[f] = {"mi_obs": round(obs, 6), "mi_z": round(z, 2), "mi_perm_p": round(p, 4)}
    return out

audit["score_x_position"] = {
    "URA":        score_position_mi(ura_m),
    "Production": score_position_mi(prod_m),
    "interpretation": "URA: production-derived scores should NOT predict position (z low). Production: high z = score drives position assignment (selection bias)."
}

# --- Test C: position-wise CTR curve ---
def pos_ctr_curve(mask):
    if mask.sum() < 100:
        return None
    sub = pdf.loc[mask].groupby("position_rank")["is_clicked"].agg(["mean", "count"]).reset_index()
    sub = sub[sub["position_rank"] < 15]
    return {
        "positions":  sub["position_rank"].astype(int).tolist(),
        "ctrs":       [round(float(x), 5) for x in sub["mean"].tolist()],
        "counts":     sub["count"].astype(int).tolist(),
    }

audit["position_ctr"] = {
    "URA":        pos_ctr_curve(ura_m),
    "Production": pos_ctr_curve(prod_m),
    "interpretation": "URA should show flatter CTR-vs-position (residual = pure visibility/eye-tracking). Production decays steeply (CTR ratio pos1/pos20 often 5-10x)."
}

# --- Test D: cardType coverage / entropy ---
def coverage_stats(mask):
    if mask.sum() < 100:
        return None
    s = pdf.loc[mask, "cardType"].astype(str)
    counts = s.value_counts()
    p = (counts / counts.sum()).values
    H = float(-np.sum(p * np.log2(p + 1e-12)))
    Hmax = float(np.log2(len(counts))) if len(counts) > 1 else 0.0
    return {
        "n_unique_cardTypes": int(len(counts)),
        "entropy_bits": round(H, 4),
        "entropy_max":  round(Hmax, 4),
        "entropy_ratio": round(H / Hmax, 4) if Hmax > 0 else None,
        "top5_share": round(float(counts.head(5).sum() / counts.sum()), 4),
    }

audit["cardType_coverage"] = {
    "URA":        coverage_stats(ura_m),
    "Production": coverage_stats(prod_m),
    "interpretation": "URA should have higher entropy_ratio + lower top5_share (uniform random promotes long-tail item exposure)."
}

print("\n=== URA Unbiasedness Audit ===")
print(_json_audit.dumps(audit, indent=2)[:3000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Export Results to JSON

# COMMAND ----------

import json

# --- Feature Coverage ---
coverage = {}
for f in SCORE_FEATURES:
    total = len(pdf)
    non_zero = int((pdf[f] != 0).sum())
    non_null = int(pdf[f].notna().sum())
    coverage[f] = {
        "total": total,
        "non_null": non_null,
        "non_zero": non_zero,
        "null_rate": round(1 - non_null / total, 4) if total > 0 else None,
        "zero_rate": round(1 - non_zero / total, 4) if total > 0 else None,
        "coverage_rate": round(non_zero / total, 4) if total > 0 else None,
        "mean": round(float(pdf[f].mean()), 6),
        "std": round(float(pdf[f].std()), 6),
        "min": round(float(pdf[f].min()), 6),
        "max": round(float(pdf[f].max()), 6),
        "median": round(float(pdf[f].median()), 6),
    }

# --- Assemble all results ---
export_data = {
    "metadata": {
        "table": "mai_ws_discover.analytics.ods_doca_ranking_ofe_wide_table",
        "run_id": "20260420",
        "total_rows": len(pdf),
        "click_rate": round(float(pdf["is_clicked"].mean()), 6),
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "position_range": [int(pdf["position_rank"].min()), int(pdf["position_rank"].max())],
        "num_sessions": int(pdf["feedId"].nunique()),
        "num_users": int(pdf["picasso_user_id"].nunique()),
    },
    "feature_coverage": coverage,
    "correlation": corr_df.to_dict(orient="records"),
    "correlation_ura": corr_df_ura.to_dict(orient="records"),
    "position_ctr": pos_ctr.to_dict(orient="records"),
    "model_results": results,
    "ura_model_results": ura_results,
    "ura_fairness": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in ura_fairness.items()},
    "ura_single_feature_ranker": ura_single_feature_ranker,
    "cross_eval_on_ura_test": cross_eval,
    "ura_unbiasedness_audit": audit,
    "ura_metadata": {
        "ura_flight": URA_FLIGHT,
        "ura_rows": int(len(pdf_ura)),
        "ura_share_of_full": round(len(pdf_ura) / max(len(pdf), 1), 6),
        "ura_click_rate": round(float(pdf_ura["is_clicked"].mean()), 6) if len(pdf_ura) else None,
        "ura_feeds": int(pdf_ura["feedId"].nunique()) if len(pdf_ura) else 0,
        "ura_users": int(pdf_ura["picasso_user_id"].nunique()) if len(pdf_ura) else 0,
    },
    "fairness": {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in fairness.items()},
    "feature_importance": {},
}

# Feature importance per model
for name, model, fnames in [
    ("Baseline", model_base, base_feature_names),
    ("Pos-Aware+Drop", model_pos, pos_feature_names),
    ("IPW-Weighted", model_ipw, base_feature_names),
]:
    imp = model.feature_importances_.tolist()
    export_data["feature_importance"][name] = [
        {"feature": f, "importance": round(v, 4)} for f, v in sorted(zip(fnames, imp), key=lambda x: -x[1])
    ]

# Write to DBFS
output_path = "/dbfs/tmp/penghu/ranking_feature_analysis_results.json"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(export_data, f, indent=2, default=str)

print(f"Exported to {output_path}")
print(f"Keys: {list(export_data.keys())}")
print(f"Feature coverage: {len(coverage)} features")
print(f"Correlation entries: {len(export_data['correlation'])}")
print(f"Model results: {list(results.keys())}")