#!/usr/bin/env python3
"""Compare v9 JSONL data with raw parquet for data consistency checking.

Picks several user_ids from v9 eval_ura.jsonl and loads the same users
from raw parquet, then compares field-by-field.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pyarrow.parquet as pq
from data import (
    _safe_json, _is_impression, _all_interests,
    _shown_titles, _clicked_titles, _group_conversations,
    _resolve_parquet_paths, load_samples, load_samples_jsonl,
    build_prompt_budgeted,
)
from collections import defaultdict

# ── 1. Pick user_ids from v9 JSONL ──────────────────────────────────
print("=" * 80)
print("STEP 1: Pick user_ids from v9 eval_ura.jsonl")
print("=" * 80)

# Get a few diverse users: one with history, one without, one with features
v9_samples = []
with open("data_v9/eval_ura.jsonl") as f:
    for line in f:
        s = json.loads(line.strip())
        v9_samples.append(s)

# Group by user_id
by_user = defaultdict(list)
for s in v9_samples:
    by_user[s["user_id"]].append(s)

# Pick users: 1 with history + features, 1 without history, 1 with many samples
users_with_hist = [(uid, ss) for uid, ss in by_user.items()
                   if any(len(s.get("history", [])) > 0 for s in ss)
                   and any(s.get("features") for s in ss)]
users_no_hist = [(uid, ss) for uid, ss in by_user.items()
                 if all(len(s.get("history", [])) == 0 for s in ss)]
users_many = sorted(by_user.items(), key=lambda x: -len(x[1]))

selected_uids = []
if users_with_hist:
    selected_uids.append(users_with_hist[0][0])
if users_no_hist:
    selected_uids.append(users_no_hist[0][0])
if users_many and users_many[0][0] not in selected_uids:
    selected_uids.append(users_many[0][0])
# Add one more random
for uid in by_user:
    if uid not in selected_uids:
        selected_uids.append(uid)
        break

print(f"Selected {len(selected_uids)} users:")
for uid in selected_uids:
    ss = by_user[uid]
    has_hist = any(len(s.get("history", [])) > 0 for s in ss)
    has_feat = any(s.get("features") for s in ss)
    has_conv = any(len(s.get("interests", {}).get("conversations", [])) > 0 for s in ss)
    print(f"  {uid}: {len(ss)} samples, hist={has_hist}, feat={has_feat}, conv={has_conv}")

# ── 2. Load same users from raw parquet ────────────────────────────
print("\n" + "=" * 80)
print("STEP 2: Load same users from raw parquet via load_samples()")
print("=" * 80)

# Load all eval data from parquet (same params as v9)
pq_samples = load_samples(
    "data",
    max_history=30,
    include_conv=True,
    flight_filter="discover-rk-ura",
    require_features=True,
    bizdate_min="20260417",
)

pq_by_user = defaultdict(list)
for s in pq_samples:
    pq_by_user[s["user_id"]].append(s)

print(f"Parquet total: {len(pq_samples)} samples, {len(pq_by_user)} users")

# ── 3. Compare per user ────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 3: Field-by-field comparison")
print("=" * 80)

for uid in selected_uids:
    v9_ss = by_user.get(uid, [])
    pq_ss = pq_by_user.get(uid, [])

    print(f"\n{'─' * 70}")
    print(f"USER: {uid}")
    print(f"  v9 JSONL samples: {len(v9_ss)}")
    print(f"  Parquet samples:  {len(pq_ss)}")

    if not pq_ss:
        print(f"  ⚠ User not found in parquet! Possible reasons:")
        print(f"    - Different bizdate range")
        print(f"    - Empty interests filtered in v9 but not matched here")
        # Try loading without flight_filter
        continue

    # Match samples by (feed_id, candidate itemid)
    v9_by_key = {}
    for s in v9_ss:
        key = (s["feed_id"], s["candidate"]["itemid"])
        v9_by_key[key] = s

    pq_by_key = {}
    for s in pq_ss:
        key = (s["feed_id"], s["candidate"]["itemid"])
        pq_by_key[key] = s

    all_keys = set(v9_by_key.keys()) | set(pq_by_key.keys())
    only_v9 = set(v9_by_key.keys()) - set(pq_by_key.keys())
    only_pq = set(pq_by_key.keys()) - set(v9_by_key.keys())
    both = set(v9_by_key.keys()) & set(pq_by_key.keys())

    print(f"  Keys in both:    {len(both)}")
    print(f"  Only in v9:      {len(only_v9)}")
    print(f"  Only in parquet: {len(only_pq)}")

    if only_v9:
        print(f"  ⚠ Samples only in v9 (first 3):")
        for k in list(only_v9)[:3]:
            print(f"    feed={k[0]}, item={k[1]}")

    if only_pq:
        print(f"  ⚠ Samples only in parquet (first 3):")
        for k in list(only_pq)[:3]:
            s = pq_by_key[k]
            has_pi = len(s["interests"]["positive"]) > 0
            print(f"    feed={k[0]}, item={k[1]}, has_pos_interests={has_pi}")

    # Detailed comparison on first matched sample
    if both:
        key = sorted(both)[0]
        v = v9_by_key[key]
        p = pq_by_key[key]

        print(f"\n  === Detailed comparison for feed={key[0]}, item={key[1]} ===")

        # Top-level fields
        for field in ["feed_id", "user_id", "bizdate", "label"]:
            vv = v.get(field)
            pp = p.get(field)
            match = "✓" if vv == pp else "✗"
            if vv != pp:
                print(f"  [{match}] {field}: v9={vv!r}, pq={pp!r}")

        # is_ura
        v_ura = v.get("is_ura")
        # parquet has flight_ids instead
        p_ura = 1 if "discover-rk-ura" in (p.get("flight_ids") or "") else 0
        match = "✓" if v_ura == p_ura else "✗"
        if v_ura != p_ura:
            print(f"  [{match}] is_ura: v9={v_ura}, pq={p_ura}")

        # Candidate
        for ck in ["itemid", "title", "summary"]:
            vv = v["candidate"].get(ck, "")
            pp = p["candidate"].get(ck, "")
            match = "✓" if vv == pp else "✗"
            if vv != pp:
                print(f"  [{match}] candidate.{ck}: v9={vv[:60]!r}, pq={pp[:60]!r}")

        # Interests positive
        v_pos = v.get("interests", {}).get("positive", [])
        p_pos = p.get("interests", {}).get("positive", [])
        match = "✓" if v_pos == p_pos else "✗"
        print(f"  [{match}] interests.positive: v9={len(v_pos)} items, pq={len(p_pos)} items")
        if v_pos != p_pos:
            # Show first difference
            for i in range(max(len(v_pos), len(p_pos))):
                vi = v_pos[i][:80] if i < len(v_pos) else "(missing)"
                pi = p_pos[i][:80] if i < len(p_pos) else "(missing)"
                if vi != pi:
                    print(f"      diff at [{i}]: v9={vi!r}")
                    print(f"                   pq={pi!r}")
                    break

        # Interests negative
        v_neg = v.get("interests", {}).get("negative", [])
        p_neg = p.get("interests", {}).get("negative", [])
        match = "✓" if v_neg == p_neg else "✗"
        print(f"  [{match}] interests.negative: v9={len(v_neg)} items, pq={len(p_neg)} items")

        # Conversations
        v_conv = v.get("interests", {}).get("conversations", [])
        p_conv = p.get("interests", {}).get("conversations", [])
        match = "✓" if len(v_conv) == len(p_conv) else "✗"
        print(f"  [{match}] conversations: v9={len(v_conv)} groups, pq={len(p_conv)} groups")
        if v_conv and p_conv:
            v_msg_count = sum(len(g.get("messages", [])) for g in v_conv)
            p_msg_count = sum(len(g.get("messages", [])) for g in p_conv)
            match = "✓" if v_msg_count == p_msg_count else "✗"
            print(f"  [{match}] conv total msgs: v9={v_msg_count}, pq={p_msg_count}")
            # Compare first group
            if v_conv[0] != p_conv[0]:
                print(f"      v9 first group id={v_conv[0].get('id','')[:30]}, msgs={len(v_conv[0].get('messages',[]))}")
                print(f"      pq first group id={p_conv[0].get('id','')[:30]}, msgs={len(p_conv[0].get('messages',[]))}")

        # Interactions (clicks, thumbsUp, thumbsDown)
        v_inter = v.get("interests", {}).get("interactions", {})
        p_inter = p.get("interests", {}).get("interactions", {})
        for ik in ["clicks", "thumbsUp", "thumbsDown"]:
            vv = v_inter.get(ik, [])
            pp = p_inter.get(ik, [])
            match = "✓" if vv == pp else "✗"
            if vv != pp:
                print(f"  [{match}] interactions.{ik}: v9={len(vv)} items, pq={len(pp)} items")

        # History
        v_hist = v.get("history", [])
        p_hist = p.get("history", [])
        match = "✓" if len(v_hist) == len(p_hist) else "✗"
        print(f"  [{match}] history: v9={len(v_hist)} items, pq={len(p_hist)} items")
        if v_hist != p_hist:
            # Check if titles match
            v_titles = [h.get("title", "") for h in v_hist]
            p_titles = [h.get("title", "") for h in p_hist]
            if v_titles != p_titles:
                for i in range(min(3, max(len(v_titles), len(p_titles)))):
                    vt = v_titles[i][:60] if i < len(v_titles) else "(missing)"
                    pt = p_titles[i][:60] if i < len(p_titles) else "(missing)"
                    if vt != pt:
                        print(f"      diff at [{i}]: v9={vt!r}")
                        print(f"                   pq={pt!r}")

        # Features
        v_feat = v.get("features")
        p_feat = p.get("features")
        if v_feat is None and p_feat is None:
            print(f"  [✓] features: both None")
        elif v_feat is None or p_feat is None:
            print(f"  [✗] features: v9={'None' if v_feat is None else 'dict'}, pq={'None' if p_feat is None else 'dict'}")
        else:
            v_keys = set(v_feat.keys())
            p_keys = set(p_feat.keys())
            match = "✓" if v_keys == p_keys else "✗"
            print(f"  [{match}] features keys: v9={sorted(v_keys)}, pq={sorted(p_keys)}")
            for fk in sorted(v_keys & p_keys):
                vv = v_feat[fk]
                pp = p_feat[fk]
                if vv != pp:
                    print(f"      [{fk}]: v9={vv}, pq={pp}")

# ── 4. Prompt comparison ───────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 4: Compare generated prompts (v9 JSONL vs parquet)")
print("=" * 80)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

for uid in selected_uids[:2]:  # just first 2 users
    v9_ss = by_user.get(uid, [])
    pq_ss = pq_by_user.get(uid, [])
    if not v9_ss or not pq_ss:
        continue

    # Find first matched sample
    v9_by_key = {(s["feed_id"], s["candidate"]["itemid"]): s for s in v9_ss}
    pq_by_key = {(s["feed_id"], s["candidate"]["itemid"]): s for s in pq_ss}
    both = set(v9_by_key.keys()) & set(pq_by_key.keys())
    if not both:
        continue
    key = sorted(both)[0]
    v = v9_by_key[key]
    p = pq_by_key[key]

    print(f"\n{'─' * 70}")
    print(f"USER: {uid}, feed={key[0]}, item={key[1]}")

    v_prompt, _, _, _ = build_prompt_budgeted(v["history"], v["interests"], v["candidate"], tokenizer, max_body_tokens=4096)
    p_prompt, _, _, _ = build_prompt_budgeted(p["history"], p["interests"], p["candidate"], tokenizer, max_body_tokens=4096)

    if v_prompt == p_prompt:
        print("  [✓] Prompts are IDENTICAL")
    else:
        print(f"  [✗] Prompts DIFFER")
        print(f"      v9 len={len(v_prompt)}, pq len={len(p_prompt)}")
        # Find first difference
        lines_v = v_prompt.split("\n")
        lines_p = p_prompt.split("\n")
        n_diff = 0
        for i in range(max(len(lines_v), len(lines_p))):
            lv = lines_v[i] if i < len(lines_v) else "(end)"
            lp = lines_p[i] if i < len(lines_p) else "(end)"
            if lv != lp:
                n_diff += 1
                if n_diff <= 5:
                    print(f"      line {i}: v9={lv[:100]!r}")
                    print(f"               pq={lp[:100]!r}")
        print(f"      total differing lines: {n_diff} / {max(len(lines_v), len(lines_p))}")

# ── 5. Summary stats ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 5: Global consistency check")
print("=" * 80)

# Check: how many parquet samples have empty positive interests?
n_total_pq = len(pq_samples)
n_empty_pos = sum(1 for s in pq_samples if not s["interests"]["positive"])
print(f"Parquet samples with empty positive interests: {n_empty_pos}/{n_total_pq} ({100*n_empty_pos/max(1,n_total_pq):.1f}%)")
print(f"  (These are filtered OUT in v9 JSONL)")

# Check: v9 vs parquet sample counts per user
v9_user_counts = {uid: len(ss) for uid, ss in by_user.items()}
pq_user_counts = {uid: len(ss) for uid, ss in pq_by_user.items()}

users_only_v9 = set(v9_user_counts.keys()) - set(pq_user_counts.keys())
users_only_pq = set(pq_user_counts.keys()) - set(v9_user_counts.keys())
users_both = set(v9_user_counts.keys()) & set(pq_user_counts.keys())

print(f"\nUser overlap: both={len(users_both)}, only_v9={len(users_only_v9)}, only_pq={len(users_only_pq)}")

# For users in both, check count match
n_exact = 0
n_v9_more = 0
n_pq_more = 0
for uid in users_both:
    vc = v9_user_counts[uid]
    pc = pq_user_counts[uid]
    if vc == pc:
        n_exact += 1
    elif vc > pc:
        n_v9_more += 1
    else:
        n_pq_more += 1

print(f"Sample count comparison (for {len(users_both)} shared users):")
print(f"  Exact match:     {n_exact}")
print(f"  v9 has more:     {n_v9_more}")
print(f"  parquet has more: {n_pq_more}")

if users_only_pq:
    # These should be users with empty interests
    print(f"\nUsers only in parquet (first 5):")
    for uid in list(users_only_pq)[:5]:
        ss = pq_by_user[uid]
        empty = sum(1 for s in ss if not s["interests"]["positive"])
        print(f"  {uid}: {len(ss)} samples, {empty} with empty pos interests")

print("\nDONE")
