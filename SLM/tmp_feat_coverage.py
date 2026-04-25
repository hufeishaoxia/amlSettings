#!/usr/bin/env python3
"""Analyze feature coverage in data_v9 JSONL files."""
import json, os
from collections import Counter

def analyze_file(path):
    total = 0
    field_present = Counter()
    field_nonempty = Counter()
    feature_keys_counter = Counter()
    feature_nonempty = Counter()
    interest_keys_counter = Counter()
    sample = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  JSON ERROR at line {total+1}: {e}")
                continue
            total += 1
            if sample is None:
                sample = d

            for k, v in d.items():
                field_present[k] += 1
                if v is not None and v != "" and v != [] and v != {}:
                    field_nonempty[k] += 1

            feat = d.get("features", {})
            if isinstance(feat, dict):
                for fk, fv in feat.items():
                    feature_keys_counter[fk] += 1
                    if fv is not None and fv != "" and fv != 0 and fv != [] and fv != {} and fv != "0":
                        feature_nonempty[fk] += 1

            interests = d.get("interests", {})
            if isinstance(interests, dict):
                for ik in interests:
                    interest_keys_counter[ik] += 1

    return total, field_present, field_nonempty, feature_keys_counter, feature_nonempty, interest_keys_counter, sample

data_dir = "data_v9"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".jsonl")])

for fname in files:
    path = os.path.join(data_dir, fname)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    print(f"{'='*60}")

    total, field_present, field_nonempty, feat_keys, feat_nonempty, interest_keys, sample = analyze_file(path)
    print(f"Total samples: {total}")

    if sample:
        print(f"Top-level keys: {list(sample.keys())}")

    print(f"\n--- Top-level field coverage ---")
    all_keys = sorted(set(list(field_present.keys()) + list(field_nonempty.keys())))
    print(f"{'Field':<20} {'Present':>10} {'Non-empty':>10} {'Coverage%':>10}")
    for k in all_keys:
        p = field_present.get(k, 0)
        ne = field_nonempty.get(k, 0)
        pct = ne / total * 100 if total > 0 else 0
        print(f"{k:<20} {p:>10} {ne:>10} {pct:>9.1f}%")

    if feat_keys:
        print(f"\n--- Features sub-keys coverage ---")
        print(f"{'Feature':<40} {'Present':>10} {'Non-empty':>10} {'Coverage%':>10}")
        for k in sorted(feat_keys.keys()):
            cnt = feat_keys[k]
            ne = feat_nonempty.get(k, 0)
            pct = ne / total * 100 if total > 0 else 0
            print(f"{k:<40} {cnt:>10} {ne:>10} {pct:>9.1f}%")

    if interest_keys:
        print(f"\n--- Interests sub-keys coverage ---")
        print(f"{'Interest field':<30} {'Present':>10} {'Coverage%':>10}")
        for k in sorted(interest_keys.keys()):
            cnt = interest_keys[k]
            pct = cnt / total * 100 if total > 0 else 0
            print(f"{k:<30} {cnt:>10} {pct:>9.1f}%")

    if fname == files[0] and sample:
        print(f"\n--- Sample features (first record) ---")
        feat = sample.get("features", {})
        if isinstance(feat, dict):
            for k, v in sorted(feat.items()):
                val_str = str(v)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"  {k}: {val_str}")

        print(f"\n--- Sample interests structure (first record) ---")
        interests = sample.get("interests", {})
        if isinstance(interests, dict):
            for k, v in sorted(interests.items()):
                if isinstance(v, list):
                    print(f"  {k}: list[{len(v)}]")
                    if v:
                        print(f"    first: {str(v[0])[:120]}")
                elif isinstance(v, str):
                    print(f"  {k}: str[{len(v)}] = {v[:120]}")
                else:
                    print(f"  {k}: {str(v)[:120]}")

        print(f"\n--- History (first record) ---")
        hist = sample.get("history", [])
        print(f"  length: {len(hist)}")
        if hist:
            print(f"  first: {str(hist[0])[:150]}")

print("\nDONE")
