"""Download a single day's parquet from Databricks and preprocess to eval JSONL.

Usage:
    python download_and_prep_day.py 20260421          # outputs data_v11/eval_ura_20260421.jsonl
    python download_and_prep_day.py 20260421 --out_dir data_v11
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime


# ── Databricks config ─────────────────────────────────────────────────────
HOST = "adb-3355567219430035.15.azuredatabricks.net"
HTTP_PATH = "/sql/1.0/warehouses/3d5effb8e09bd9e7"
TABLE = "mai_ws_discover.analytics.ods_doca_feed_grounded_v8_partitioned"

# Token: env var > SLM/aad_token > ../aad_token
_HERE = os.path.dirname(os.path.abspath(__file__))
def _read_token():
    if os.environ.get("DATABRICKS_TOKEN"):
        return os.environ["DATABRICKS_TOKEN"]
    for p in [
        os.path.join(_HERE, "aad_token"),
        os.path.join(_HERE, "..", "aad_token"),
    ]:
        if os.path.isfile(p):
            return open(p).read().strip()
    raise FileNotFoundError(
        "No DATABRICKS_TOKEN env var and no aad_token file found. "
        "Run: export DATABRICKS_TOKEN=$(cat amlSettings/aad_token)"
    )


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def download_parquet(bizdate: str, out_dir: str) -> str:
    """Download parquet for bizdate into out_dir. Returns path to parquet file."""
    import pyarrow.parquet as pq
    from databricks import sql as dbsql

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"v8_grounded_{bizdate}.parquet")
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        print(f"[{_ts()}] skip download — {out} exists ({os.path.getsize(out)/1e6:.1f} MB)")
        return out

    token = _read_token()
    q = f"SELECT * FROM {TABLE} WHERE bizdate='{bizdate}'"
    print(f"[{_ts()}] downloading bizdate={bizdate} from Databricks ...")
    t0 = time.time()
    with dbsql.connect(
        server_hostname=HOST,
        http_path=HTTP_PATH,
        access_token=token,
        auth_type="access-token",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            table = cur.fetchall_arrow()
    print(f"[{_ts()}] fetched rows={table.num_rows} cols={table.num_columns} "
          f"({time.time()-t0:.1f}s)")
    tmp = out + ".tmp"
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, out)
    print(f"[{_ts()}] wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return out


def preprocess_day(parquet_path: str, bizdate: str, out_dir: str) -> str:
    """Extract URA eval samples from one parquet day into a JSONL file."""
    sys.path.insert(0, _HERE)
    from preprocess_data import extract_samples, save_jsonl  # type: ignore

    out_jsonl = os.path.join(out_dir, f"eval_ura_{bizdate}.jsonl")
    if os.path.isfile(out_jsonl) and os.path.getsize(out_jsonl) > 0:
        n = sum(1 for _ in open(out_jsonl))
        print(f"[{_ts()}] skip preprocess — {out_jsonl} exists ({n} samples)")
        return out_jsonl

    print(f"[{_ts()}] preprocessing {parquet_path} ...")
    samples = extract_samples(
        parquet_path,
        bizdate_min=bizdate,
        bizdate_max=bizdate,
        flight_filter="discover-rk-ura",
        require_features=True,
    )
    print(f"[{_ts()}] extracted {len(samples)} URA samples for {bizdate}")
    os.makedirs(out_dir, exist_ok=True)
    save_jsonl(samples, out_jsonl)
    print(f"[{_ts()}] wrote {out_jsonl}")
    return out_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bizdate", help="e.g. 20260421")
    ap.add_argument("--raw_dir", default="data",
                    help="directory to save downloaded parquet")
    ap.add_argument("--out_dir", default="data_v11",
                    help="directory to save preprocessed JSONL")
    args = ap.parse_args()

    parquet = download_parquet(args.bizdate, args.raw_dir)
    jsonl   = preprocess_day(parquet, args.bizdate, args.out_dir)
    print(f"\n[done] eval JSONL: {jsonl}")


if __name__ == "__main__":
    main()
