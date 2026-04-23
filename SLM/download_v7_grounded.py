"""Download v8_grounded for a range of bizdates from Databricks SQL Warehouse to local parquet."""
import os
import sys
import time
from datetime import date, timedelta
from databricks import sql
import pyarrow.parquet as pq

HOST = "adb-3355567219430035.15.azuredatabricks.net"
HTTP_PATH = "/sql/1.0/warehouses/3d5effb8e09bd9e7"

_TOKEN_FILE = os.path.join(os.path.dirname(__file__), "aad_token")
TOKEN = os.environ.get("DATABRICKS_TOKEN") or open(_TOKEN_FILE).read().strip()

OUT_DIR = "/scratch/azureml/cr/j/cb7f3b2f13af4de88e98a157ca0e3eaa/exe/wd/amlSettings/data_v8"
os.makedirs(OUT_DIR, exist_ok=True)

# Date range: 2026-03-30 .. 2026-04-20 (inclusive). Override with CLI args: START END (YYYYMMDD).
if len(sys.argv) >= 3:
    start = date(int(sys.argv[1][:4]), int(sys.argv[1][4:6]), int(sys.argv[1][6:8]))
    end = date(int(sys.argv[2][:4]), int(sys.argv[2][4:6]), int(sys.argv[2][6:8]))
else:
    start = date(2026, 3, 30)
    end = date(2026, 4, 20)

bizdates = []
d = start
while d <= end:
    bizdates.append(d.strftime("%Y%m%d"))
    d += timedelta(days=1)

print(f"[info] bizdates: {bizdates[0]} .. {bizdates[-1]} ({len(bizdates)} days)")

with sql.connect(
    server_hostname=HOST,
    http_path=HTTP_PATH,
    access_token=TOKEN,
    auth_type="access-token",
) as conn:
    with conn.cursor() as cur:
        for bd in bizdates:
            out = os.path.join(OUT_DIR, f"v8_grounded_{bd}.parquet")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                print(f"[skip] {out} exists ({os.path.getsize(out)/1e6:.2f} MB)")
                continue
            q = (
                "SELECT * FROM mai_ws_discover.analytics."
                f"ods_doca_feed_grounded_v8_partitioned WHERE bizdate='{bd}'"
            )
            t0 = time.time()
            print(f"[info] {bd}: running query...")
            cur.execute(q)
            print(f"[info] {bd}: fetching arrow table...")
            table = cur.fetchall_arrow()
            print(f"[info] {bd}: rows={table.num_rows} cols={table.num_columns} "
                  f"({time.time()-t0:.1f}s)")
            tmp = out + ".tmp"
            pq.write_table(table, tmp, compression="snappy")
            os.replace(tmp, out)
            print(f"[done] {bd}: wrote {out}  size={os.path.getsize(out)/1e6:.2f} MB")

print("[all done]")
