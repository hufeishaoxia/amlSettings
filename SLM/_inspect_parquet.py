import pyarrow.parquet as pq, glob, json

files = sorted(glob.glob('data/**/*.parquet', recursive=True))
print(f'Total parquet files: {len(files)}')

for f in files[:3]:
    print(f'\n=== {f} ===')
    pf = pq.ParquetFile(f)
    print(f'schema: {pf.schema_arrow.names}')
    print(f'num_row_groups: {pf.metadata.num_row_groups}')
    t = pf.read_row_group(0, columns=pf.schema_arrow.names[:])
    print(f'rows in rg0: {len(t)}')
    row = t.to_pylist()[0]
    for k, v in row.items():
        val_str = str(v)[:200] if v else str(v)
        print(f'  {k}: type={type(v).__name__}  preview={val_str}')
