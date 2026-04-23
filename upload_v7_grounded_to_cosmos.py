"""
Upload mai_ws_discover.analytics.ods_doca_feed_grounded_v7_partitioned (bizdate=20260420)
to Cosmos DB using the Databricks Spark Cosmos OLTP connector (Approach A).

Run on a Databricks cluster with the Maven library installed:
    com.azure.cosmos.spark:azure-cosmos-spark_3-5_2-12:4.36.0

Submit from the repo root:
    databricks workspace import_dir . /Repos/<you>/amlSettings
    databricks jobs submit --python-file upload_v7_grounded_to_cosmos.py
or just run it as a notebook cell.

Required env / secrets (use Databricks secret scope in production):
    COSMOS_ENDPOINT   e.g. https://<acct>.documents.azure.com:443/
    COSMOS_KEY        primary key
    COSMOS_DATABASE   default: doca
    COSMOS_CONTAINER  default: v7_grounded_20260420
    PARTITION_KEY     column to use as /<pk>; default: conversation_id
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


SOURCE_TABLE = "mai_ws_discover.analytics.ods_doca_feed_grounded_v7_partitioned"
BIZDATE = "20260420"


def get_cfg() -> dict:
    endpoint = os.environ.get("COSMOS_ENDPOINT")
    key = os.environ.get("COSMOS_KEY")
    if not endpoint or not key:
        sys.exit("ERROR: set COSMOS_ENDPOINT and COSMOS_KEY env vars (or Databricks secrets).")

    database = os.environ.get("COSMOS_DATABASE", "doca")
    container = os.environ.get("COSMOS_CONTAINER", f"v7_grounded_{BIZDATE}")
    partition_key = os.environ.get("PARTITION_KEY", "conversation_id")

    return {
        "endpoint": endpoint,
        "key": key,
        "database": database,
        "container": container,
        "partition_key": partition_key,
    }


def ensure_container(cfg: dict) -> None:
    """Create database + container if they don't exist. Uses Catalog API."""
    spark = SparkSession.getActiveSession()
    spark.conf.set("spark.sql.catalog.cosmosCatalog", "com.azure.cosmos.spark.CosmosCatalog")
    spark.conf.set("spark.sql.catalog.cosmosCatalog.spark.cosmos.accountEndpoint", cfg["endpoint"])
    spark.conf.set("spark.sql.catalog.cosmosCatalog.spark.cosmos.accountKey", cfg["key"])

    spark.sql(f"CREATE DATABASE IF NOT EXISTS cosmosCatalog.{cfg['database']};")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS cosmosCatalog.{cfg['database']}.{cfg['container']}
        USING cosmos.oltp
        TBLPROPERTIES(
            partitionKeyPath = '/{cfg['partition_key']}',
            manualThroughput = '4000',
            indexingPolicy   = 'AllProperties',
            defaultTtlInSeconds = '-1'
        );
        """
    )


def main() -> None:
    cfg = get_cfg()
    spark = SparkSession.builder.appName("upload_v7_grounded_to_cosmos").getOrCreate()

    ensure_container(cfg)

    df = (
        spark.table(SOURCE_TABLE)
        .where(F.col("bizdate") == BIZDATE)
    )

    # Cosmos requires a string `id`. Prefer conversation_id if present, else uuid.
    if "id" not in df.columns:
        if "conversation_id" in df.columns:
            df = df.withColumn("id", F.col("conversation_id").cast("string"))
        else:
            df = df.withColumn("id", F.expr("uuid()"))

    # Make sure the partition key column exists & is string.
    pk = cfg["partition_key"]
    if pk not in df.columns:
        sys.exit(f"ERROR: partition key column '{pk}' not in source table.")
    df = df.withColumn(pk, F.col(pk).cast("string"))

    write_opts = {
        "spark.cosmos.accountEndpoint": cfg["endpoint"],
        "spark.cosmos.accountKey": cfg["key"],
        "spark.cosmos.database": cfg["database"],
        "spark.cosmos.container": cfg["container"],
        "spark.cosmos.write.strategy": "ItemOverwrite",
        "spark.cosmos.write.bulk.enabled": "true",
    }

    total = df.count()
    print(f"[info] writing {total} rows from {SOURCE_TABLE} (bizdate={BIZDATE}) "
          f"to {cfg['database']}.{cfg['container']} pk=/{pk}")

    (
        df.write.format("cosmos.oltp")
        .options(**write_opts)
        .mode("append")
        .save()
    )

    print("[done] upload complete.")


if __name__ == "__main__":
    main()
