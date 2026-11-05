# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "73166bec-8d01-49be-b864-6c93bbc66b49",
# META       "default_lakehouse_name": "lh_fortpro_silver",
# META       "default_lakehouse_workspace_id": "e2b0b198-404d-4a39-887c-6eea3fa6ae4d"
# META     }
# META   }
# META }

# MARKDOWN ********************

# # NB_Transform_Bronze_Stock_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Cleans the stock-by-warehouse snapshot into one Silver row per item and active warehouse, with the column names the FORTPRO planner expects.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/stock_by_warehouse (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.stock
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver. Read access to the Landing workspace. Upstream: NB_Bronze_Load_Raw_To_Tables.
# 
# **Change History:**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-11-04 | Dagiana Toussaint | Initial notebook creation |

# MARKDOWN ********************

# ## 01. Configuration
# Imports and settings. Change paths and business rules here only, not in the code below.
# - `BRONZE_TABLES` is the OneLake address of the Bronze tables (Bronze lives in a different workspace).
# - `ACTIVE_WAREHOUSES` are the warehouses with real activity: A1 Master (Florida), H1 Houston, A2 Trading. Stock rows in other warehouses (test, transit, showroom) are dropped.
# - `DEFAULT_WAREHOUSE` is used when a row has no warehouse.
# 
# This table is a **snapshot** of one day, not a history. The snapshot date is kept in a column so it is clear how old the numbers are.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, TARGET_TABLE, ACTIVE_WAREHOUSES,
#           DEFAULT_WAREHOUSE, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "stock_by_warehouse"
TARGET_TABLE = "dbo.stock"

ACTIVE_WAREHOUSES = ["A1", "H1", "A2"]
DEFAULT_WAREHOUSE = "A1"

RUN_ID = str(uuid.uuid4())[:8]
RUN_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
print(f"Run {RUN_ID} started {RUN_UTC}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 02. Source Validation
# Reads the stock table from Bronze. Stops with an error if it cannot be read, is empty, or a column this notebook needs is missing. `SnapshotDate` is optional: if the column is missing, the snapshot date is left empty.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows, has_snapshot
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = [
    "ItemCode", "WhsCode", "OnHand", "IsCommited", "OnOrder", "MinStock", "MaxStock",
]

try:
    src = spark.read.format("delta").load(f"{BRONZE_TABLES}/{SOURCE_TABLE}")
    src_rows = src.count()
except Exception as e:
    raise RuntimeError(
        f"FAILED: cannot read {SOURCE_TABLE} from Bronze. Check the path in 01 and "
        f"that you have read access to the Landing workspace. Details: {e}"
    )

missing_columns = [c for c in REQUIRED_COLUMNS if c not in src.columns]
if src_rows == 0 or missing_columns:
    raise RuntimeError(f"FAILED: source has {src_rows} rows, missing columns: {missing_columns}")

has_snapshot = "SnapshotDate" in src.columns
print(f"SUCCESS: read {SOURCE_TABLE}, {src_rows:,} rows, all required columns present. SnapshotDate column: {has_snapshot}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Transform and Validate
# Applies the cleaning rules in order and prints how many rows each rule removes.
# 1. Warehouse: trim and upper-case. A blank warehouse becomes the default.
# 2. Keep only the active warehouses. This drops transit, showroom and test warehouses.
# 3. Rename columns to the planner's names (`on_hand`, `committed`, `on_order`, `min_stock`, `max_stock`). Quantities are cast to numbers.
# 4. Drop rows with no item code.
# 5. Check that there is only one row per item and warehouse. If not, the notebook stops, because duplicate rows would double the stock.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Clean and conform the stock snapshot; log rows removed per rule.
# Input   : src, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE, has_snapshot
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : committed = stock promised to open sales orders; on_order =
#           stock on open purchase orders (both as reported by SAP).
# =====================================================================
def log_step(label, df, rows_before):
    """Print how many rows a cleaning rule removed. Returns the new row count."""
    rows_after = df.count()
    print(f"{label}: {rows_before - rows_after:,} removed, {rows_after:,} left")
    return rows_after

rows = src_rows

# Rule 1: warehouse code (blank -> default).
whs = F.upper(F.trim(F.col("WhsCode")))
step = src.withColumn(
    "warehouse", F.when(whs.isNull() | (whs == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(whs)
)

# Rule 2: active warehouses only.
step = step.filter(F.col("warehouse").isin(ACTIVE_WAREHOUSES))
rows = log_step("2. non-active warehouses", step, rows)

# Rule 3: planner column names; quantities as numbers.
snapshot = F.to_date("SnapshotDate") if has_snapshot else F.lit(None).cast("date")
silver = step.select(
    F.trim(F.col("ItemCode")).alias("item_code"),
    "warehouse",
    F.col("OnHand").cast("double").alias("on_hand"),
    F.col("IsCommited").cast("double").alias("committed"),
    F.col("OnOrder").cast("double").alias("on_order"),
    F.col("MinStock").cast("double").alias("min_stock"),
    F.col("MaxStock").cast("double").alias("max_stock"),
    snapshot.alias("snapshot_date"),
)

# Rule 4: rows with no item code.
silver = silver.filter(F.col("item_code").isNotNull() & (F.col("item_code") != ""))
silver_rows = log_step("4. missing item code", silver, rows)

# Rule 5: one row per item and warehouse.
distinct_keys = silver.select("item_code", "warehouse").distinct().count()
if distinct_keys != silver_rows:
    raise RuntimeError(f"FAILED: {silver_rows - distinct_keys:,} duplicate item-warehouse rows (stock would double).")
print(f"5. one row per item and warehouse: confirmed ({distinct_keys:,} keys)")
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.stock`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the stock table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.stock in lh_fortpro_silver
# Notes   : overwrite = repeatable; a row-count mismatch stops the run.
# =====================================================================
silver.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(TARGET_TABLE)

written_rows = spark.table(TARGET_TABLE).count()
if written_rows != silver_rows:
    raise RuntimeError(f"FAILED: wrote {silver_rows:,} rows but {TARGET_TABLE} has {written_rows:,}.")
print(f"SUCCESS: {TARGET_TABLE} written and verified, {written_rows:,} rows.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 05. Execution Summary and Logging
# Checks the finished table: snapshot date, rows and items per warehouse, how many items actually hold stock, negative quantities, and that every stocked item exists in `dbo.items`. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if key columns have gaps.
# Input   : dbo.stock, dbo.items
# Output  : Printed summary (no raw records are displayed)
# Notes   : Only a share of the item master holds stock; see the counts.
# =====================================================================
result = spark.table(TARGET_TABLE)

print("Snapshot date(s):", [str(r[0]) for r in result.select("snapshot_date").distinct().collect()])

print("Rows and items holding stock, by warehouse:")
result.groupBy("warehouse").agg(
    F.count("*").alias("rows"),
    F.sum((F.col("on_hand") > 0).cast("int")).alias("items_with_stock"),
    F.sum("on_hand").alias("units_on_hand"),
).orderBy("warehouse").show()

stocked_items = result.filter(F.col("on_hand") > 0).select("item_code").distinct().count()
total_items = spark.table("dbo.items").count()
print(f"Items with stock on hand in any warehouse: {stocked_items:,} of {total_items:,} in the item master")

negative = result.filter(F.col("on_hand") < 0).count()
print(f"Rows with negative on_hand: {negative:,}")

items_missing = result.join(spark.table("dbo.items").select("item_code"), "item_code", "left_anti").count()
print(f"Stock rows whose item is not in dbo.items: {items_missing:,}")

KEY_COLUMNS = ["item_code", "warehouse", "on_hand"]
null_counts = result.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in KEY_COLUMNS]).first().asDict()
print("Empty values in key columns:", null_counts)

if any(v for v in null_counts.values()):
    raise RuntimeError("FAILED: key columns have empty values.")
print(f"SUCCESS: run {RUN_ID} finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, {written_rows:,} rows in {TARGET_TABLE}.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
