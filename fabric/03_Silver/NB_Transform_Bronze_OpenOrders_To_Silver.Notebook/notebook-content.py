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

# # NB_Transform_Bronze_OpenOrders_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Builds the list of open customer sales-order lines (ordered but not yet delivered), so the planner can reserve stock for them.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/sales_lines (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.open_orders
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
# - `ACTIVE_WAREHOUSES` are A1 Master (Florida), H1 Houston, A2 Trading.
# - Open orders come from the same Bronze sales file as `dbo.sales`, but only sales orders (`ORDR`) that are still open.

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
SOURCE_TABLE = "sales_lines"
TARGET_TABLE = "dbo.open_orders"

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
# Reads the sales file from Bronze. Stops with an error if it cannot be read, is empty, or a needed column is missing.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = [
    "DocType", "ItemCode", "OpenQty", "WhsCode", "ShipDate", "CustomerCode",
    "CANCELED", "LineStatus",
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
print(f"SUCCESS: read {SOURCE_TABLE}, {src_rows:,} rows, all required columns present.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Transform and Validate
# Applies the rules in order and prints how many rows each removes.
# 1. Keep sales orders only (`ORDR`). Deliveries and invoices are not open demand.
# 2. Drop canceled documents (a missing flag means not canceled).
# 3. Keep lines still open (`LineStatus` = `O`).
# 4. Keep lines with open quantity above zero.
# 5. Warehouse: trim and upper-case, blank becomes the default; keep active warehouses only.
# 6. Rename to the planner's columns; `due_date` is the line's ship date.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Build the open sales-order lines; log rows removed per rule.
# Input   : src, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : open_qty is what is still to be delivered, not the ordered qty.
# =====================================================================
def log_step(label, df, rows_before):
    """Print how many rows a cleaning rule removed. Returns the new row count."""
    rows_after = df.count()
    print(f"{label}: {rows_before - rows_after:,} removed, {rows_after:,} left")
    return rows_after

rows = src_rows

step = src.filter(F.col("DocType") == "ORDR")
rows = log_step("1. not a sales order", step, rows)

step = step.filter(F.coalesce(F.col("CANCELED"), F.lit("N")) != "Y")
rows = log_step("2. canceled documents", step, rows)

step = step.filter(F.col("LineStatus") == "O")
rows = log_step("3. line not open", step, rows)

step = step.withColumn("open_qty", F.col("OpenQty").cast("double")).filter(F.col("open_qty") > 0)
rows = log_step("4. no open quantity", step, rows)

whs = F.upper(F.trim(F.col("WhsCode")))
step = step.withColumn(
    "warehouse", F.when(whs.isNull() | (whs == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(whs)
).filter(F.col("warehouse").isin(ACTIVE_WAREHOUSES))
rows = log_step("5. non-active warehouses", step, rows)

silver = step.select(
    F.trim(F.col("ItemCode")).alias("item_code"),
    "open_qty",
    "warehouse",
    F.to_date("ShipDate").alias("due_date"),
    F.trim(F.col("CustomerCode")).alias("customer_code"),
).filter(F.col("item_code").isNotNull() & (F.col("item_code") != ""))
silver_rows = log_step("6. missing item code", silver, rows)
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.open_orders`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the open orders table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.open_orders in lh_fortpro_silver
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
# Checks the finished table: open lines and units by warehouse, due-date range, how many are already overdue, and that every item exists in `dbo.items`. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if key columns have gaps.
# Input   : dbo.open_orders, dbo.items
# Output  : Printed summary (no raw records are displayed)
# Notes   : Old open lines are often orders nobody closed in SAP.
# =====================================================================
result = spark.table(TARGET_TABLE)

result.groupBy("warehouse").agg(
    F.count("*").alias("lines"), F.countDistinct("item_code").alias("items"), F.sum("open_qty").alias("open_units")
).orderBy("warehouse").show()

dates = result.agg(F.min("due_date").alias("first"), F.max("due_date").alias("last")).first()
print(f"Due-date range: {dates['first']} to {dates['last']}")

overdue = result.filter(F.col("due_date") < F.current_date()).count()
print(f"Lines already past due date: {overdue:,} of {written_rows:,}")
older = result.filter(F.col("due_date") < F.date_sub(F.current_date(), 180)).count()
print(f"Lines more than 180 days past due (possibly never closed): {older:,}")

items_missing = result.join(spark.table("dbo.items").select("item_code"), "item_code", "left_anti").count()
print(f"Open lines whose item is not in dbo.items: {items_missing:,}")

KEY_COLUMNS = ["item_code", "open_qty", "warehouse"]
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
