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

# # NB_Transform_Bronze_SapForecasts_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Cleans the forecasts SAP saved (one row per forecast version, item, warehouse and month), so SAP's own forecast can be scored against actual sales.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/sap_forecasts (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.sap_forecasts
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
# 
# Each SAP forecast version (for example "Marzo 2026-v1") lists many items and months. Keeping every version lets us score what SAP predicted at each point in time.

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
SOURCE_TABLE = "sap_forecasts"
TARGET_TABLE = "dbo.sap_forecasts"

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
# Reads the SAP forecast table from Bronze. Stops with an error if it cannot be read, is empty, or a needed column is missing.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = ["FcstName", "AbsID", "ItemCode", "Date", "Quantity", "WhsCode"]

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
# 1. Warehouse: trim and upper-case (fixes `h1` vs `H1`), blank becomes the default. Keep active warehouses only: forecasts for test or transit warehouses cannot be compared with sales.
# 2. Quantity as a number; drop zero or negative quantities (no forecast).
# 3. Drop rows with no item code or date.
# 4. Rename to clean names. `forecast_month` is the month the forecast is *for*.
# 5. Check there is one row per forecast version, item, warehouse and month. If not, the notebook stops.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Clean SAP's saved forecasts; log rows removed per rule.
# Input   : src, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : SAP forecast for month M = deliveries of month M-9 (see analysis).
# =====================================================================
def log_step(label, df, rows_before):
    """Print how many rows a cleaning rule removed. Returns the new row count."""
    rows_after = df.count()
    print(f"{label}: {rows_before - rows_after:,} removed, {rows_after:,} left")
    return rows_after

rows = src_rows

whs = F.upper(F.trim(F.col("WhsCode")))
step = src.withColumn(
    "warehouse", F.when(whs.isNull() | (whs == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(whs)
).filter(F.col("warehouse").isin(ACTIVE_WAREHOUSES))
rows = log_step("1. non-active warehouses", step, rows)

step = step.withColumn("forecast_qty", F.col("Quantity").cast("double")).filter(F.col("forecast_qty") > 0)
rows = log_step("2. zero or negative quantity", step, rows)

step = step.filter(F.col("ItemCode").isNotNull() & (F.trim(F.col("ItemCode")) != "") & F.col("Date").isNotNull())
rows = log_step("3. missing item or date", step, rows)

silver = step.select(
    F.col("AbsID").cast("long").alias("forecast_id"),
    F.trim(F.col("FcstName")).alias("forecast_name"),
    F.trim(F.col("ItemCode")).alias("item_code"),
    "warehouse",
    F.trunc(F.to_date("Date"), "month").alias("forecast_month"),
    "forecast_qty",
)
silver_rows = silver.count()

distinct_keys = silver.select("forecast_id", "item_code", "warehouse", "forecast_month").distinct().count()
if distinct_keys != silver_rows:
    raise RuntimeError(f"FAILED: {silver_rows - distinct_keys:,} duplicate forecast rows.")
print("5. one row per forecast, item, warehouse and month: confirmed")
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.sap_forecasts`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the SAP forecast table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.sap_forecasts in lh_fortpro_silver
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
# Checks the finished table: number of forecast versions, month range covered, rows per warehouse, and items missing from `dbo.items`. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if key columns have gaps.
# Input   : dbo.sap_forecasts, dbo.items
# Output  : Printed summary (no raw records are displayed)
# Notes   : Items missing from dbo.items are items SAP forecast but that
#           have no master record (informational).
# =====================================================================
result = spark.table(TARGET_TABLE)

print("Forecast versions:", result.select("forecast_id").distinct().count())
rng = result.agg(F.min("forecast_month").alias("first"), F.max("forecast_month").alias("last")).first()
print(f"Forecast months covered: {rng['first']} to {rng['last']}")

result.groupBy("warehouse").agg(
    F.count("*").alias("rows"), F.countDistinct("item_code").alias("items")
).orderBy("warehouse").show()

items_missing = result.join(spark.table("dbo.items").select("item_code"), "item_code", "left_anti").select("item_code").distinct().count()
print(f"Forecast items not in dbo.items: {items_missing:,}")

KEY_COLUMNS = ["forecast_id", "item_code", "warehouse", "forecast_month", "forecast_qty"]
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
