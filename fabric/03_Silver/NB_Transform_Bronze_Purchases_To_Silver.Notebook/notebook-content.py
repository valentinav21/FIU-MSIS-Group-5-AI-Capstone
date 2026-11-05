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

# # NB_Transform_Bronze_Purchases_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Cleans the raw purchase order lines into one Silver table of real, non-canceled PO lines with the column names the FORTPRO planner expects.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/purchase_orders (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.purchases
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
# - `BRONZE_TABLES` is the OneLake address of the Bronze tables, used because Bronze lives in a different workspace from Silver.
# - `ACTIVE_WAREHOUSES` and `DEFAULT_WAREHOUSE` are the same rules used in `dbo.sales`: A1 Master (Florida), H1 Houston, A2 Trading. A blank warehouse becomes A1.
# - `CURRENCY` is fixed to USD because 99.9% of PO lines are in dollars.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, TARGET_TABLE, ACTIVE_WAREHOUSES,
#           DEFAULT_WAREHOUSE, CURRENCY, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "purchase_orders"
TARGET_TABLE = "dbo.purchases"

ACTIVE_WAREHOUSES = ["A1", "H1", "A2"]
DEFAULT_WAREHOUSE = "A1"
CURRENCY = "USD"

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
# Reads the Bronze table and confirms it has the columns this notebook needs. Stops with an error if the table cannot be read, is empty, or a column is missing.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = [
    "DocEntry", "LineNum", "DocDate", "ItemCode", "SupplierCode", "Quantity",
    "OpenQty", "Price", "WhsCode", "CANCELED", "LineStatus",
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
print("Canceled flag values in the source:")
src.groupBy("CANCELED").count().show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Transform and Validate
# Applies the cleaning rules in order and prints how many rows each rule removes.
# 1. Drop canceled PO lines. About 31% of lines are canceled, and a canceled line is something we never actually bought.
# 2. Drop lines with quantity zero or less.
# 3. Warehouse: trim and upper-case. A blank warehouse becomes the default.
# 4. Rename columns to the planner's names (`po_entry`, `po_date`, `item_code`, `supplier_code`, ...).
# 5. Drop lines with no item, date or supplier.
# 
# `promised_date` is left empty on purpose. In 96% of lines the due date equals the PO date, so SAP has no real promise date. Observed lead times come from receipts instead (next notebooks).

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Clean and conform the PO lines; log rows removed per rule.
# Input   : src, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE, CURRENCY
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : Quantities and prices are cast to double before comparing.
# =====================================================================
def log_step(label, df, rows_before):
    """Print how many rows a cleaning rule removed. Returns the new row count."""
    rows_after = df.count()
    print(f"{label}: {rows_before - rows_after:,} removed, {rows_after:,} left")
    return rows_after

rows = src_rows

# Rule 1: canceled lines. A missing flag counts as not canceled.
step = src.filter(F.coalesce(F.col("CANCELED"), F.lit("N")) != "Y")
rows = log_step("1. canceled lines", step, rows)

# Rule 2: zero or negative quantities.
step = step.withColumn("quantity", F.col("Quantity").cast("double")).filter(F.col("quantity") > 0)
rows = log_step("2. quantity <= 0", step, rows)

# Rule 3: warehouse code (blank -> default). Unlike sales, no warehouse is dropped:
# an order into another warehouse is still a purchase, so we only report them.
whs = F.upper(F.trim(F.col("WhsCode")))
step = step.withColumn(
    "warehouse",
    F.when(whs.isNull() | (whs == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(whs),
)
other_whs = step.filter(~F.col("warehouse").isin(ACTIVE_WAREHOUSES)).count()
print(f"3. warehouse cleaned. Lines in non-active warehouses (kept): {other_whs:,}")

# Rule 4: planner column names.
silver = step.select(
    F.col("DocEntry").cast("string").alias("po_entry"),
    F.to_date("DocDate").alias("po_date"),
    F.trim(F.col("ItemCode")).alias("item_code"),
    F.trim(F.col("SupplierCode")).alias("supplier_code"),
    "quantity",
    F.col("LineNum").alias("line_num"),
    F.col("Price").cast("double").alias("unit_price"),
    F.lit(CURRENCY).alias("currency"),
    "warehouse",
    F.lit(None).cast("date").alias("promised_date"),
    F.col("OpenQty").cast("double").alias("open_qty"),
    F.col("LineStatus").alias("status"),
)

# Rule 5: lines we cannot place in time or tie to an item and supplier.
silver = silver.filter(
    F.col("item_code").isNotNull() & F.col("po_date").isNotNull() & F.col("supplier_code").isNotNull()
)
silver_rows = log_step("5. missing item, date or supplier", silver, rows)
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.purchases`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the cleaned table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.purchases in lh_fortpro_silver
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
# Checks the finished table: date range, distinct items and suppliers, warehouses, price coverage, and empty values in key columns. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if key columns have gaps.
# Input   : dbo.purchases
# Output  : Printed summary (no raw records are displayed)
# Notes   : unit_price > 0 coverage matters: it becomes the item cost later.
# =====================================================================
result = spark.table(TARGET_TABLE)

result.agg(
    F.min("po_date").alias("first_po"),
    F.max("po_date").alias("last_po"),
    F.countDistinct("item_code").alias("items"),
    F.countDistinct("supplier_code").alias("suppliers"),
    F.countDistinct("po_entry").alias("purchase_orders"),
).show()

print("Rows by warehouse:")
result.groupBy("warehouse").count().orderBy(F.desc("count")).show()

with_price = result.filter(F.col("unit_price") > 0).count()
print(f"Lines with a unit price > 0: {with_price:,} of {written_rows:,} ({with_price / written_rows:.1%})")

KEY_COLUMNS = ["po_entry", "po_date", "item_code", "supplier_code", "quantity"]
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
