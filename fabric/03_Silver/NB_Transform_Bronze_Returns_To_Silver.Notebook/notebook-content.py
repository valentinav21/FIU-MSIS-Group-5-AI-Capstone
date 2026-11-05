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

# # NB_Transform_Bronze_Returns_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Cleans customer returns and credit memos into one Silver row per returned item line, so returned units can be netted against sales.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/returns_credit_memos (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.returns
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
# - `BASE_RETURN = 16` is the SAP code for a return document. A credit memo created *from* a return (BaseType 16) is the same event as the return, so it is dropped to avoid counting it twice.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, TARGET_TABLE, ACTIVE_WAREHOUSES,
#           DEFAULT_WAREHOUSE, BASE_RETURN, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "returns_credit_memos"
TARGET_TABLE = "dbo.returns"

ACTIVE_WAREHOUSES = ["A1", "H1", "A2"]
DEFAULT_WAREHOUSE = "A1"
BASE_RETURN = 16

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
# Reads the returns file from Bronze. It holds two document types: `ORDN` (goods returns) and `ORIN` (A/R credit memos). Stops with an error if it cannot be read, is empty, or a needed column is missing.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = [
    "DocType", "BaseType", "DocDate", "ItemCode", "Quantity", "CustomerCode",
    "WhsCode", "LineTotal", "CANCELED",
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
# Applies the cleaning rules in order and prints how many rows each removes.
# 1. Drop canceled documents (a missing flag means not canceled).
# 2. Drop credit memos created from a return (`ORIN` with BaseType 16): the return itself is already in the table.
# 3. Quantity as a positive number (SAP stores returns as negatives); drop zero quantities.
# 4. Warehouse: trim and upper-case, blank becomes the default; keep active warehouses only.
# 5. Rename to the planner's columns and label each row `return` or `credit memo`.
# 6. Drop rows with no item code or date.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Clean and conform returns; log rows removed per rule.
# Input   : src, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE, BASE_RETURN
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : quantity and line_total are stored as positive numbers.
# =====================================================================
def log_step(label, df, rows_before):
    """Print how many rows a cleaning rule removed. Returns the new row count."""
    rows_after = df.count()
    print(f"{label}: {rows_before - rows_after:,} removed, {rows_after:,} left")
    return rows_after

rows = src_rows

# Rule 1: canceled documents.
step = src.filter(F.coalesce(F.col("CANCELED"), F.lit("N")) != "Y")
rows = log_step("1. canceled documents", step, rows)

# Rule 2: credit memos based on a return (same event as the return).
step = step.filter(~((F.col("DocType") == "ORIN") & (F.col("BaseType").cast("long") == BASE_RETURN)))
rows = log_step("2. credit memos based on a return", step, rows)

# Rule 3: positive quantity.
step = step.withColumn("quantity", F.abs(F.col("Quantity").cast("double"))).filter(F.col("quantity") > 0)
rows = log_step("3. zero quantity", step, rows)

# Rule 4: warehouse.
whs = F.upper(F.trim(F.col("WhsCode")))
step = step.withColumn(
    "warehouse", F.when(whs.isNull() | (whs == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(whs)
).filter(F.col("warehouse").isin(ACTIVE_WAREHOUSES))
rows = log_step("4. non-active warehouses", step, rows)

# Rule 5: planner columns.
silver = step.select(
    F.to_date("DocDate").alias("doc_date"),
    F.trim(F.col("ItemCode")).alias("item_code"),
    "quantity",
    F.trim(F.col("CustomerCode")).alias("customer_code"),
    "warehouse",
    F.abs(F.col("LineTotal").cast("double")).alias("line_total"),
    F.when(F.col("DocType") == "ORDN", F.lit("return")).otherwise(F.lit("credit memo")).alias("reason"),
)

# Rule 6: missing item or date.
silver = silver.filter(F.col("item_code").isNotNull() & (F.col("item_code") != "") & F.col("doc_date").isNotNull())
silver_rows = log_step("6. missing item or date", silver, rows)
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.returns`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the returns table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.returns in lh_fortpro_silver
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
# Checks the finished table: rows and units by reason and warehouse, date range, and that every returned item exists in `dbo.items`. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if key columns have gaps.
# Input   : dbo.returns, dbo.items
# Output  : Printed summary (no raw records are displayed)
# Notes   : Returns are small next to sales; see units by reason.
# =====================================================================
result = spark.table(TARGET_TABLE)

result.groupBy("reason", "warehouse").agg(
    F.count("*").alias("rows"), F.sum("quantity").alias("units")
).orderBy("reason", "warehouse").show()

dates = result.agg(F.min("doc_date").alias("first"), F.max("doc_date").alias("last")).first()
print(f"Date range: {dates['first']} to {dates['last']}")

items_missing = result.join(spark.table("dbo.items").select("item_code"), "item_code", "left_anti").count()
print(f"Return rows whose item is not in dbo.items: {items_missing:,}")

KEY_COLUMNS = ["doc_date", "item_code", "quantity", "warehouse"]
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
