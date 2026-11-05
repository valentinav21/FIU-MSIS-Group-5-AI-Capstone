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

# # NB_Transform_Bronze_Sales_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Cleans the raw sales document lines (orders, deliveries and A/R invoices) and conforms them into one Silver table with the column names the FORTPRO planner expects.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/sales_lines (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.sales
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
# - `BRONZE_TABLES` is the OneLake address of the Bronze tables (workspace ID, lakehouse ID, `Tables/dbo`). It is used because Bronze lives in a different workspace from Silver.
# - `ACTIVE_WAREHOUSES` are the warehouses with real activity: A1 Master (Florida), H1 Houston, A2 Trading. Others are test, transit or showroom.
# - `DEFAULT_WAREHOUSE` is used when a line has no warehouse.

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
TARGET_TABLE = "dbo.sales"

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
# Reads the Bronze table and confirms it has the columns this notebook needs. Stops with an error if the table cannot be read, is empty, or a column is missing, so a bad source never reaches Silver. Expected: three document types (ORDR orders, ODLN deliveries, OINV A/R invoices).

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = [
    "DocType", "DocDate", "DocEntry", "LineNum", "ItemCode", "Quantity", "OpenQty",
    "LineTotal", "DiscPrcnt", "WhsCode", "CustomerCode", "Channel", "StateS",
    "CountryS", "SlpCode", "CANCELED", "LineStatus", "ShipDate",
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
src.groupBy("DocType").count().orderBy("DocType").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Transform and Validate
# Applies the cleaning rules in order and prints how many rows each rule removes, so every drop can be explained.
# 1. Drop canceled documents.
# 2. Drop lines with quantity zero or less. They are not demand.
# 3. Warehouse: trim and upper-case. A blank warehouse becomes the default. Keep only the active warehouses.
# 4. Rename columns to the planner's names. Unit price = line total / quantity (net of discount).
# 5. Drop lines with no item code or no date (cannot be placed in time).
# 
# All three document types stay in the table (`doc_type`). Never add them together: one sale appears as an order, a delivery and an invoice. The planner extract uses orders only (our demand decision); deliveries are kept to score SAP's forecast.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Clean and conform the sales lines; log rows removed per rule.
# Input   : src, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : Quantity and amounts arrive as decimals/strings in Bronze, so
#           they are cast to double before any comparison.
# =====================================================================
def log_step(label, df, rows_before):
    """Print how many rows a cleaning rule removed. Returns the new row count."""
    rows_after = df.count()
    print(f"{label}: {rows_before - rows_after:,} removed, {rows_after:,} left")
    return rows_after

rows = src_rows

# Rule 1: canceled documents are not real sales. A missing flag counts as not canceled.
step = src.filter(F.coalesce(F.col("CANCELED"), F.lit("N")) != "Y")
rows = log_step("1. canceled documents", step, rows)

# Rule 2: zero or negative quantities (returns, adjustments) are not demand.
step = step.withColumn("quantity", F.col("Quantity").cast("double")).filter(F.col("quantity") > 0)
rows = log_step("2. quantity <= 0", step, rows)

# Rule 3: normalize the warehouse code, default blanks to Master, keep active warehouses only.
warehouse_clean = F.upper(F.trim(F.col("WhsCode")))
step = step.withColumn(
    "warehouse",
    F.when(warehouse_clean.isNull() | (warehouse_clean == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(warehouse_clean),
)
step = step.filter(F.col("warehouse").isin(ACTIVE_WAREHOUSES))
rows = log_step("3. non-active warehouses", step, rows)

# Rule 4: planner column names. Net unit price is derived from the line total.
line_total = F.col("LineTotal").cast("double")
silver = step.select(
    F.col("DocType").alias("doc_type"),
    F.to_date("DocDate").alias("doc_date"),
    F.trim(F.col("ItemCode")).alias("item_code"),
    "quantity",
    F.col("DocEntry").cast("string").alias("doc_entry"),
    F.col("LineNum").alias("line_num"),
    F.trim(F.col("CustomerCode")).alias("customer_code"),
    "warehouse",
    F.round(line_total / F.col("quantity"), 4).alias("unit_price"),
    F.col("DiscPrcnt").cast("double").alias("discount_pct"),
    line_total.alias("line_total"),
    F.col("Channel").alias("channel"),
    F.col("StateS").alias("ship_state"),
    F.col("CountryS").alias("ship_country"),
    F.col("SlpCode").cast("string").alias("sales_employee"),
    F.col("LineStatus").alias("line_status"),
    F.col("OpenQty").cast("double").alias("open_qty"),
    F.to_date("ShipDate").alias("ship_date"),
)

# Rule 5: a line without an item or a date cannot be used for forecasting.
silver = silver.filter(F.col("item_code").isNotNull() & F.col("doc_date").isNotNull())
silver_rows = log_step("5. missing item or date", silver, rows)
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.sales`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the cleaned table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.sales in lh_fortpro_silver
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
# Checks the finished table: rows by document type, warehouses, date range, distinct items and customers, and that the key columns have no empty values. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if key columns have gaps.
# Input   : dbo.sales
# Output  : Printed summary (no raw records are displayed)
# Notes   : ORDR rows are the demand used by the planner extract.
# =====================================================================
result = spark.table(TARGET_TABLE)

print("Rows by document type:")
result.groupBy("doc_type").count().orderBy("doc_type").show()

warehouses = sorted(r[0] for r in result.select("warehouse").distinct().collect())
print("Warehouses:", warehouses)
result.agg(
    F.min("doc_date").alias("first_date"),
    F.max("doc_date").alias("last_date"),
    F.countDistinct("item_code").alias("items"),
    F.countDistinct("customer_code").alias("customers"),
).show()

KEY_COLUMNS = ["doc_date", "item_code", "quantity", "warehouse"]
null_counts = result.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in KEY_COLUMNS]).first().asDict()
print("Empty values in key columns:", null_counts)

if any(v for v in null_counts.values()) or not set(warehouses) <= set(ACTIVE_WAREHOUSES):
    raise RuntimeError("FAILED: key columns have empty values or an unexpected warehouse.")
print(f"SUCCESS: run {RUN_ID} finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, {written_rows:,} rows in {TARGET_TABLE}.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
