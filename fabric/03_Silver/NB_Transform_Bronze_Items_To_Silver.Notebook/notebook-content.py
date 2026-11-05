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

# # NB_Transform_Bronze_Items_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Builds one Silver row per item from the item master, adding a unit cost (latest purchase price) and a list price (median selling price, last 12 months), because the item master has neither.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/item_master (read by OneLake path); Silver dbo.purchases and dbo.sales
# 
# **Destination:** lh_fortpro_silver / dbo.items
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver. Read access to the Landing workspace. Upstream: NB_Bronze_Load_Raw_To_Tables, NB_Transform_Bronze_Sales_To_Silver, NB_Transform_Bronze_Purchases_To_Silver (run those first).
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
# - `PURCHASES_TABLE` and `SALES_TABLE` are Silver tables built earlier. This notebook needs them for cost and price.
# - `PRICE_WINDOW_MONTHS` is how far back we look for the selling price (12 months, counted back from the last order date, not from today).

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, PURCHASES_TABLE, SALES_TABLE,
#           TARGET_TABLE, PRICE_WINDOW_MONTHS, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql import Window

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "item_master"
PURCHASES_TABLE = "dbo.purchases"
SALES_TABLE = "dbo.sales"
TARGET_TABLE = "dbo.items"

PRICE_WINDOW_MONTHS = 12

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
# Reads the item master from Bronze and the two Silver tables built earlier. Stops with an error if any is unreadable, empty, or missing a column this notebook needs.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm all three inputs are readable and have the columns we use.
# Input   : BRONZE_TABLES, SOURCE_TABLE, PURCHASES_TABLE, SALES_TABLE
# Output  : src, src_rows, purchases, sales
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = [
    "ItemCode", "ItemName", "ItmsGrpNam", "FirmName", "validFor", "frozenFor",
    "MinOrdrQty", "OrdrMulti", "LeadTime", "PrefVendorCode",
]

try:
    src = spark.read.format("delta").load(f"{BRONZE_TABLES}/{SOURCE_TABLE}")
    src_rows = src.count()
    purchases = spark.table(PURCHASES_TABLE)
    sales = spark.table(SALES_TABLE)
except Exception as e:
    raise RuntimeError(
        f"FAILED: cannot read an input. Check the path in 01, read access to Landing, "
        f"and that the sales and purchases notebooks ran first. Details: {e}"
    )

missing_columns = [c for c in REQUIRED_COLUMNS if c not in src.columns]
if src_rows == 0 or missing_columns:
    raise RuntimeError(f"FAILED: source has {src_rows} rows, missing columns: {missing_columns}")

print(f"SUCCESS: read {SOURCE_TABLE}, {src_rows:,} rows, all required columns present.")
print(f"SUCCESS: {PURCHASES_TABLE} has {purchases.count():,} rows, {SALES_TABLE} has {sales.count():,} rows.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Transform and Validate
# Builds the item table in four steps.
# 1. **Unit cost:** the item master has no cost, so use the price on the item's most recent purchase order line (latest PO date, ties broken by PO number and line).
# 2. **List price:** the median net selling price of the item's orders (`doc_type = ORDR`) in the last 12 months. Orders only, so a sale is not counted three times.
# 3. **Active flag:** an item is active when it is marked valid and not frozen.
# 4. **Planner columns:** names, minimum order quantity, order multiple, lead time and preferred supplier. A minimum order quantity or multiple of 0 means "not set", so it becomes empty.
# 
# Items with no code are dropped and duplicate codes are collapsed to one row. Coverage of cost and price is printed because many items were never bought or sold recently.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Build one row per item with cost, price and planning fields.
# Input   : src, purchases, sales, PRICE_WINDOW_MONTHS
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : unit_cost = latest PO price; list_price = median ORDR price.
# =====================================================================
# Step 1: latest PO price per item.
latest_po = (
    purchases.filter(F.col("unit_price") > 0)
    .withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("item_code").orderBy(
                F.col("po_date").desc(), F.col("po_entry").desc(), F.col("line_num").desc()
            )
        ),
    )
    .filter(F.col("rn") == 1)
    .select("item_code", F.col("unit_price").alias("unit_cost"))
)

# Step 2: median selling price of orders in the last N months.
last_order_date = sales.filter(F.col("doc_type") == "ORDR").agg(F.max("doc_date")).first()[0]
window_start = F.add_months(F.lit(last_order_date), -PRICE_WINDOW_MONTHS)
list_price = (
    sales.filter((F.col("doc_type") == "ORDR") & (F.col("doc_date") >= window_start) & (F.col("unit_price") > 0))
    .groupBy("item_code")
    .agg(F.expr("percentile(unit_price, 0.5)").alias("list_price"))
)
print(f"Price window: {PRICE_WINDOW_MONTHS} months back from the last order date {last_order_date}")

# Steps 3 and 4: active flag and planner columns.
active = (F.upper(F.trim(F.col("validFor"))) == "Y") & (F.coalesce(F.upper(F.trim(F.col("frozenFor"))), F.lit("N")) != "Y")

def zero_to_null(col):
    """A value of 0 means 'not set' in the item master, so make it empty."""
    value = F.col(col).cast("double")
    return F.when(value == 0, F.lit(None)).otherwise(value)

items = src.select(
    F.trim(F.col("ItemCode")).alias("item_code"),
    F.col("ItemName").alias("item_name"),
    F.col("ItmsGrpNam").alias("category"),
    F.col("FirmName").alias("brand"),
    F.when(active, F.lit("Y")).otherwise(F.lit("N")).alias("active"),
    zero_to_null("MinOrdrQty").alias("moq"),
    zero_to_null("OrdrMulti").alias("order_multiple"),
    F.col("LeadTime").cast("double").alias("lead_time_days"),
    F.trim(F.col("PrefVendorCode")).alias("preferred_supplier"),
)

silver = (
    items.join(latest_po, "item_code", "left")
    .join(list_price, "item_code", "left")
    .select(
        "item_code", "item_name", "category", "brand", "unit_cost", "list_price",
        "active", "moq", "order_multiple", "lead_time_days", "preferred_supplier",
    )
)

# Items with no code cannot be used; keep one row per code.
rows_before = silver.count()
silver = silver.filter(F.col("item_code").isNotNull() & (F.col("item_code") != "")).dropDuplicates(["item_code"])
silver_rows = silver.count()
print(f"Items with no code or duplicate code: {rows_before - silver_rows:,} removed, {silver_rows:,} left")

cost_n = silver.filter(F.col("unit_cost").isNotNull()).count()
price_n = silver.filter(F.col("list_price").isNotNull()).count()
print(f"unit_cost filled for {cost_n:,} items ({cost_n / silver_rows:.0%}); list_price for {price_n:,} ({price_n / silver_rows:.0%})")
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.items`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the item table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.items in lh_fortpro_silver
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
# Checks the finished table: active vs inactive items, cost and price coverage, and that every item code is unique and not empty. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if the item key is not clean.
# Input   : dbo.items
# Output  : Printed summary (no raw records are displayed)
# Notes   : Missing cost is expected for items never bought; the planner
#           falls back to a share of the selling price for those.
# =====================================================================
result = spark.table(TARGET_TABLE)

print("Items by active flag:")
result.groupBy("active").count().orderBy("active").show()

missing_lt = result.filter(F.col("lead_time_days").isNull() | (F.col("lead_time_days") == 0)).count()
print(f"Items with no lead time set in SAP: {missing_lt:,} of {written_rows:,}")

distinct_codes = result.select("item_code").distinct().count()
empty_codes = result.filter(F.col("item_code").isNull() | (F.col("item_code") == "")).count()
print(f"Distinct item codes: {distinct_codes:,}; empty codes: {empty_codes:,}")

if distinct_codes != written_rows or empty_codes:
    raise RuntimeError("FAILED: item_code is not unique or has empty values.")
print(f"SUCCESS: run {RUN_ID} finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, {written_rows:,} rows in {TARGET_TABLE}.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
