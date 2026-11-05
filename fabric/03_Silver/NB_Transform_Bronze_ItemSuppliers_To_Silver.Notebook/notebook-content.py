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

# # NB_Transform_Bronze_ItemSuppliers_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Builds one Silver row per item and supplier pair (who can supply which item) with the price from the last purchase order for that pair.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/item_supplier (read by OneLake path); Silver dbo.purchases
# 
# **Destination:** lh_fortpro_silver / dbo.item_suppliers
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver. Read access to the Landing workspace. Upstream: NB_Bronze_Load_Raw_To_Tables. Run NB_Transform_Bronze_Purchases_To_Silver first (the price comes from dbo.purchases).
# 
# **Change History:**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-11-04 | Dagiana Toussaint | Initial notebook creation |

# MARKDOWN ********************

# ## 01. Configuration
# Imports and settings. Change paths and table names here only, not in the code below.
# - `BRONZE_TABLES` is the OneLake address of the Bronze tables (Bronze lives in a different workspace).
# - `PURCHASES_TABLE` is the Silver purchase order table, used for the last price paid.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, PURCHASES_TABLE, TARGET_TABLE, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql import Window

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "item_supplier"
PURCHASES_TABLE = "dbo.purchases"
TARGET_TABLE = "dbo.item_suppliers"

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
# Reads the item-supplier table from Bronze. Stops with an error if it cannot be read, is empty, or a column is missing.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = ['ItemCode', 'SupplierCode']

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
# Builds the item-supplier table in four steps.
# 1. Take each distinct item and supplier pair from SAP's item-supplier list. Codes are trimmed and renamed.
# 2. Drop pairs with no item or no supplier.
# 3. **Price:** for each pair, take the price on the most recent purchase order line for that item and supplier (latest PO date, ties broken by PO number and line). Pairs that were never ordered have no price, which is expected.
# 4. Print how many pairs got a price.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Build item-supplier pairs with the last price paid.
# Input   : src, PURCHASES_TABLE
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : price = latest PO unit_price > 0 for the same item and supplier.
# =====================================================================
pairs = src.select(
    F.trim(F.col("ItemCode")).alias("item_code"),
    F.trim(F.col("SupplierCode")).alias("supplier_code"),
).distinct()
pair_rows = pairs.count()
print(f"1. distinct item-supplier pairs: {pair_rows:,} (from {src_rows:,} source rows)")

pairs = pairs.filter(
    F.col("item_code").isNotNull() & (F.col("item_code") != "")
    & F.col("supplier_code").isNotNull() & (F.col("supplier_code") != "")
)
rows = pairs.count()
print(f"2. missing item or supplier: {pair_rows - rows:,} removed, {rows:,} left")

last_price = (
    spark.table(PURCHASES_TABLE).filter(F.col("unit_price") > 0)
    .withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("item_code", "supplier_code").orderBy(
                F.col("po_date").desc(), F.col("po_entry").desc(), F.col("line_num").desc()
            )
        ),
    )
    .filter(F.col("rn") == 1)
    .select("item_code", "supplier_code", "unit_price")
)
silver = pairs.join(last_price, ["item_code", "supplier_code"], "left")
silver_rows = silver.count()
if silver_rows != rows:
    raise RuntimeError("FAILED: the price join changed the row count.")

priced = silver.filter(F.col("unit_price").isNotNull()).count()
print(f"4. pairs with a price from a purchase order: {priced:,} of {silver_rows:,} ({priced / silver_rows:.0%})")
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.item_suppliers`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the item-supplier table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.item_suppliers in lh_fortpro_silver
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
# Checks the finished table: pair count, items and suppliers covered, how many items have more than one supplier, and whether the codes exist in `dbo.items` and `dbo.suppliers`. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail on empty or duplicate pairs.
# Input   : dbo.item_suppliers, dbo.items, dbo.suppliers
# Output  : Printed summary (no raw records are displayed)
# Notes   : Codes missing from items or suppliers are reported, not fatal.
# =====================================================================
result = spark.table(TARGET_TABLE)

result.agg(
    F.countDistinct("item_code").alias("items"),
    F.countDistinct("supplier_code").alias("suppliers"),
).show()

multi = result.groupBy("item_code").count().filter(F.col("count") > 1).count()
print(f"Items with more than one supplier: {multi:,}")

no_item = result.join(spark.table("dbo.items").select("item_code"), "item_code", "left_anti").count()
no_sup = result.join(spark.table("dbo.suppliers").select("supplier_code"), "supplier_code", "left_anti").count()
print(f"Pairs whose item is not in dbo.items: {no_item:,}; whose supplier is not in dbo.suppliers: {no_sup:,}")

distinct_pairs = result.select("item_code", "supplier_code").distinct().count()
print(f"Distinct pairs: {distinct_pairs:,} of {written_rows:,} rows")

if distinct_pairs != written_rows:
    raise RuntimeError("FAILED: duplicate item-supplier pairs.")
print(f"SUCCESS: run {RUN_ID} finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, {written_rows:,} rows in {TARGET_TABLE}.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
