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

# # NB_Transform_Bronze_Suppliers_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Builds one Silver row per supplier (code and country) from the supplier master.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/supplier_master (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.suppliers
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
# Imports and settings. Change paths and table names here only, not in the code below.
# - `BRONZE_TABLES` is the OneLake address of the Bronze tables (Bronze lives in a different workspace).

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, TARGET_TABLE, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql import Window

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "supplier_master"
TARGET_TABLE = "dbo.suppliers"

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
# Reads the supplier master from Bronze. Stops with an error if it cannot be read, is empty, or a column is missing.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the source is readable and has the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE
# Output  : src (Spark DataFrame), src_rows
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_COLUMNS = ['SupplierCode', 'Country']

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
# Builds the supplier table in three steps.
# 1. Trim the supplier code and rename columns to the planner's names. The supplier **name** is set equal to the code because the export has no separate supplier name.
# 2. Drop rows with no supplier code.
# 3. Keep one row per supplier code. If a code appears twice, one row is kept and the number removed is printed.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Build one clean row per supplier.
# Input   : src
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : supplier_name = supplier_code (no name in the export).
# =====================================================================
silver = src.select(
    F.trim(F.col("SupplierCode")).alias("supplier_code"),
    F.trim(F.col("SupplierCode")).alias("supplier_name"),
    F.col("Country").alias("country"),
)
rows_before = silver.count()

silver = silver.filter(F.col("supplier_code").isNotNull() & (F.col("supplier_code") != ""))
rows_no_code = silver.count()
print(f"2. missing supplier code: {rows_before - rows_no_code:,} removed, {rows_no_code:,} left")

silver = silver.dropDuplicates(["supplier_code"])
silver_rows = silver.count()
print(f"3. duplicate supplier codes: {rows_no_code - silver_rows:,} removed, {silver_rows:,} left")
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.suppliers`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the supplier table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.suppliers in lh_fortpro_silver
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
# Checks the finished table: how many suppliers, how many have a country, and how many are actually used in `dbo.purchases`. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output and fail if the supplier key is not clean.
# Input   : dbo.suppliers, dbo.purchases
# Output  : Printed summary (no raw records are displayed)
# Notes   : Most suppliers in the master were never used on a PO.
# =====================================================================
result = spark.table(TARGET_TABLE)

with_country = result.filter(F.col("country").isNotNull() & (F.col("country") != "")).count()
print(f"Suppliers with a country: {with_country:,} of {written_rows:,}")

used = spark.table("dbo.purchases").select("supplier_code").distinct()
used_n = used.count()
unknown = used.join(result.select("supplier_code"), "supplier_code", "left_anti").count()
print(f"Suppliers used on purchase orders: {used_n:,}; of those not in dbo.suppliers: {unknown:,}")

distinct_codes = result.select("supplier_code").distinct().count()
empty_codes = result.filter(F.col("supplier_code").isNull() | (F.col("supplier_code") == "")).count()
print(f"Distinct supplier codes: {distinct_codes:,}; empty codes: {empty_codes:,}")

if distinct_codes != written_rows or empty_codes:
    raise RuntimeError("FAILED: supplier_code is not unique or has empty values.")
print(f"SUCCESS: run {RUN_ID} finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, {written_rows:,} rows in {TARGET_TABLE}.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
