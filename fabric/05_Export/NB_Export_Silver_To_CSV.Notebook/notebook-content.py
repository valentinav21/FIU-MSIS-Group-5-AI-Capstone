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

# # NB_Export_Silver_To_CSV
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Exports the Silver tables as plain CSV files (one per table) so the data can be shared with the team building the AI Planner. Customer and supplier codes were already replaced with consistent IDs by FORTPRO's IT before delivery, so no further masking is applied.
# 
# **Source:** lh_fortpro_silver / dbo tables: sales, open_orders, returns, purchases, receipts, stock, items, suppliers, item_suppliers
# 
# **Destination:** lh_fortpro_silver / Files/csv_export (one CSV per table)
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver. Upstream: all NB_Transform_Bronze_*_To_Silver notebooks must have run.
# 
# **Change History:**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-11-04 | Dagiana Toussaint | Initial notebook creation |

# MARKDOWN ********************

# ## 01. Configuration
# Imports and settings. Change tables, columns and folders here only, not in the code below.
# - `EXPORT_COLUMNS` lists, per table, exactly which columns go into the CSV and in what order. Silver has a few extra columns the AI Planner does not use.
# - Sales is exported for **sales orders only** (`doc_type = ORDR`). Deliveries and invoices describe the same sales again and would triple the demand.
# - No masking is applied. FORTPRO's IT already replaced customer and supplier names with consistent ID codes, and the export carries no names, contacts or tax IDs.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep tables, columns and folders in one place.
# Input   : None
# Output  : EXPORT_COLUMNS, SALES_DOC_TYPE, OUT_DIR, RUN_ID, RUN_UTC
# Notes   : /lakehouse/default is the attached lakehouse lh_fortpro_silver.
# =====================================================================
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from pyspark.sql import functions as F

OUT_DIR = "/lakehouse/default/Files/csv_export"

SALES_DOC_TYPE = "ORDR"

EXPORT_COLUMNS = {
    "sales": ["doc_date", "item_code", "quantity", "doc_entry", "line_num", "customer_code", "warehouse",
              "unit_price", "discount_pct", "line_total", "channel", "ship_state", "ship_country", "sales_employee"],
    "open_orders": ["item_code", "open_qty", "warehouse", "due_date", "customer_code"],
    "returns": ["doc_date", "item_code", "quantity", "customer_code", "warehouse", "line_total", "reason"],
    "purchases": ["po_entry", "po_date", "item_code", "supplier_code", "quantity", "line_num", "unit_price",
                  "currency", "warehouse", "promised_date", "open_qty", "status"],
    "receipts": ["receipt_date", "item_code", "quantity", "po_entry", "po_line", "supplier_code", "warehouse"],
    "stock": ["item_code", "warehouse", "on_hand", "committed", "on_order", "min_stock", "max_stock"],
    "items": ["item_code", "item_name", "category", "brand", "unit_cost", "list_price", "active", "moq",
              "order_multiple", "lead_time_days", "preferred_supplier"],
    "suppliers": ["supplier_code", "supplier_name", "country"],
    "item_suppliers": ["item_code", "supplier_code", "unit_price"],
}
WHOLE_NUMBER_COLUMNS = ["line_num", "po_line"]

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
# Reads each Silver table and checks it is not empty and has every column listed in `EXPORT_COLUMNS`. Sales is filtered to sales orders. Stops with an error if anything is missing, so a partial export is never written.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Load the Silver tables and confirm columns are present.
# Input   : EXPORT_COLUMNS, SALES_DOC_TYPE
# Output  : tables (dict of pandas DataFrames), source_rows
# Notes   : Tables are small enough (largest ~250k rows) for pandas.
# =====================================================================
tables = {}
source_rows = {}

for name, columns in EXPORT_COLUMNS.items():
    try:
        df = spark.table(f"dbo.{name}")
    except Exception as e:
        raise RuntimeError(f"FAILED: cannot read dbo.{name}. Run its Silver notebook first. Details: {e}")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"FAILED: dbo.{name} is missing columns {missing}.")

    if name == "sales":
        df = df.filter(F.col("doc_type") == SALES_DOC_TYPE)

    pdf = df.select(columns).toPandas()
    if len(pdf) == 0:
        raise RuntimeError(f"FAILED: dbo.{name} has no rows to export.")
    tables[name] = pdf
    source_rows[name] = len(pdf)
    print(f"  {name}: {len(pdf):,} rows, {len(columns)} columns")

print("SUCCESS: all 9 Silver tables read and validated.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Transform and Validate
# Prepares the tables for export and checks them before anything is written.
# 1. Whole-number columns (`line_num`, `po_line`) are converted so the CSV shows `3`, not `3.0`.
# 2. Every table that refers to an item is checked against `dbo.items`, so the CSVs agree with each other.
# 3. Privacy check: supplier names must equal the supplier code (the export has no real names), and no column in the export is called a name, contact or tax ID, apart from the item name.
# 
# If a check fails the notebook stops and nothing is written.

# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Prepare tables for CSV and run consistency/privacy checks.
# Input   : tables, EXPORT_COLUMNS, WHOLE_NUMBER_COLUMNS
# Output  : prepared (dict of DataFrames)
# Notes   : Stops before writing if any check fails.
# =====================================================================
prepared = {}
for name, pdf in tables.items():
    out = pdf.copy()
    for column in WHOLE_NUMBER_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    prepared[name] = out

item_codes = set(tables["items"]["item_code"])
for name, pdf in prepared.items():
    if "item_code" in pdf.columns and name != "items":
        missing = (~pdf["item_code"].isin(item_codes)).sum()
        print(f"  {name}: rows whose item is not in items: {missing:,}")
        if missing:
            raise RuntimeError(f"FAILED: {name} has {missing:,} rows with an item missing from items.")

suppliers = prepared["suppliers"]
if not (suppliers["supplier_name"] == suppliers["supplier_code"]).all():
    raise RuntimeError("FAILED: suppliers.supplier_name holds real names; it should equal the supplier code.")

forbidden = [c for name, pdf in prepared.items() for c in pdf.columns
             if any(w in c.lower() for w in ("customer_name", "contact", "tax", "email", "phone"))]
if forbidden:
    raise RuntimeError(f"FAILED: columns that may hold personal data: {forbidden}")
print("SUCCESS: tables prepared, consistent with items, and free of names and contacts.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write CSV Files
# Writes one CSV per table to `Files/csv_export` (header row, UTF-8, dates as `yyyy-MM-dd`, no row numbers). Each run replaces the previous files. Then every file is read back and its row count and column list are checked against what was exported.

# CELL ********************

# =====================================================================
# 04. Write CSV Files
# Purpose : Save the tables as CSV and verify them.
# Input   : prepared, OUT_DIR
# Output  : 9 CSV files in OUT_DIR
# Notes   : Read-back check stops the run if a file is short or reordered.
# =====================================================================
os.makedirs(OUT_DIR, exist_ok=True)

for name, pdf in prepared.items():
    pdf.to_csv(f"{OUT_DIR}/{name}.csv", index=False, encoding="utf-8", date_format="%Y-%m-%d")

written_rows = {}
for name, pdf in prepared.items():
    check = pd.read_csv(f"{OUT_DIR}/{name}.csv", dtype=str, keep_default_na=False)
    written_rows[name] = len(check)
    if len(check) != len(pdf) or list(check.columns) != EXPORT_COLUMNS[name]:
        raise RuntimeError(f"FAILED: {name}.csv does not match what was exported.")
    print(f"  {name}.csv: {len(check):,} rows verified")
print("SUCCESS: 9 CSV files written and verified.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 05. Execution Summary and Logging
# Prints one line per file (rows and size), confirms the sales file has the same number of rows as the sales-order lines in Silver, and says where the files are. Prints the final status line with the run ID and UTC time. No records are displayed.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the export and print where the files are.
# Input   : written_rows, OUT_DIR
# Output  : Printed summary (no raw records are displayed)
# Notes   : Download the files from Files/csv_export in the lakehouse.
# =====================================================================
total = 0
for name, rows in written_rows.items():
    size_kb = os.path.getsize(f"{OUT_DIR}/{name}.csv") / 1024
    total += rows
    print(f"  {name:<15} {rows:>9,} rows  {size_kb:>10,.0f} KB")
print(f"  {'TOTAL':<15} {total:>9,} rows")

sales_check = spark.table("dbo.sales").filter(F.col("doc_type") == SALES_DOC_TYPE).count()
if written_rows["sales"] != sales_check:
    raise RuntimeError(f"FAILED: sales.csv has {written_rows['sales']:,} rows, expected {sales_check:,} sales-order lines.")
print(f"Sales file holds {SALES_DOC_TYPE} lines only: {written_rows['sales']:,} rows")

print("Files are in: lh_fortpro_silver / Files / csv_export")
print(f"SUCCESS: run {RUN_ID} finished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}, 9 CSV files written.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
