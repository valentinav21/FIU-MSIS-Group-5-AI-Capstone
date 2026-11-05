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

# # NB_Transform_Bronze_Receipts_To_Silver
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-11-04
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Cleans the goods receipt lines (stock actually received from suppliers) into one Silver table and recovers the purchase order each receipt belongs to, so observed lead times can be measured.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Tables/dbo/goods_receipts and purchase_orders (read by OneLake path)
# 
# **Destination:** lh_fortpro_silver / dbo.receipts
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver. Read access to the Landing workspace. Upstream: NB_Bronze_Load_Raw_To_Tables. For the lead-time check in section 05, NB_Transform_Bronze_Purchases_To_Silver and NB_Transform_Bronze_Items_To_Silver should have run first.
# 
# **Change History:**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-11-04 | Dagiana Toussaint | Initial notebook creation |

# MARKDOWN ********************

# ## 01. Configuration
# Imports and settings. Change paths and business rules here only, not in the code below.
# - `SOURCE_TABLE` is the goods receipt file. It mixes two document types: `OPDN` (goods receipts, what we want) and `OPCH` (A/P invoices for the same purchases). Counting both would double the received units, so only `OPDN` is a receipt.
# - `PO_TABLE` is the raw purchase order table. It is only used to check that a linked PO line is for the same item.
# - Base type codes in SAP: `22` = purchase order, `18` = A/P invoice.
# - Warehouse rules match the other Silver tables.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep paths, table names and business rules in one place.
# Input   : None
# Output  : BRONZE_TABLES, SOURCE_TABLE, PO_TABLE, TARGET_TABLE, BASE_PO,
#           BASE_INVOICE, ACTIVE_WAREHOUSES, DEFAULT_WAREHOUSE, RUN_ID, RUN_UTC
# Notes   : RUN_ID and RUN_UTC label this run in the printed log.
# =====================================================================
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

BRONZE_TABLES = "abfss://814a9fdc-9de6-48f7-9ce8-d92ad6f849ea@onelake.dfs.fabric.microsoft.com/d1c58e08-ec7c-4c5f-b461-ea23feb4d875/Tables/dbo"
SOURCE_TABLE = "goods_receipts"
PO_TABLE = "purchase_orders"
TARGET_TABLE = "dbo.receipts"

BASE_PO = 22        # SAP base type: purchase order
BASE_INVOICE = 18   # SAP base type: A/P invoice

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
# Reads the goods receipt table and the purchase order table from Bronze. Stops with an error if either cannot be read, is empty, or is missing a column this notebook needs. Also shows the document types in the receipt file: expected `OPDN` (receipts) and `OPCH` (invoices).

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm both sources are readable and have the expected columns.
# Input   : BRONZE_TABLES, SOURCE_TABLE, PO_TABLE
# Output  : src, src_rows, po_lines
# Notes   : Fails loudly instead of continuing with a partial source.
# =====================================================================
REQUIRED_RECEIPT_COLUMNS = [
    "DocType", "DocEntry", "LineNum", "BaseType", "BaseEntry", "BaseLine",
    "DocDate", "ItemCode", "SupplierCode", "Quantity", "WhsCode", "CANCELED",
]
REQUIRED_PO_COLUMNS = ["DocEntry", "LineNum", "ItemCode"]

try:
    src = spark.read.format("delta").load(f"{BRONZE_TABLES}/{SOURCE_TABLE}")
    src_rows = src.count()
    po_lines = spark.read.format("delta").load(f"{BRONZE_TABLES}/{PO_TABLE}")
except Exception as e:
    raise RuntimeError(
        f"FAILED: cannot read a source from Bronze. Check the path in 01 and that you "
        f"have read access to the Landing workspace. Details: {e}"
    )

missing_columns = [c for c in REQUIRED_RECEIPT_COLUMNS if c not in src.columns]
missing_columns += [c for c in REQUIRED_PO_COLUMNS if c not in po_lines.columns]
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
# Builds the receipts table in six steps and prints what each one does.
# 1. Keep goods receipts only (`OPDN`) and drop canceled ones. The A/P invoice lines (`OPCH`) are set aside, but used in step 2.
# 2. **Find the purchase order each receipt belongs to.** About half of the receipts point straight to the PO. The others point to an A/P invoice line, and that invoice line points to the PO. So the link is: receipt, then invoice line, then PO line.
# 3. **Check the link.** The PO line found must be for the same item as the receipt. If it is not, the link is removed (the receipt itself is kept).
# 4. Drop lines with quantity zero or less.
# 5. Warehouse: trim and upper-case, blank becomes the default. Nothing is dropped; non-active warehouses are only counted.
# 6. Rename columns to the planner's names and drop lines with no item or date.
# 
# Unlinked receipts stay in the table with an empty `po_entry`. They still count as stock received, they just cannot be used to measure lead time.


# CELL ********************

# =====================================================================
# 03. Transform and Validate
# Purpose : Clean receipts and recover the PO link (direct or via invoice).
# Input   : src, po_lines, BASE_PO, BASE_INVOICE, warehouse settings
# Output  : silver (Spark DataFrame), silver_rows
# Notes   : Join keys are cast to long so text vs number types cannot break
#           the matches. A duplicate-match check protects the row count.
# =====================================================================
def to_long(name):
    """Cast a SAP key column to a whole number under its own name."""
    return F.col(name).cast("long").alias(name)

docs = src.select(
    "DocType", to_long("DocEntry"), to_long("LineNum"), to_long("BaseType"),
    to_long("BaseEntry"), to_long("BaseLine"), "DocDate", "ItemCode",
    "SupplierCode", "Quantity", "WhsCode", "CANCELED",
)

# Step 1: goods receipts only, not canceled (a missing flag counts as not canceled).
receipts = docs.filter((F.col("DocType") == "OPDN") & (F.coalesce(F.col("CANCELED"), F.lit("N")) != "Y"))
receipt_rows = receipts.count()
print(f"1. goods receipt lines (OPDN, not canceled): {receipt_rows:,} (of {src_rows:,} lines in the file)")

# Step 2: link to the purchase order, directly or through the A/P invoice line.
invoices = docs.filter(F.col("DocType") == "OPCH").select(
    F.col("DocEntry").alias("inv_entry"), F.col("LineNum").alias("inv_line"),
    F.col("BaseType").alias("inv_base_type"), F.col("BaseEntry").alias("inv_base_entry"),
    F.col("BaseLine").alias("inv_base_line"),
)
linked = receipts.join(
    invoices,
    (receipts["BaseEntry"] == invoices["inv_entry"]) & (receipts["BaseLine"] == invoices["inv_line"]),
    "left",
)
if linked.count() != receipt_rows:
    raise RuntimeError("FAILED: the invoice join duplicated receipt lines (invoice keys are not unique).")

direct = F.col("BaseType") == BASE_PO
via_invoice = (F.col("BaseType") == BASE_INVOICE) & (F.col("inv_base_type") == BASE_PO)
linked = linked.withColumn(
    "po_entry", F.when(direct, F.col("BaseEntry")).when(via_invoice, F.col("inv_base_entry"))
).withColumn(
    "po_line", F.when(direct, F.col("BaseLine")).when(via_invoice, F.col("inv_base_line"))
)
n_direct = linked.filter(direct).count()
n_via = linked.filter(~direct & via_invoice).count()
print(f"2. PO link found: {n_direct:,} directly ({n_direct / receipt_rows:.1%}), "
      f"{n_via:,} through an A/P invoice ({n_via / receipt_rows:.1%})")

# Step 3: the linked PO line must be for the same item, otherwise remove the link.
po_check = po_lines.select(
    to_long("DocEntry").alias("chk_entry"), to_long("LineNum").alias("chk_line"),
    F.trim(F.col("ItemCode")).alias("po_item"),
)
linked = linked.join(
    po_check,
    (linked["po_entry"] == po_check["chk_entry"]) & (linked["po_line"] == po_check["chk_line"]),
    "left",
)
bad_link = F.col("po_entry").isNotNull() & ~F.col("po_item").eqNullSafe(F.trim(F.col("ItemCode")))
n_bad = linked.filter(bad_link).count()
linked = linked.withColumn("po_entry", F.when(bad_link, F.lit(None)).otherwise(F.col("po_entry"))) \
               .withColumn("po_line", F.when(bad_link, F.lit(None)).otherwise(F.col("po_line")))
n_linked = linked.filter(F.col("po_entry").isNotNull()).count()
print(f"3. links removed because the PO line is a different item or was not found: {n_bad:,}")
print(f"   receipts with a PO link: {n_linked:,} ({n_linked / receipt_rows:.1%}); without: {receipt_rows - n_linked:,}")

# Step 4: zero or negative quantities.
step = linked.withColumn("quantity", F.col("Quantity").cast("double")).filter(F.col("quantity") > 0)
rows = step.count()
print(f"4. quantity <= 0: {receipt_rows - rows:,} removed, {rows:,} left")

# Step 5: warehouse code. Nothing is dropped; non-active warehouses are only counted.
whs = F.upper(F.trim(F.col("WhsCode")))
step = step.withColumn(
    "warehouse", F.when(whs.isNull() | (whs == ""), F.lit(DEFAULT_WAREHOUSE)).otherwise(whs)
)
other_whs = step.filter(~F.col("warehouse").isin(ACTIVE_WAREHOUSES)).count()
print(f"5. warehouse cleaned. Lines in non-active warehouses (kept): {other_whs:,}")

# Step 6: planner column names; drop lines with no item or date.
silver = step.select(
    F.to_date("DocDate").alias("receipt_date"),
    F.trim(F.col("ItemCode")).alias("item_code"),
    "quantity",
    F.col("po_entry").cast("string").alias("po_entry"),
    F.col("po_line").cast("int").alias("po_line"),
    F.trim(F.col("SupplierCode")).alias("supplier_code"),
    "warehouse",
).filter(F.col("item_code").isNotNull() & (F.col("item_code") != "") & F.col("receipt_date").isNotNull())
silver_rows = silver.count()
print(f"6. missing item or date: {rows - silver_rows:,} removed, {silver_rows:,} left")
print(f"Result: {src_rows:,} Bronze rows -> {silver_rows:,} Silver rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Write Silver Table
# Writes the result to `lh_fortpro_silver.dbo.receipts`, replacing the previous run (safe to re-run), then reads the table back and confirms the row count matches what was written.

# CELL ********************

# =====================================================================
# 04. Write Silver Table
# Purpose : Save the receipts table and verify what was saved.
# Input   : silver, silver_rows, TARGET_TABLE
# Output  : Delta table dbo.receipts in lh_fortpro_silver
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
# Checks the finished table: date range, items, suppliers, warehouses, the share of receipts linked to a PO, and empty values in key columns.
# 
# It also measures the **observed lead time**: days from the PO date to the receipt date, for receipts linked to a PO line in `dbo.purchases`. SAP has no lead time for most items, so this is the lead time the planner can rely on. Prints one final status line with the run ID and UTC time.

# CELL ********************

# =====================================================================
# 05. Execution Summary and Logging
# Purpose : Summarize the output, measure lead time, fail on key gaps.
# Input   : dbo.receipts, dbo.purchases, dbo.items
# Output  : Printed summary (no raw records are displayed)
# Notes   : Lead time is per receipt line (partial deliveries count each).
# =====================================================================
result = spark.table(TARGET_TABLE)

result.agg(
    F.min("receipt_date").alias("first_receipt"),
    F.max("receipt_date").alias("last_receipt"),
    F.countDistinct("item_code").alias("items"),
    F.countDistinct("supplier_code").alias("suppliers"),
    F.sum("quantity").alias("units_received"),
).show()

print("Rows by warehouse:")
result.groupBy("warehouse").count().orderBy(F.desc("count")).show()

with_po = result.filter(F.col("po_entry").isNotNull()).count()
print(f"Receipts linked to a PO: {with_po:,} of {written_rows:,} ({with_po / written_rows:.1%})")

# Observed lead time: PO date -> receipt date.
purchases = spark.table("dbo.purchases").select(
    F.col("po_entry").alias("p_entry"), F.col("line_num").cast("int").alias("p_line"), "po_date"
)
timed = result.join(
    purchases, (result["po_entry"] == purchases["p_entry"]) & (result["po_line"] == purchases["p_line"]), "inner"
).withColumn("lead_days", F.datediff("receipt_date", "po_date"))
stats = timed.agg(
    F.count("*").alias("n"),
    F.expr("percentile(lead_days, 0.5)").alias("median"),
    F.expr("percentile(lead_days, 0.9)").alias("p90"),
    F.sum((F.col("lead_days") < 0).cast("int")).alias("negative"),
).first()
print(f"Observed lead time on {stats['n']:,} linked receipts: median {stats['median']:.0f} days, "
      f"p90 {stats['p90']:.0f} days; receipts dated before their PO: {stats['negative']:,}")

items_missing = result.join(spark.table("dbo.items").select("item_code"), "item_code", "left_anti").count()
print(f"Receipt lines whose item is not in dbo.items: {items_missing:,}")

KEY_COLUMNS = ["receipt_date", "item_code", "quantity", "warehouse"]
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
