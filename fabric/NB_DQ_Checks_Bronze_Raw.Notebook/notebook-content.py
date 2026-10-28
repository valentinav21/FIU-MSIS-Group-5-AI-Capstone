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
# META       "default_lakehouse_workspace_id": "e2b0b198-404d-4a39-887c-6eea3fa6ae4d",
# META       "known_lakehouses": [
# META         {
# META           "id": "73166bec-8d01-49be-b864-6c93bbc66b49"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # NB_DQ_Checks_Bronze_Raw
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-10-24
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Data quality checks on the raw SAP Business One extracts: key nulls, orphan keys, duplicate documents, date gaps and invalid values. Read-only; it does not modify any data.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Files/raw, read through the bronze_raw shortcut in lh_fortpro_silver
# 
# **Destination:** None (results are shown in the notebook output only)
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver with the bronze_raw shortcut. Follows NB_Profile_Bronze_Raw (Jira FP-16). Jira: FP-24.
# 
# **Change History**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-10-24 | Dagiana Toussaint | Initial notebook creation |
# | 2026-10-27 | Dagiana Toussaint | Added sections 03-08: keys, orphans, empty codes, dates and invalid values, date gaps, summary |

# MARKDOWN ********************

# ## 01. Configuration
# Imports and settings. Change paths here only. `bronze_raw` is a shortcut to the protected raw files in the Landing workspace. Also defines `log_issue`, which collects every finding for the summary at the end.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep imports, settings and the issue log in one place.
# Input   : None
# Output  : RAW_PATH, EXPECTED_FILE_COUNT, issues, log_issue()
# Notes   : RAW_PATH points at the bronze_raw shortcut, so this notebook
#           reads Landing's files without copying or changing them.
# =====================================================================
from notebookutils import mssparkutils
from pyspark.sql import functions as F
import pandas as pd

RAW_PATH = "Files/bronze_raw"   # shortcut to Landing > lh_fortpro_bronze > Files/raw
EXPECTED_FILE_COUNT = 18        # per the data guide; update if the extract changes

# Every check appends its findings here; section 08 turns them into a summary.
issues = []

def log_issue(table, check, affected_rows, total_rows, severity, note=""):
    """Record one data quality finding.

    Args:
        table: raw file name the finding belongs to.
        check: short name of the check (e.g. "null key: ItemCode").
        affected_rows: number of rows that fail the check.
        total_rows: total rows in the table, to compute the percentage.
        severity: "High", "Medium" or "Low".
        note: optional detail, no raw sensitive values.
    Side effect: appends a dict to the global `issues` list.
    """
    pct = round(100 * affected_rows / total_rows, 2) if total_rows else None
    issues.append({"table": table, "check": check, "affected_rows": int(affected_rows),
                   "total_rows": int(total_rows), "pct": pct,
                   "severity": severity, "note": note})

# Confirm the shortcut is reachable before running any checks.
# Fails loudly if the path or permissions are wrong.
try:
    shortcut_items = mssparkutils.fs.ls(RAW_PATH)
    print(f"SUCCESS: {RAW_PATH} is reachable ({len(shortcut_items)} items found).")
except Exception as e:
    raise RuntimeError(
        f"FAILED: cannot read {RAW_PATH}. Check that the bronze_raw shortcut "
        f"exists in lh_fortpro_silver and that you have read access to the "
        f"Landing workspace. Details: {e}"
    )

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 02. Source Validation
# Confirms the expected Parquet files exist, loads each one into a dictionary and counts rows once so every later check reuses them. Expected: 18 files.

# CELL ********************

# =====================================================================
# 02. Source Validation
# Purpose : Confirm the raw files exist, load them once, count rows once.
# Input   : RAW_PATH
# Output  : dfs (name -> DataFrame), row_counts (name -> rows)
# Notes   : Non-Parquet files are ignored. Fails if the file count
#           differs from what the extract should contain.
# =====================================================================
raw_files = [f for f in mssparkutils.fs.ls(RAW_PATH) if f.name.endswith(".parquet")]

if len(raw_files) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        f"FAILED: found {len(raw_files)} Parquet files, expected {EXPECTED_FILE_COUNT}."
    )

dfs = {f.name: spark.read.parquet(f"{RAW_PATH}/{f.name}") for f in raw_files}
row_counts = {name: df.count() for name, df in dfs.items()}

print(f"SUCCESS: loaded {len(dfs)} Parquet files, {sum(row_counts.values()):,} rows in total.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Print the columns of each table so the key and join checks use real names.
for name, df in dfs.items():
    print(f"\n{name}")
    print(", ".join(df.columns))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Key Nulls and Uniqueness
# For each table, checks that the business key is never empty and never repeated. A repeated key means the same document line, item or supplier appears twice, which would double count in sales, PO and stock numbers. Goods receipts include DocType in the key because receipts and invoices share DocEntry numbers.

# CELL ********************

# =====================================================================
# 03. Key Nulls and Uniqueness
# Purpose : Find empty or repeated business keys in each table.
# Input   : dfs, row_counts, log_issue()
# Output  : key_results (one row per table), findings added to issues
# Notes   : Keys are the SAP line identifiers (DocEntry + LineNum, etc.).
#           Rows with an empty key part are counted as null-key rows and
#           left out of the duplicate check so they are not counted twice.
#           Tables 00, 01 and 02 are added once their columns are known.
# =====================================================================
KEYS = {
    "00_udf":       ["TableID", "AliasID"],
    "01_sales":     ["DocType", "DocEntry", "LineNum"],   # orders, deliveries and invoices share DocEntry numbers
    "02_returns":   ["DocType", "DocEntry", "LineNum"],   # returns and credit memos share DocEntry numbers
    "03_item":      ["ItemCode"],
    "04_warehouses": ["WhsCode"],
    "05_stock":     ["ItemCode", "WhsCode"],
    "06_inventory": ["TransNum"],
    "07_inventory": ["DocEntry", "LineNum"],
    "08_purchase":  ["DocEntry", "LineNum"],
    "09_goods":     ["DocType", "DocEntry", "LineNum"],   # PDN and PCH share DocEntry numbers
    "10_supplier_returns": ["DocEntry", "LineNum"],
    "11_supplier_master":  ["SupplierCode"],
    "12_item_supplier":    ["Source", "ItemCode", "SupplierCode"],
    "13_landed_cost_lines": ["DocEntry", "LineNum"],
    "14_landed_cost_charges": ["DocEntry", "AlcCode"],
    "15_sales_quotations": ["DocEntry", "LineNum"],
    "16_sap_forecasts":    ["AbsID", "LineID"],
    "17_series":    ["Kind", "Code"],
}

def find_table(prefix):
    """Return the raw file name that starts with the given prefix."""
    matches = [n for n in dfs if n.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one table for prefix {prefix}, found {matches}")
    return matches[0]

def is_empty(col_name):
    """Spark condition: column is null or only blanks."""
    c = F.col(f"`{col_name}`")
    return c.isNull() | (F.trim(c.cast("string")) == "")

results = []
for prefix, key_cols in KEYS.items():
    name = find_table(prefix)
    df, total = dfs[name], row_counts[name]

    # Rows where any part of the key is empty.
    any_key_empty = F.lit(False)
    for k in key_cols:
        any_key_empty = any_key_empty | is_empty(k)
    null_key_rows = df.where(any_key_empty).count()

    # Repeated keys among rows that do have a full key.
    valid = df.where(~any_key_empty)
    duplicate_key_rows = valid.count() - valid.select(*key_cols).distinct().count()

    results.append({"table": name, "key": " + ".join(key_cols), "rows": total,
                    "null_key_rows": null_key_rows, "duplicate_key_rows": duplicate_key_rows})

    if null_key_rows > 0:
        log_issue(name, f"null key: {' + '.join(key_cols)}", null_key_rows, total, "High")
    if duplicate_key_rows > 0:
        log_issue(name, f"duplicate key: {' + '.join(key_cols)}", duplicate_key_rows, total, "High")

key_results = pd.DataFrame(results)
display(key_results)
print(f"Tables with null keys: {(key_results['null_key_rows'] > 0).sum()}")
print(f"Tables with duplicate keys: {(key_results['duplicate_key_rows'] > 0).sum()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Orphan Keys
# Checks that every ItemCode, WhsCode and SupplierCode used in a transaction table exists in its master table (item master, warehouses, supplier master). An orphan row cannot be joined to item or supplier attributes, so it would drop out of any forecast or lead-time model built on joins. Empty codes are counted separately.

# CELL ********************

# =====================================================================
# 04. Orphan Keys
# Purpose : Find rows whose item, warehouse or supplier code is missing
#           from the master table it should point to.
# Input   : dfs, row_counts, find_table(), is_empty(), log_issue()
# Output  : link_results (one row per table and column), findings in issues
# Notes   : Only non-empty codes are tested; empty codes are reported as
#           null_code_rows. Orphan severity: High if 1% or more of rows,
#           Medium if 0.1% or more, otherwise Low. Empty-code severity:
#           Medium if 10% or more of rows, otherwise Low (section 05
#           explains them as service/charge lines). Compares distinct
#           master values with a join, so the large tables are never
#           collected to the driver.
# =====================================================================
ITEM, WHS, SUP = ("03_item", "ItemCode"), ("04_warehouses", "WhsCode"), ("11_supplier_master", "SupplierCode")

LINKS = [
    # (child table prefix, child column, (parent table prefix, parent column))
    ("01_sales", "ItemCode", ITEM),          ("01_sales", "WhsCode", WHS),
    ("02_returns", "ItemCode", ITEM),        ("02_returns", "WhsCode", WHS),
    ("03_item", "PrefVendorCode", SUP),
    ("05_stock", "ItemCode", ITEM),          ("05_stock", "WhsCode", WHS),
    ("06_inventory", "ItemCode", ITEM),      ("06_inventory", "Warehouse", WHS),
    ("07_inventory", "ItemCode", ITEM),      ("07_inventory", "FromWhsCod", WHS),
    ("07_inventory", "ToWhs", WHS),
    ("08_purchase", "ItemCode", ITEM),       ("08_purchase", "WhsCode", WHS),
    ("08_purchase", "SupplierCode", SUP),
    ("09_goods", "ItemCode", ITEM),          ("09_goods", "WhsCode", WHS),
    ("09_goods", "SupplierCode", SUP),
    ("10_supplier_returns", "ItemCode", ITEM), ("10_supplier_returns", "WhsCode", WHS),
    ("10_supplier_returns", "SupplierCode", SUP),
    ("12_item_supplier", "ItemCode", ITEM),  ("12_item_supplier", "SupplierCode", SUP),
    ("13_landed_cost_lines", "ItemCode", ITEM), ("13_landed_cost_lines", "WhsCode", WHS),
    ("15_sales_quotations", "ItemCode", ITEM), ("15_sales_quotations", "WhsCode", WHS),
    ("16_sap_forecasts", "ItemCode", ITEM),  ("16_sap_forecasts", "WhsCode", WHS),
]

# Distinct master values, trimmed and cast to string so both sides compare equally.
master_keys = {}
for prefix, col in {ITEM, WHS, SUP}:
    master_keys[(prefix, col)] = (dfs[find_table(prefix)]
                                  .select(F.trim(F.col(col).cast("string")).alias("k"))
                                  .distinct())

link_rows = []
for child_prefix, child_col, (parent_prefix, parent_col) in LINKS:
    name = find_table(child_prefix)
    df, total = dfs[name], row_counts[name]

    null_code_rows = df.where(is_empty(child_col)).count()

    # Rows with a code that has no match in the master (left anti join).
    codes = (df.where(~is_empty(child_col))
               .select(F.trim(F.col(child_col).cast("string")).alias("k")))
    orphans = codes.join(master_keys[(parent_prefix, parent_col)], on="k", how="left_anti")
    orphan_rows = orphans.count()
    distinct_orphans = orphans.distinct().count()

    link_rows.append({"table": name, "column": child_col,
                      "parent": f"{parent_prefix}.{parent_col}", "rows": total,
                      "null_code_rows": null_code_rows,
                      "orphan_rows": orphan_rows, "distinct_orphan_codes": distinct_orphans})

    if orphan_rows > 0:
        orphan_pct = 100 * orphan_rows / total
        severity = "High" if orphan_pct >= 1 else "Medium" if orphan_pct >= 0.1 else "Low"
        log_issue(name, f"orphan {child_col} (not in {parent_prefix})", orphan_rows, total, severity,
                  f"{distinct_orphans} distinct codes")
    if null_code_rows > 0:
        empty_pct = 100 * null_code_rows / total
        note = ("optional field: no preferred vendor set" if child_col == "PrefVendorCode"
                else "likely service/charge lines, see section 05")
        log_issue(name, f"empty {child_col}", null_code_rows, total,
                  "Medium" if empty_pct >= 10 else "Low", note)

link_results = pd.DataFrame(link_rows)
link_results["orphan_pct"] = (100 * link_results["orphan_rows"] / link_results["rows"]).round(2)
display(link_results.sort_values("orphan_rows", ascending=False))
print(f"Links checked: {len(link_results)}")
print(f"Links with orphan codes: {(link_results['orphan_rows'] > 0).sum()}")
print(f"Links with empty codes: {(link_results['null_code_rows'] > 0).sum()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Findings: Orphan Keys**
# 
# 29 links checked (item, warehouse and supplier codes against their master tables). 3 links have orphan codes and 11 have empty codes.
# 
# **Orphan codes (code present, but not in the master table):**
# 
# | Table | Column | Orphan rows | Distinct codes | Share of rows |
# |---|---|---|---|---|
# | 16_sap_forecasts | ItemCode | 9 | 4 | under 0.01% |
# | 16_sap_forecasts | WhsCode | 5 | 1 | under 0.01% |
# | 15_sales_quotations | WhsCode | 1 | 1 | under 0.01% |
# 
# - Negligible. Likely old forecast or quote lines that reference an item or warehouse that was later deleted or renamed. Some forecasts date back to 2017.
# - Treatment in Silver: map these rows to an "Unknown" member or exclude them from joins, and note it in the data quality report.
# 
# **Empty codes (no code at all, not counted as orphans):**
# - 11 links are affected: sales (ItemCode, WhsCode), returns and credit memos (ItemCode, WhsCode), goods receipts and supplier invoices (ItemCode, WhsCode), purchase orders (ItemCode, WhsCode), quotations (ItemCode, WhsCode) and the item master (PrefVendorCode, 30.8% empty).
# - Section 05 shows the item and warehouse codes are empty on service or charge lines (quantity 0), not on product lines.
# - An empty PrefVendorCode means the item has no preferred supplier set in SAP. It is an optional field, not an error.
# 
# **Conclusion:** every code that is filled in points to a real item, warehouse or supplier, apart from the 15 rows above. Joins on these keys are safe.


# MARKDOWN ********************

# ## 05. Empty Codes by Document Type
# Explains the empty item and warehouse codes found in section 04. Splits the rows by document type and canceled flag to test whether they are service lines, canceled documents or a real data gap. Also lists the five PO lines with no item code.

# CELL ********************

# =====================================================================
# 05. Empty Codes by Document Type
# Purpose : Find out which document types carry the empty ItemCode rows
#           from section 04 (sales, returns, goods receipts).
# Input   : dfs, find_table(), is_empty()
# Output  : breakdown tables per document group, empty PO lines
# Notes   : Shows document identifiers only, no customer or supplier data.
#           Prints the CANCELED values first so later checks use the real
#           flag values instead of an assumed "Y"/"N".
# =====================================================================
empty_item = is_empty("ItemCode")

for prefix in ["01_sales", "02_returns", "09_goods"]:
    name = find_table(prefix)
    df = dfs[name]
    print(f"\n{name}")
    breakdown = (df.groupBy("DocType", "CANCELED")
                   .agg(F.count("*").alias("rows"),
                        F.sum(empty_item.cast("int")).alias("empty_item_rows"))
                   .withColumn("empty_pct",
                               F.round(100 * F.col("empty_item_rows") / F.col("rows"), 1))
                   .orderBy("DocType", "CANCELED")
                   .toPandas())
    display(breakdown)

# The five PO lines without an item code: which documents are they?
po_name = find_table("08_purchase")
print(f"\nPO lines with empty ItemCode ({po_name})")
display(dfs[po_name].where(empty_item)
        .select("DocEntry", "LineNum", "DocDate", "CANCELED", "DocStatus",
                "LineStatus", "Quantity", "LineTotal")
        .toPandas())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# =====================================================================
# 05b. Evidence: Are the Empty-Item Lines Service or Charge Lines?
# Purpose : Test whether empty-ItemCode lines have quantity 0 and no
#           source document, as service or charge lines would.
# Input   : dfs, find_table(), is_empty()
# Output  : evidence table per document type
# Notes   : Counts only, no identifiers or amounts.
# =====================================================================
for prefix in ["01_sales", "02_returns", "09_goods"]:
    name = find_table(prefix)
    evidence = (dfs[name].where(empty_item)
                  .groupBy("DocType")
                  .agg(F.count("*").alias("empty_item_rows"),
                       F.sum((F.col("Quantity").cast("double") == 0).cast("int")).alias("qty_zero_rows"),
                       F.sum((F.col("BaseType").cast("int") == -1).cast("int")).alias("no_base_doc_rows"))
                  .orderBy("DocType")
                  .toPandas())
    print(f"\n{name}")
    display(evidence)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Findings: Empty Codes by Document Type (reviewed 2026-10-27)**
# 
# - Empty ItemCode and WhsCode rows are concentrated in invoice-type documents: supplier invoices (OPCH) 7,051 of 19,301 lines (36.5%), credit memos (ORIN) 1,212 of 6,761 (17.9%), sales invoices (OINV) 482 of 239,650 (0.2%).
# - Documents that move stock (deliveries, goods receipts, customer returns) have no empty codes.
# - Canceled documents are not the cause: only 1 canceled order line is empty.
# - Evidence: all 482 empty sales-invoice lines and all 7,051 empty supplier-invoice lines have quantity 0, and 7,046 of the supplier-invoice lines have no base document (direct invoices). The five empty PO lines are closed, not canceled, quantity 0, with a nonzero value.
# - Conclusion: these are service or charge lines (freight, fees, adjustments), not lost product data. Pending confirmation from FORTPRO.
# - Treatment: keep them in Silver flagged as service lines; exclude them from demand and forecasting.

# MARKDOWN ********************

# ## 06. Dates and Invalid Values
# Checks that document dates fall inside the extract window (2023-01-01 to the stock snapshot date), that orders and purchase orders were entered close to their business date, and that quantities and prices are not impossible. Item-less service lines are left out of the zero-quantity and zero-price checks, because they legitimately have quantity 0.

# CELL ********************

# =====================================================================
# 06. Dates and Invalid Values
# Purpose : Check document dates against the extract window and entry
#           latency, and look for impossible quantities and prices.
# Input   : dfs, find_table(), is_empty(), log_issue()
# Output  : date_results, value_results, findings added to issues
# Notes   : Window is 2023-01-01 to the stock SnapshotDate (extract date).
#           FORTPRO states orders are entered immediately, so a CreateDate
#           more than 7 days after DocDate, or before it, is flagged.
#           Severity is percent-based: High >= 1%, Medium >= 0.1%, Low > 0.
# =====================================================================
START_DATE = "2023-01-01"
extract_date = dfs[find_table("05_stock")].agg(F.max("SnapshotDate")).first()[0]
print(f"Extract date (stock snapshot): {extract_date}")

def sev(affected, total):
    pct = 100 * affected / total if total else 0
    return "High" if pct >= 1 else "Medium" if pct >= 0.1 else "Low"

date_rows, value_rows = [], []

for prefix in ["01_sales", "02_returns", "06_inventory", "08_purchase",
               "09_goods", "10_supplier_returns", "13_landed_cost_lines",
               "15_sales_quotations"]:
    name = find_table(prefix)
    df = dfs[name]
    total = row_counts[name]
    cols = df.columns
    doc = F.col("DocDate").cast("date")

    # Date range and out-of-window rows
    stats = df.agg(F.min(doc).alias("min"), F.max(doc).alias("max"),
                   F.sum((doc > F.lit(str(extract_date)).cast("date")).cast("int")).alias("future"),
                   F.sum((doc < F.lit(START_DATE).cast("date")).cast("int")).alias("early")).first()
    date_rows.append((name, stats["min"], stats["max"], stats["future"], stats["early"]))
    if stats["future"]:
        log_issue(name, "DocDate after extract date", stats["future"], total,
                  sev(stats["future"], total), f"latest DocDate {stats['max']}")
    if stats["early"]:
        log_issue(name, "DocDate before 2023-01-01", stats["early"], total,
                  sev(stats["early"], total))

    # Entry latency (CreateDate vs DocDate)
    if "CreateDate" in cols:
        gap = F.datediff(F.col("CreateDate").cast("date"), doc)
        late = df.where(gap > 7).count()
        neg = df.where(gap < 0).count()
        if late: log_issue(name, "CreateDate > 7 days after DocDate", late, total, sev(late, total))
        if neg:  log_issue(name, "CreateDate before DocDate", neg, total, sev(neg, total))

    # Impossible values
    has_item = ~is_empty("ItemCode") if "ItemCode" in cols else F.lit(True)
    for col, label in [("Quantity", "negative quantity"), ("Price", "negative price"),
                       ("LineTotal", "negative line total")]:
        if col in cols:
            n = df.where(F.col(col).cast("double") < 0).count()
            value_rows.append((name, label, n))
            if n: log_issue(name, label, n, total, sev(n, total))
    if "Quantity" in cols and "ItemCode" in cols:
        n = df.where(has_item & (F.col("Quantity").cast("double") == 0)).count()
        value_rows.append((name, "zero quantity on item lines", n))
        if n: log_issue(name, "zero quantity on item lines", n, total, sev(n, total))
    if "Price" in cols and "ItemCode" in cols:
        n = df.where(has_item & (F.col("Price").cast("double") == 0)).count()
        value_rows.append((name, "zero price on item lines (info)", n))
        if n: log_issue(name, "zero price on item lines", n, total, "Low", "possible free or warranty items")

date_results = pd.DataFrame(date_rows, columns=["table", "min_date", "max_date", "after_extract", "before_start"])
value_results = pd.DataFrame(value_rows, columns=["table", "check", "rows"])
display(date_results)
display(value_results)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 07. Date Gaps
# Counts rows per month for the main transaction tables and flags months with no rows at all or with unusually low volume (under 25% of the table's median month). The current partial month is excluded. Missing months would break the demand history, so they are logged for follow-up with FORTPRO.

# CELL ********************

# =====================================================================
# 07. Date Gaps
# Purpose : Find months with no rows, or very low volume, in the main
#           transaction tables.
# Input   : dfs, row_counts, find_table(), log_issue(), extract_date (06)
# Output  : gap_summary (one row per table and document type), monthly
#           (counts per month), findings added to issues
# Notes   : Only row counts per month are collected, no client data.
#           The current partial month is excluded. "Low" means under 25%
#           of the table's median month; December or holiday dips may be
#           genuine seasonality and are reviewed, not assumed errors.
# =====================================================================
GAP_TABLES = [("01_sales", "DocType"), ("02_returns", "DocType"),
              ("06_inventory", None), ("08_purchase", None), ("09_goods", "DocType")]
LOW_SHARE = 0.25
last_full_month = pd.Period(str(extract_date)[:7], freq="M") - 1

gap_rows, monthly_frames = [], []

for prefix, split_col in GAP_TABLES:
    name = find_table(prefix)
    df = dfs[name]
    month = F.date_format(F.col("DocDate").cast("date"), "yyyy-MM").alias("month")
    group_cols = [month] + ([F.col(split_col)] if split_col else [])
    counts = df.groupBy(*group_cols).count().toPandas()
    if split_col is None:
        counts[split_col or "DocType"] = "all"
        split = "DocType"
    else:
        split = split_col

    for doc_type, part in counts.groupby(split):
        s = part.set_index(pd.PeriodIndex(part["month"], freq="M"))["count"].sort_index()
        full = pd.period_range(s.index.min(), last_full_month, freq="M")
        s = s.reindex(full, fill_value=0)
        median = s.median()
        missing = [str(m) for m in s[s == 0].index]
        low = [str(m) for m in s[(s > 0) & (s < LOW_SHARE * median)].index]
        gap_rows.append((name, doc_type, len(s), int(median), missing, low))
        monthly_frames.append(s.rename(f"{prefix}|{doc_type}"))
        if missing:
            log_issue(name, f"months with no rows ({doc_type})", len(missing), len(s),
                      "High", "months: " + ", ".join(missing))
        if low:
            log_issue(name, f"months with very low volume ({doc_type})", len(low), len(s),
                      "Low", "months: " + ", ".join(low))

gap_summary = pd.DataFrame(gap_rows, columns=["table", "doc_type", "months", "median_rows",
                                              "missing_months", "low_months"])
monthly = pd.concat(monthly_frames, axis=1).fillna(0).astype(int)
display(gap_summary)
display(monthly)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 08. Summary
# Turns the issue log into one table, sorted by severity and then by the share of rows affected. It contains counts and table names only, no client data. This is the input for the data quality report.

# CELL ********************

# =====================================================================
# 08. Summary
# Purpose : Show every finding from sections 03-07 in one place.
# Input   : issues, row_counts
# Output  : issues_df (all findings), severity counts
# Notes   : Sorted High > Medium > Low, then by % of rows affected.
#           Counts and table names only, no row-level client data.
# =====================================================================
SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}

issues_df = pd.DataFrame(issues)
issues_df["table"] = issues_df["table"].str.replace(".parquet", "", regex=False)
issues_df["rank"] = issues_df["severity"].map(SEV_ORDER)
issues_df = (issues_df.sort_values(["rank", "pct"], ascending=[True, False])
                      .drop(columns="rank").reset_index(drop=True))

print(f"Tables checked: {len(row_counts)} | Rows checked: {sum(row_counts.values()):,}")
print(f"Findings logged: {len(issues_df)}")
display(issues_df["severity"].value_counts().rename_axis("severity").reset_index(name="findings"))
display(issues_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Findings: Dates, Gaps and Summary, sections 06 to 08 (reviewed 2026-10-27)**
# 
# **Dates and values (06)**
# - All transaction dates fall between 2023-01-02 and 2026-10-21, except 2 goods receipt rows dated after the extract date (latest 2026-12-02).
# - Supplier-side documents are often keyed in more than 7 days after their date: landed cost lines 67.4%, goods receipts 29.4%, inventory movements 7.9%, purchase orders 1.2%. Sales documents are almost always entered on time (0.5%). This is flagged High by percentage, but it looks like a process habit (receipts backdated to arrival, freight invoices arriving late), not corrupt data. To confirm with FORTPRO: is the receipt DocDate the physical arrival date? It decides how we calculate supplier lead time.
# - Impossible values are rare: 3 negative-price sales lines (no item), 107 zero-quantity item lines, 1 negative supplier return quantity, 952 inventory movements with a negative price (likely reversals or adjustments).
# - Zero-price item lines (sales 0.5%, quotations 1.4%, inventory movements 2.8%) may be free or warranty items. Kept as real demand.
# 
# **Date gaps (07)**
# - No month is missing in any main table (45 months, Jan 2023 to Sep 2026).
# - Purchase orders arrive in waves (about 19 to 1,859 lines per month), so low months are normal. Customer return documents (ORDN) rise from about 7 to 33 a month through 2024 to about 45 to 140 from Dec 2024. To confirm with FORTPRO.
# 
# **Summary (08)**
# - 42 findings across 18 tables and 2.6M rows: 5 High, 9 Medium, 28 Low. None blocks the forecasting work.
# 
# **Expected, not defects (already explained by FORTPRO)**
# - LeadTime is filled for 28.7% of items and MinOrdrQty for 1.5%. FORTPRO gets lead times and minimums from suppliers by email or phone and does not store them in SAP (discovery call, data guide). Actual lead times will be calculated from PO and receipt dates.
# 
# **Next checks (our own analysis)**
# - Reconcile the stock snapshot (file 05) with the movements (file 06) when rebuilding stock history.
# - Convert purchase quantities to inventory units using NumInBuy (item master) before comparing POs with stock.
# - Review which SAP forecasts in file 16 are current once we look at their dates.

