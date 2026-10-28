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

# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-10-23
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** High-level profiling of the raw SAP Business One extracts: file inventory, row counts, column types, date ranges, null percentages and a masking check. Read-only; it does not modify any data.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Files/raw, read through the bronze_raw shortcut in lh_fortpro_silver
# 
# **Destination:** None (results are shown in the notebook output only)
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver with the bronze_raw shortcut
# 
# **Change History**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-10-23 | Dagiana Toussaint | Initial notebook creation |
# | 2026-10-23 | Dagiana Toussaint | Date-range check now matches timestamp_ntz columns |


# MARKDOWN ********************

# ## 01. Configuration
# Imports and settings. Change paths here only, not in the code below. `bronze_raw` is a shortcut to the protected raw files in the Landing workspace.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Keep imports and environment settings in one place.
# Input   : None
# Output  : RAW_PATH, TOP_N_NULL_COLS
# Notes   : RAW_PATH points at the bronze_raw shortcut, so this
#           notebook reads Landing's files without copying or changing them.
# =====================================================================
from notebookutils import mssparkutils
from pyspark.sql import functions as F

RAW_PATH = "Files/bronze_raw"   # shortcut to Landing > lh_fortpro_bronze > Files/raw
TOP_N_NULL_COLS = 3             # how many "most empty" columns to show per file
EXPECTED_FILE_COUNT = 18        # per the data guide; update if the extract change

# Confirm the shortcut is reachable before running any checks.
# Fails loudly (instead of silently) if the path or permissions are wrong.
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
# Confirms the expected Parquet files exist and loads each one into a dictionary. Expected: 18 files. Spark reads lazily, so this step is fast.

# CELL ********************

# =====================================================================
# 02. Source Validation / File Inventory
# Purpose : Confirm the raw files exist and load them once so every
#           later check reuses the same DataFrames.
# Input   : RAW_PATH
# Output  : dfs (name -> DataFrame), size_mb (name -> file size in MB)
# Notes   : Non-Parquet files are ignored. Fails if the file count
#           differs from what the extract should contain.
# =====================================================================

raw_files = [f for f in mssparkutils.fs.ls(RAW_PATH) if f.name.endswith(".parquet")]

if len(raw_files) != EXPECTED_FILE_COUNT:
    raise RuntimeError(
        f"FAILED: found {len(raw_files)} Parquet files, expected {EXPECTED_FILE_COUNT}."
    )

dfs     = {f.name: spark.read.parquet(f"{RAW_PATH}/{f.name}") for f in raw_files}
size_mb = {f.name: round(f.size / 1e6, 2) for f in raw_files}

print(f"SUCCESS: loaded {len(dfs)} Parquet files from {RAW_PATH}.")
print(f"Total size: {sum(size_mb.values()):,.1f} MB")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Row Counts
# 
# Counts the rows in each of the 18 tables and keeps the result for later checks. 

# CELL ********************

# ======================================================================
# 03. Row Counts
# Purpose : Count rows per table once and keep the result for later checks.
# Input   : dfs (name -> DataFrame), size_mb (name -> file size in MB)
# Output  : row_counts (name -> rows), summary table
# ======================================================================
import pandas as pd

row_counts = {name: df.count() for name, df in dfs.items()}

summary = (
    pd.DataFrame({
        "table": list(row_counts.keys()),
        "rows": list(row_counts.values()),
        "size_mb": [size_mb[n] for n in row_counts],
        "columns": [len(dfs[n].columns) for n in row_counts],
    })
    .sort_values("rows", ascending=False)
    .reset_index(drop=True)
)

empty = summary[summary["rows"] == 0]["table"].tolist()

display(summary)
print(f"Tables: {len(summary)}")
print(f"Total rows: {summary['rows'].sum():,}")
print(f"Empty tables: {empty if empty else 'none'}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Null Profile
# 
# Measures how much of each column is empty, per table. Flags columns that are mostly empty so we know which fields we can trust before modeling.

# CELL ********************

# ======================================================================
# 04. Null Profile
# Purpose : Measure null share for every column of every table.
# Input   : dfs (name -> DataFrame), row_counts (name -> rows)
# Output  : null_profile (one row per table and column)
# Notes   : Empty strings are counted as null. One Spark job per table.
# ======================================================================
from pyspark.sql import functions as F
import pandas as pd

rows = []
for name, df in dfs.items():
    exprs = []
    for c in df.columns:
        col = F.col(f"`{c}`")
        is_empty = col.isNull() | (F.trim(col.cast("string")) == "")
        exprs.append(F.sum(is_empty.cast("int")).alias(c))
    nulls = df.agg(*exprs).collect()[0].asDict()
    total = row_counts[name]
    for c, n in nulls.items():
        n = int(n or 0)
        rows.append({"table": name, "column": c, "rows": total,
                     "nulls": n, "null_pct": round(100 * n / total, 1) if total else None})

null_profile = pd.DataFrame(rows)

mostly_empty = (null_profile[null_profile["null_pct"] >= 50]
                .sort_values("null_pct", ascending=False))

print(f"Columns profiled: {len(null_profile)}")
print(f"Columns at least 50% empty: {len(mostly_empty)}")
display(mostly_empty.head(30))

# Lead time check (slide 5 says about 29% populated)
lt = null_profile[null_profile["column"].str.contains("lead", case=False)]
display(lt)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 05. Duplicates and Lead Time Values
# 
# Counts fully duplicated rows in each table, checks that the item master has one row per item, and looks at the values inside LeadTime (zeros, range), since filled does not always mean usable.

# CELL ********************

# ======================================================================
# 05. Duplicates and Lead Time Values
# Purpose : Find duplicate rows, check item-master key, profile LeadTime.
# Input   : dfs, row_counts
# Output  : dup_summary, lead-time profile
# Notes   : Full-row duplicates only. Key check runs if ItemCode exists.
# ======================================================================
from pyspark.sql import functions as F
import pandas as pd

# 1) Full-row duplicates per table
dup_rows = []
for name, df in dfs.items():
    distinct_rows = df.distinct().count()
    dup_rows.append({"table": name, "rows": row_counts[name],
                     "distinct_rows": distinct_rows,
                     "duplicate_rows": row_counts[name] - distinct_rows})
dup_summary = pd.DataFrame(dup_rows).sort_values("duplicate_rows", ascending=False)
display(dup_summary)
print(f"Tables with duplicate rows: {(dup_summary['duplicate_rows'] > 0).sum()}")

# 2) Item master key check
item_name = [n for n in dfs if n.startswith("03_item")][0]
item = dfs[item_name]
key_col = next((c for c in item.columns if c.lower() == "itemcode"), None)
if key_col:
    n_items = item.select(key_col).distinct().count()
    print(f"Item master: {row_counts[item_name]:,} rows, {n_items:,} distinct {key_col}")
else:
    print("ItemCode column not found. Columns:", item.columns[:15])

# 3) LeadTime values
lt = item.select(F.col("LeadTime").cast("double").alias("lt"))
stats = lt.agg(
    F.count("lt").alias("filled"),
    F.sum((F.col("lt") == 0).cast("int")).alias("zeros"),
    F.min("lt").alias("min"),
    F.expr("percentile_approx(lt, 0.5)").alias("median"),
    F.max("lt").alias("max"),
).collect()[0].asDict()
print("LeadTime:", stats)
usable = (stats["filled"] or 0) - (stats["zeros"] or 0)
print(f"LeadTime filled and above zero: {usable:,} of {row_counts[item_name]:,} "
      f"({100 * usable / row_counts[item_name]:.1f}%)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 06. Date Granularity
# For every date column in every table, shows the range, how many distinct days and months it has, the median gap between distinct dates, and whether rows carry a time of day. Classifies each column as snapshot, daily, monthly, or irregular. Read-only.

# CELL ********************

# ======================================================================
# 06. Date Granularity
# Purpose : Tell whether each date column is a snapshot, daily or monthly.
# Input   : dfs (name -> DataFrame), row_counts (name -> rows)
# Output  : granularity (one row per table and date column)
# Notes   : Matches date, timestamp and timestamp_ntz (Fabric reads these
#           columns as timestamp_ntz). Collects only the distinct dates
#           per column, so the driver never pulls full tables.
#           Transaction tables (sales, POs, receipts) will show as daily
#           because documents are created on any day. That is event time,
#           not a daily time series.
# ======================================================================
from pyspark.sql import functions as F
import pandas as pd

DATE_TYPES = ("date", "timestamp", "timestamp_ntz")

rows = []
for name, df in dfs.items():
    date_cols = [c for c, t in df.dtypes if t in DATE_TYPES]
    for c in date_cols:
        ts = F.col(f"`{c}`").cast("timestamp")
        base = df.select(ts.alias("ts")).where(F.col("ts").isNotNull())

        # share of rows that carry a time of day (not just a date)
        pct_time = base.agg(
            F.avg((F.col("ts") != F.to_date("ts").cast("timestamp")).cast("int"))
        ).collect()[0][0] or 0

        # distinct calendar dates only (small), then gaps in pandas
        days = (base.select(F.to_date("ts").alias("d")).distinct()
                    .toPandas()["d"].pipe(pd.to_datetime).sort_values())
        if len(days) == 0:
            continue
        gaps = days.diff().dt.days.dropna()
        med_gap = float(gaps.median()) if len(gaps) else None
        n_days = len(days)
        n_months = days.dt.to_period("M").nunique()
        per_month = n_days / n_months

        if n_days == 1:
            kind = "snapshot (single date)"
        elif per_month <= 1.5:
            kind = "monthly"
        elif med_gap is not None and med_gap <= 3 and per_month >= 5:
            kind = "daily"
        else:
            kind = "irregular"

        rows.append({
            "table": name, "column": c,
            "min": days.min().date(), "max": days.max().date(),
            "distinct_days": n_days, "distinct_months": n_months,
            "days_per_month": round(per_month, 1),
            "median_gap_days": med_gap,
            "pct_rows_with_time": round(100 * pct_time, 1),
            "granularity": kind,
        })

granularity = pd.DataFrame(rows).sort_values(["table", "column"]).reset_index(drop=True)
display(granularity)
print(f"Date columns profiled: {len(granularity)}")
print(granularity["granularity"].value_counts().to_string())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ======================================================================
# 06b. Non-daily date columns
# Purpose : Show which columns are monthly, snapshot or irregular.
# Input   : granularity (from section 06)
# Output  : the non-daily rows with their ranges and gaps
# ======================================================================
display(granularity[granularity["granularity"] != "daily"]
        [["table", "column", "min", "max", "distinct_days",
          "distinct_months", "days_per_month", "median_gap_days", "granularity"]])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
