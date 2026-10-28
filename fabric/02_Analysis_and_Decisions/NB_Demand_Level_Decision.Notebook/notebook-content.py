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

# # NB_Demand_Level_Decision
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-10-27
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Evidence for decision D1: which SAP document level (order, delivery or invoice) we use as the demand history for forecasting. Compares the three levels, measures how much customer demand is never delivered, and records the decision. Read-only; it does not modify any data.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Files/raw, read through the bronze_raw shortcut in lh_fortpro_silver (file 01, sales lines)
# 
# **Destination:** None (results are shown in the notebook output only)
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver with the bronze_raw shortcut. Follows NB_DQ_Checks_Bronze_Raw (empty item codes are service or charge lines; no missing months). Jira: add ticket number.
# 
# **Change History**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-10-27 | Dagiana Toussaint | Initial notebook creation |


# MARKDOWN ********************

# ## What the three levels mean
# One sale creates up to three SAP documents, so file 01 holds the same sale up to three times:
# 
# | DocType | Document | Moment | What its quantity tells us |
# |---|---|---|---|
# | ORDR | Sales order | The customer asks for the parts | What customers **requested** |
# | ODLN | Delivery | The parts leave the warehouse | What we **shipped** |
# | OINV | A/R invoice | We bill the customer | What we **billed** (normally follows the delivery) |
# 
# If a customer orders 10 and we only have 6, the order says 10 and the delivery says 6. A forecast built on deliveries would learn "6" and under-buy again; a forecast built on orders learns "10". Sections 03 to 06 measure how often this happens.

# MARKDOWN ********************

# ## 01. Configuration
# Loads the sales lines file and fixes the column types. Quantities and document keys arrive as decimals or text in Bronze, so they are cast once here.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Load file 01 (sales lines) and cast numeric columns once.
# Input   : bronze_raw shortcut, file starting with "01_sales"
# Output  : sales (DataFrame with typed columns)
# Notes   : Read-only. DocEntry/LineNum/BaseEntry/BaseLine are cast to
#           long so order-to-delivery joins compare equal types.
# =====================================================================
from notebookutils import mssparkutils
from pyspark.sql import functions as F
import pandas as pd

RAW_PATH = "Files/bronze_raw"
sales_name = [f.name for f in mssparkutils.fs.ls(RAW_PATH) if f.name.startswith("01_sales")][0]

sales = (spark.read.parquet(f"{RAW_PATH}/{sales_name}")
         .withColumn("qty", F.col("Quantity").cast("double"))
         .withColumn("open_qty", F.col("OpenQty").cast("double"))
         .withColumn("DocEntry", F.col("DocEntry").cast("long"))
         .withColumn("LineNum", F.col("LineNum").cast("long"))
         .withColumn("BaseType", F.col("BaseType").cast("int"))
         .withColumn("BaseEntry", F.col("BaseEntry").cast("long"))
         .withColumn("BaseLine", F.col("BaseLine").cast("long"))
         .withColumn("year", F.year(F.col("DocDate").cast("date"))))

print(f"Loaded {sales_name}: {sales.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 02. Analysis Base
# Applies the exclusion rules from the data guide and the data quality checks: canceled documents (CANCELED not N), lines with no item code (service or charge lines, quantity 0), and lines with quantity 0 or less. The table shows how many rows each rule removes, so the base is traceable.

# CELL ********************

# =====================================================================
# 02. Analysis Base
# Purpose : Keep only real, non-canceled item lines with a positive
#           quantity, and show what each rule removes.
# Input   : sales
# Output  : base (DataFrame), exclusion_table
# Notes   : Rules are applied in order, so each row is counted once.
#           Rules: (1) CANCELED <> 'N'; (2) empty ItemCode; (3) qty <= 0.
# =====================================================================
item_empty = F.col("ItemCode").isNull() | (F.trim(F.col("ItemCode").cast("string")) == "")

total = sales.count()
canceled = sales.where(F.col("CANCELED") != "N").count()
no_item = sales.where((F.col("CANCELED") == "N") & item_empty).count()
no_qty = sales.where((F.col("CANCELED") == "N") & ~item_empty & (F.col("qty") <= 0)).count()

base = sales.where((F.col("CANCELED") == "N") & ~item_empty & (F.col("qty") > 0))
kept = base.count()

exclusion_table = pd.DataFrame([
    ("All rows in file 01", total),
    ("Removed: canceled documents", canceled),
    ("Removed: no item code (service/charge lines)", no_item),
    ("Removed: quantity 0 or less on item lines", no_qty),
    ("Kept for analysis", kept),
], columns=["step", "rows"])
display(exclusion_table)
assert total - canceled - no_item - no_qty == kept, "Exclusion counts do not add up"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Units by Year and Document Level
# Adds up the quantity at each level per year. The delivered and invoiced percentages show how much of what customers requested was shipped and billed.

# CELL ********************

# =====================================================================
# 03. Units by Year and Document Level
# Purpose : Compare total units at order, delivery and invoice level.
# Input   : base
# Output  : by_year (units per level, delivered % and invoiced % of orders)
# Notes   : 2026 is partial (data to the extract date, 2026-10-21).
# =====================================================================
by_year = (base.groupBy("year")
               .pivot("DocType", ["ORDR", "ODLN", "OINV"])
               .agg(F.sum("qty"))
               .orderBy("year")
               .toPandas())
by_year["delivered_pct_of_ordered"] = (100 * by_year["ODLN"] / by_year["ORDR"]).round(1)
by_year["invoiced_pct_of_ordered"] = (100 * by_year["OINV"] / by_year["ORDR"]).round(1)
by_year["ordered_not_delivered"] = (by_year["ORDR"] - by_year["ODLN"]).round(0)
display(by_year)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Still Open, or Closed Short?
# If the gap between ordered and delivered were only orders waiting to ship, the missing units would still be open. This checks how much of the ordered quantity is still open (`OpenQty`) and how many order lines are closed.

# CELL ********************

# =====================================================================
# 04. Still Open, or Closed Short?
# Purpose : Tell backlog (still open) apart from lines closed short.
# Input   : base
# Output  : status_counts, open_share
# Notes   : LineStatus: O = open, C = closed.
# =====================================================================
orders_base = base.where(F.col("DocType") == "ORDR")

status_counts = (orders_base.groupBy("LineStatus").count().toPandas())
status_counts["pct"] = (100 * status_counts["count"] / status_counts["count"].sum()).round(2)
display(status_counts)

tot = orders_base.agg(F.sum("qty").alias("ordered"), F.sum("open_qty").alias("open")).first()
open_share = 100 * tot["open"] / tot["ordered"]
print(f"Units still open: {open_share:.2f}% of units ordered")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 05. What Happened to Each Order Line
# Follows each order line to the delivery lines created from it (delivery `BaseType` 17, `BaseEntry` = order DocEntry, `BaseLine` = order LineNum) and compares the quantities. This shows the real pattern behind the gap: fully delivered, partly delivered, or not delivered at all.

# CELL ********************

# =====================================================================
# 05. What Happened to Each Order Line
# Purpose : Compare each order line's quantity with what was delivered
#           against it.
# Input   : base
# Output  : linked (one row per order line), outcome, link_share
# Notes   : Join key per the data guide: delivery BaseType 17 ->
#           ORDR DocEntry / LineNum. Deliveries copied from a quote or
#           created without a base document cannot be linked to an order
#           and are reported as link_share.
# =====================================================================
orders = (base.where(F.col("DocType") == "ORDR")
              .select(F.col("DocEntry").alias("o_entry"), F.col("LineNum").alias("o_line"),
                      F.col("qty").alias("ordered"), "Channel", "CustCountry"))

deliveries = base.where(F.col("DocType") == "ODLN")
delivered = (deliveries.where(F.col("BaseType") == 17)
                       .groupBy(F.col("BaseEntry").alias("o_entry"), F.col("BaseLine").alias("o_line"))
                       .agg(F.sum("qty").alias("delivered")))

linked = (orders.join(delivered, ["o_entry", "o_line"], "left")
                .fillna({"delivered": 0})
                .withColumn("outcome",
                            F.when(F.col("delivered") == 0, "not delivered")
                             .when(F.col("delivered") < F.col("ordered"), "partly delivered")
                             .when(F.col("delivered") == F.col("ordered"), "fully delivered")
                             .otherwise("over-delivered")))

n_deliv = deliveries.count()
n_linked = deliveries.where(F.col("BaseType") == 17).count()
print(f"Delivery lines linked to an order: {n_linked:,} of {n_deliv:,} ({100*n_linked/n_deliv:.1f}%)")

outcome = (linked.groupBy("outcome")
                 .agg(F.count("*").alias("order_lines"),
                      F.sum("ordered").alias("units_ordered"),
                      F.sum("delivered").alias("units_delivered"))
                 .toPandas())
outcome["pct_of_lines"] = (100 * outcome["order_lines"] / outcome["order_lines"].sum()).round(2)
outcome["pct_of_units_ordered"] = (100 * outcome["units_ordered"] / outcome["units_ordered"].sum()).round(2)
display(outcome.sort_values("order_lines", ascending=False))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 06. Where Is the Shortfall? By Channel and Customer Country
# Shows delivered units as a percentage of ordered units for each sales channel and for US versus export customers. If one group is always fully delivered and another is not, the gap is concentrated rather than random.

# CELL ********************

# =====================================================================
# 06. Where Is the Shortfall? By Channel and Customer Country
# Purpose : Find which groups of order lines are not fully delivered.
# Input   : linked
# Output  : by_channel, by_region
# Notes   : Export = customer country other than US (CustCountry).
#           Aggregates only; no customer or item identifiers shown.
# =====================================================================
def summarize(df, col):
    return (df.groupBy(col)
              .agg(F.count("*").alias("order_lines"),
                   F.sum("ordered").alias("units_ordered"),
                   F.sum("delivered").alias("units_delivered"))
              .withColumn("delivered_pct", F.round(100 * F.col("units_delivered") / F.col("units_ordered"), 1))
              .orderBy(F.col("order_lines").desc())
              .toPandas())

by_channel = summarize(linked, "Channel")
by_region = summarize(linked.withColumn("region",
                          F.when(F.col("CustCountry") == "US", "US customers").otherwise("Export customers")),
                      "region")
display(by_channel)
display(by_region)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 07. Decision D1: Demand Level for Forecasting
# **Date:** 2026-10-27 | **Decided by:** Dagiana Toussaint (Data Engineer) | **Status:** Proposed to the team; confirm with Product Owner.
# 
# **Decision:** Use **sales orders (ORDR)** as the demand history for the forecast. Keep deliveries (ODLN) and invoices (OINV) as the "fulfilled" comparison.
# 
# **Why (data as of the 2026-10-21 extract):**
# - Customers ordered 5 to 11 percent more units each year than were delivered (delivered was 89 to 95 percent of ordered units).
# - The gap is not backlog: 99 percent of order lines are closed and under 1 percent of ordered units are still open.
# - About 93 percent of order lines are fully delivered. About 6 percent of order lines (about 7.5 percent of ordered units) have no delivery at all, and about 1 percent are partly delivered. The shortfall is mostly whole lines that were never shipped.
# - The shortfall is concentrated: marketplace orders under MAXITRUCKS.COM are delivered at about 100 percent, B2B orders at about 88 to 90 percent, and export customers at about 87 percent versus 93 percent for US customers.
# - Delivery and invoice quantities are almost identical (within about 0.4 percent), so choosing between those two does not matter. Order versus delivery is the real choice.
# 
# **Alternatives considered:**
# - Deliveries (ODLN): only counts what we could ship, so shortages hide themselves and the forecast under-buys again.
# - Invoices (OINV): same as deliveries, plus billing timing.
# 
# **Risks and caveats:**
# - A line closed without delivery is not proof of a stockout. The customer may also have canceled or changed the order. Question for FORTPRO: why are about 6 percent of order lines closed with no delivery?
# - Orders can include duplicates or lines the customer later dropped. Canceled documents are already excluded.
# - About 0.2 percent of delivery lines cannot be linked to an order (created from a quote or without a base document).
# 
# **Exclusions applied (section 02):** canceled documents, lines with no item code (service or charge lines), lines with quantity 0 or less.
# 
# **Revisit if:** FORTPRO says most unshipped lines are customer cancellations rather than missing stock. Then compare a forecast on orders against one on deliveries and keep the one that predicts better.

