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

# # NB_Verify_Supply_Side_Facts
# 
# **Project:** FORTPRO Capstone
# 
# **Author:** Dagiana Toussaint
# 
# **Created:** 2026-10-27
# 
# **Environment:** Development (FORTPRO-Capstone-Analytics)
# 
# **Purpose:** Verifies four supply-side facts in the real SAP data that purchasing and stock-risk forecasting depend on: how complete the SAP lead time is, which warehouses receive goods and move stock, whether open purchase orders look like real incoming stock, and the actual time from purchase order to receipt. Read-only; it does not modify any data.
# 
# **Source:** FORTPRO-Capstone-Landing / lh_fortpro_bronze / Files/raw, read through the bronze_raw shortcut in lh_fortpro_silver (files 03, 04, 05, 07, 08, 09)
# 
# **Destination:** None (results are shown in the notebook output only)
# 
# **Dependencies:** Attached lakehouse lh_fortpro_silver with the bronze_raw shortcut. Follows NB_DQ_Checks_Bronze_Raw. Jira: add ticket number.
# 
# **Change History**
# 
# | Date | Author | Description |
# |---|---|---|
# | 2026-10-27 | Dagiana Toussaint | Initial notebook creation |


# MARKDOWN ********************

# ## What this notebook checks
# The Sprint 1 review (October 2026) included statements about lead time, warehouses and open purchase orders that we had not yet tested against the full data, and questions from the professor and a classmate about the same topics. Each section below tests one of them, so later decisions rest on verified facts.
# 
# | Section | Statement or question from the review | Files used |
# |---|---|---|
# | 02 | "Only 29% of lead times are filled in" | 03 item master, 08 purchase orders |
# | 03 | "Two warehouses, and all imports come into Miami" and "Do they ship from Miami to Houston?" | 04 warehouses, 08, 09, 07 transfers |
# | 04 | "SAP treats open POs as incoming stock, which is false" | 05 stock, 08 purchase orders |
# | 05 | "Lead time is 3 to 6 months" and "we will compute it from PO date to receipt date" | 08, 09 goods receipts, 03 |

# MARKDOWN ********************

# ## 01. Configuration
# Loads the six files used below and casts the columns once. Cancelled purchase orders and receipts (CANCELED not N) are excluded in the sections that use them.

# CELL ********************

# =====================================================================
# 01. Configuration
# Purpose : Load files 03, 04, 05, 07, 08, 09 and cast key columns.
# Input   : bronze_raw shortcut (Files/bronze_raw)
# Output  : items, whs, stock, transfers, po, receipts, SNAPSHOT
# Notes   : Read-only. Keys are cast to long so joins compare equal types.
# =====================================================================
from notebookutils import mssparkutils
from pyspark.sql import functions as F

RAW_PATH = "Files/bronze_raw"

def load(prefix):
    name = [f.name for f in mssparkutils.fs.ls(RAW_PATH) if f.name.startswith(prefix)][0]
    return spark.read.parquet(f"{RAW_PATH}/{name}")

items = (load("03_item").withColumn("lead_time", F.col("LeadTime").cast("int")))
whs = load("04_warehouses")
stock = (load("05_stock")
         .withColumn("on_order", F.col("OnOrder").cast("double"))
         .withColumn("on_hand", F.col("OnHand").cast("double")))
transfers = (load("07_inventory_transfers")
             .withColumn("qty", F.col("Quantity").cast("double"))
             .filter(F.col("CANCELED") == "N"))
po = (load("08_purchase_orders")
      .withColumn("DocEntry", F.col("DocEntry").cast("long"))
      .withColumn("LineNum", F.col("LineNum").cast("long"))
      .withColumn("open_qty", F.col("OpenQty").cast("double"))
      .withColumn("po_date", F.col("DocDate").cast("date"))
      .filter((F.col("CANCELED") == "N") & (F.col("ItemCode").isNotNull()) & (F.col("ItemCode") != "")))
receipts = (load("09_goods_receipts")
            .withColumn("BaseType", F.col("BaseType").cast("int"))
            .withColumn("BaseEntry", F.col("BaseEntry").cast("long"))
            .withColumn("BaseLine", F.col("BaseLine").cast("long"))
            .withColumn("receipt_date", F.col("DocDate").cast("date"))
            .filter((F.col("DocType") == "OPDN") & (F.col("CANCELED") == "N")))

SNAPSHOT = stock.agg(F.max(F.col("SnapshotDate").cast("date"))).first()[0]
print(f"Items: {items.count():,} | PO lines: {po.count():,} | Receipt lines: {receipts.count():,}")
print(f"Stock snapshot date: {SNAPSHOT}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 02. How Complete Is the SAP Lead Time?
# Lead time is the number of days SAP expects between placing a purchase order and receiving the goods. It is stored on each item (file 03, column LeadTime). A value of 0 or empty means nobody filled it in. The presentation said only 29% are filled in. That is the share across the whole catalog, but most catalog items are never purchased, so the table also shows the share for items that actually appear on purchase orders.

# CELL ********************

# =====================================================================
# 02. SAP lead time completeness
# Purpose : Share of items with a lead time above 0, for three groups.
# Input   : items (file 03), po (file 08)
# Output  : table with group, items, with_lead_time, pct
# Notes   : "Purchased" = item appears on at least one non-cancelled PO line.
# =====================================================================
purchased = po.select("ItemCode").distinct().withColumn("purchased", F.lit(1))
base = items.join(purchased, "ItemCode", "left").withColumn("has_lt", (F.col("lead_time") > 0).cast("int"))

def summarize(label, df):
    r = df.agg(F.count("*").alias("n"), F.sum("has_lt").alias("k")).first()
    return (label, r["n"], r["k"], round(100 * r["k"] / r["n"], 1))

rows = [summarize("All items in the catalog", base),
        summarize("Items flagged as purchase items", base.filter(F.col("PrchseItem") == "Y")),
        summarize("Items that appear on a purchase order", base.filter(F.col("purchased") == 1))]
display(spark.createDataFrame(rows, ["group", "items", "with_lead_time", "pct_with_lead_time"]))

print("Distribution of lead time values above 0 (days):")
display(items.filter(F.col("lead_time") > 0).select("lead_time").summary("min", "25%", "50%", "75%", "max"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 03. Warehouses: Where Do Goods Arrive and Move?
# The presentation said there are two main warehouses (Miami and Houston) and that all imports come into Miami. The professor asked whether stock is shipped from Miami to Houston based on inventory levels. This section shows the warehouse list, where purchase order lines and goods receipts land, and the transfers between warehouses. Transfers where the from and to warehouse are the same are shown separately because they do not move stock between locations.

# CELL ********************

# =====================================================================
# 03. Warehouses
# Purpose : Warehouse list, where POs and receipts land, and transfers.
# Input   : whs (04), po (08), receipts (09), transfers (07)
# Output  : four tables
# Notes   : A1 = Master Warehouse (FL), H1 = Houston Main Warehouse (TX).
#           To-A1 / To-H1 = in-transit locations.
# =====================================================================
print("Warehouse list:")
display(whs.select("WhsCode", "WhsName", "State", "Country", "Inactive"))

def share_by_wh(df, label):
    total = df.count()
    return (df.groupBy("WhsCode").count()
              .withColumn("pct", F.round(100 * F.col("count") / total, 1))
              .join(whs.select("WhsCode", "WhsName"), "WhsCode", "left")
              .orderBy(F.desc("count"))
              .withColumnRenamed("count", label))

print("Purchase order lines by warehouse:")
display(share_by_wh(po, "po_lines"))
print("Goods receipt lines (OPDN) by warehouse:")
display(share_by_wh(receipts, "receipt_lines"))

print("Transfers between different warehouses:")
display(transfers.filter(F.col("FromWhsCod") != F.col("ToWhs"))
        .groupBy("FromWhsCod", "ToWhs")
        .agg(F.count("*").alias("lines"), F.round(F.sum("qty"), 0).alias("units"))
        .orderBy(F.desc("lines")))

same = transfers.filter(F.col("FromWhsCod") == F.col("ToWhs")).count()
print(f"Transfer lines where from and to warehouse are the same: {same:,} of {transfers.count():,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 04. Do Open Purchase Orders Look Like Real Incoming Stock?
# In SAP, an open purchase order line (LineStatus O) is a request to a supplier that has not been fully received. SAP adds the open quantity to a stock field called OnOrder (file 05), separate from OnHand. The team said SAP treats this as incoming stock even when it is not realistic. The data cannot prove which POs will never arrive, but it can show how much stock is in OnOrder, how old the open lines are, and whether the due dates are usable. Check the due dates first: if the due date equals the order date, we cannot say a PO is late.

# CELL ********************

# =====================================================================
# 04. Open purchase orders vs OnOrder
# Purpose : Compare open PO quantities with OnOrder, age open lines, and
#           test whether DocDueDate carries real information.
# Input   : po (08), stock (05), SNAPSHOT
# Output  : summary table, age table
# Notes   : Age = days from PO date to the stock snapshot date.
# =====================================================================
open_po = po.filter(F.col("LineStatus") == "O")

summary = [("Open PO lines", float(open_po.count())),
           ("Open PO quantity (units)", float(open_po.agg(F.sum("open_qty")).first()[0])),
           ("OnOrder in stock file (units)", float(stock.agg(F.sum("on_order")).first()[0])),
           ("OnHand in stock file (units)", float(stock.agg(F.sum("on_hand")).first()[0]))]
display(spark.createDataFrame(summary, ["measure", "value"]))

by_item = (open_po.groupBy("ItemCode", "WhsCode").agg(F.sum("open_qty").alias("po_open")))
cmp = (by_item.join(stock.select("ItemCode", "WhsCode", "on_order"), ["ItemCode", "WhsCode"], "full")
       .fillna(0, ["po_open", "on_order"]))
n = cmp.count()
match = cmp.filter(F.abs(F.col("po_open") - F.col("on_order")) < 1e-6).count()
print(f"Item-warehouse pairs where OnOrder equals the open PO quantity: {match:,} of {n:,} ({100*match/n:.1f}%)")

same_due = open_po.filter(F.col("DocDueDate").cast("date") == F.col("po_date")).count()
print(f"Open PO lines whose due date equals the PO date: {same_due:,} of {open_po.count():,} "
      f"({100*same_due/open_po.count():.1f}%)")

aged = open_po.withColumn("age_days", F.datediff(F.lit(SNAPSHOT), F.col("po_date")))
buckets = (aged.withColumn("age_bucket",
            F.when(F.col("age_days") <= 30, "1) 0-30 days")
             .when(F.col("age_days") <= 90, "2) 31-90 days")
             .when(F.col("age_days") <= 180, "3) 91-180 days")
             .when(F.col("age_days") <= 365, "4) 181-365 days")
             .otherwise("5) over 1 year"))
           .groupBy("age_bucket")
           .agg(F.count("*").alias("open_lines"), F.round(F.sum("open_qty"), 0).alias("open_units"))
           .orderBy("age_bucket"))
display(buckets)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 05. Actual Lead Time: PO Date to Receipt Date
# The real lead time is the number of days between the purchase order date and the date the goods were received. A goods receipt line (OPDN) points back to its purchase order line through BaseType 22, BaseEntry (the PO number) and BaseLine. Receipts with another BaseType (18 = supplier invoice, -1 = no base document) cannot be traced to a PO and are left out of the timing. The table shows how many receipts can be traced, the timing by supplier country (imports are mostly from China and India), and how the actual time compares with the lead time stored in SAP.

# CELL ********************

# =====================================================================
# 05. Actual lead time
# Purpose : Days from PO date to receipt date, by supplier country, and
#           comparison with the SAP lead time per item.
# Input   : receipts (09), po (08), items (03)
# Output  : coverage table, country table, comparison summary
# Notes   : Median and 90th percentile are used because a few very late
#           receipts would distort the average.
# =====================================================================
print("Receipt lines by the document they point back to:")
display(receipts.groupBy("BaseType").count().orderBy(F.desc("count")))

po_keys = po.select(F.col("DocEntry").alias("BaseEntry"), F.col("LineNum").alias("BaseLine"),
                    "po_date", "SupplierCountry")
timed = (receipts.filter(F.col("BaseType") == 22)
         .join(po_keys, ["BaseEntry", "BaseLine"], "inner")
         .withColumn("days", F.datediff("receipt_date", "po_date")))

print(f"Receipt lines linked to a PO line: {timed.count():,}")
display(timed.groupBy("SupplierCountry")
        .agg(F.count("*").alias("receipt_lines"),
             F.percentile_approx("days", 0.5).alias("median_days"),
             F.percentile_approx("days", 0.9).alias("p90_days"),
             F.max("days").alias("max_days"))
        .orderBy(F.desc("receipt_lines")))

actual = (timed.groupBy("ItemCode").agg(F.percentile_approx("days", 0.5).alias("actual_days"))
          .join(items.select("ItemCode", "lead_time"), "ItemCode")
          .filter(F.col("lead_time") > 0)
          .withColumn("diff", F.col("actual_days") - F.col("lead_time")))
n = actual.count()
within = actual.filter(F.abs("diff") <= 30).count()
print(f"Items with both an actual lead time and a SAP lead time: {n:,}")
print(f"Items where actual is within 30 days of SAP: {within:,} ({100*within/n:.1f}%)")
display(actual.select("actual_days", "lead_time", "diff").summary("25%", "50%", "75%"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 06. Findings and What We Can Say
# Numbers below are taken from the output above (run on 2026-10-27).
# 
# **Lead time (section 02).** About 29% of the whole catalog has a lead time, which is the figure the team quoted in the Sprint 1 review. For items that actually appear on purchase orders it is about 83%. So the statement is true for the catalog but misleading for purchasing: most items never bought have no lead time, and most bought items do. Use both numbers.
# 
# **Warehouses (section 03).** The active warehouses are A1 (Master Warehouse, Florida), H1 (Houston Main) and A2 (Trading). About three quarters of goods receipts land in A1, about 17% in H1 and about 9% in A2. So "all imports come into Miami" is mostly true but not exactly: Houston receives some purchases directly. Transfers between different warehouses are small: about 500 lines from A1 into the In Transit to H1 location (Miami to Houston) and about 1,150 lines from H1 into the In Transit to A1 location (Houston to Miami). Stock moves in both directions, and more often from Houston to Miami, so "Miami ships to Houston" is only part of the picture. Most transfer lines have the same from and to warehouse; ask FORTPRO what those represent.
# 
# **Open POs (section 04).** The open purchase order quantity is about 346,000 units, while OnOrder in the stock file is about 424,000 (about 22% higher). They match exactly for about 92% of item-warehouse pairs; we cannot explain the rest. The due date on 96% of open PO lines equals the PO date, so it is not a real promised date and cannot tell us whether a PO is late. Open lines are mostly recent: 68% are under 90 days old and 16% are older than 180 days (140 lines are over a year old). So the data does not support calling most open POs stale. Questions for FORTPRO: why is OnOrder higher than the open PO quantity, do buyers update the due date, and do they close POs that will never arrive?
# 
# **Actual lead time (section 05).** Only about 44% of goods receipt lines point back to a PO; the rest point to a supplier invoice or to nothing, so the timing uses that subset. Purchases from China show a median of about 108 days and a 90th percentile of about 153 days, which fits the "3 to 6 months" statement for imports. US suppliers have a median of about 13 days. Using one lead time for all items would be wrong; lead time has to be by supplier country or by supplier. SAP's lead time is usually longer than reality: only 53% of the 1,579 items with both values are within 30 days of the SAP value, and the middle item's actual time is 6 days shorter.
# 
# **Open questions from the Sprint 1 review.** Containers or pallets: not in the data (no shipment size). Miami to Houston: transfers exist but are small. Forecast baseline and horizon: not covered here; see the item classification and baseline notebooks.

