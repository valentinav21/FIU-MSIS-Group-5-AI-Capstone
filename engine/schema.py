"""Data contract for the FORTPRO AI Planner.

Every table the engine reads is described here once: its file name, its columns,
which columns are required, the standard SAP Business One field each column comes
from, and the alternate column names the loader accepts automatically.

The loader matches column headers case-insensitively and ignores spaces, dots,
underscores and dashes, so "DocDate", "doc_date", "Doc Date" and "OINV.DocDate"
all map to the same field.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str  # "str", "float", "date", "bool", "int"
    required: bool = False
    sap: str = ""
    desc: str = ""
    synonyms: tuple = ()


@dataclass(frozen=True)
class Table:
    key: str
    title: str
    sap_source: str
    problem: str  # BP1, BP2 or both
    essential: bool
    desc: str
    cols: tuple = field(default_factory=tuple)

    def col(self, name: str) -> Col:
        for c in self.cols:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def required(self):
        return [c.name for c in self.cols if c.required]


TABLES: dict[str, Table] = {}


def _t(table: Table):
    TABLES[table.key] = table
    return table


_t(Table(
    key="items", title="Product master", sap_source="OITM, OITB, OMRC", problem="BP1, BP2", essential=True,
    desc="One row per item code.",
    cols=(
        Col("item_code", "str", True, "OITM.ItemCode", "SAP item code (SKU)", ("itemcode", "sku", "item", "itemno", "partnumber")),
        Col("item_name", "str", False, "OITM.ItemName", "Description", ("itemname", "description", "productname", "dscription", "name")),
        Col("category", "str", False, "OITB.ItmsGrpNam", "Product family or item group", ("itmsgrpnam", "itemgroup", "itemgroupname", "topcategory", "productline", "family")),
        Col("subcategory", "str", False, "U_ field or web category", "Sub family", ("primarycategory", "subfamily", "parttype")),
        Col("brand", "str", False, "OMRC.FirmName", "Brand or manufacturer", ("firmname", "manufacturer", "brandname")),
        Col("unit_cost", "float", False, "OITM.AvgPrice", "Average unit cost (USD)", ("avgprice", "cost", "avgcost", "unitcost", "lastpurprc")),
        Col("list_price", "float", False, "ITM1.Price (base list)", "List price (USD)", ("price", "listprice", "baseprice")),
        Col("active", "bool", False, "OITM.validFor / frozenFor", "Y if item is active", ("validfor", "isactive", "status")),
        Col("moq", "float", False, "OITM.MinOrdrQty", "Minimum order quantity", ("minordrqty", "minorderqty", "minimumorderquantity")),
        Col("order_multiple", "float", False, "OITM.OrdrMulti", "Order multiple / pack size", ("ordrmulti", "ordermultiple", "packsize", "casepack")),
        Col("lead_time_days", "float", False, "OITM.LeadTime", "Planning lead time setting (days)", ("leadtime", "leadtimedays")),
        Col("preferred_supplier", "str", False, "OITM.CardCode", "Preferred vendor code", ("cardcode", "preferredvendor", "defaultvendor")),
        Col("export_only", "bool", False, "U_ field or EX suffix", "Y if export-only version", ("exportonly", "exportflag")),
    ),
))

_t(Table(
    key="sales", title="Sales transaction lines", sap_source="OINV/INV1 (or ORDR/RDR1, ODLN/DLN1)", problem="BP1, BP2", essential=True,
    desc="One row per invoice (or order) line. This is the demand history and the customer purchase history.",
    cols=(
        Col("doc_date", "date", True, "OINV.DocDate", "Document date used as the demand date", ("docdate", "invoicedate", "date", "postingdate", "orderdate")),
        Col("item_code", "str", True, "INV1.ItemCode", "Item code", ("itemcode", "sku", "item")),
        Col("quantity", "float", True, "INV1.Quantity", "Units sold", ("quantity", "qty", "units")),
        Col("doc_entry", "str", False, "OINV.DocEntry", "Document key", ("docentry", "docnum", "invoicenumber", "documentnumber", "orderid")),
        Col("line_num", "int", False, "INV1.LineNum", "Line number", ("linenum", "line")),
        Col("customer_code", "str", False, "OINV.CardCode", "Customer code (anonymized is fine)", ("cardcode", "customer", "customerid", "customercode", "bpcode")),
        Col("warehouse", "str", False, "INV1.WhsCode", "Warehouse code", ("whscode", "warehousecode", "whs", "location")),
        Col("unit_price", "float", False, "INV1.Price", "Net unit price after discount (USD)", ("price", "unitprice", "netprice")),
        Col("discount_pct", "float", False, "INV1.DiscPrcnt", "Line discount percent", ("discprcnt", "discount", "discountpercent")),
        Col("line_total", "float", False, "INV1.LineTotal", "Line net amount (USD)", ("linetotal", "netamount", "amount", "revenue", "sales")),
        Col("gross_profit", "float", False, "INV1.GrssProfit", "Line gross profit (USD)", ("grssprofit", "grossprofit", "margin")),
        Col("channel", "str", False, "U_ field, series or OSLP", "Sales channel", ("saleschannel", "source", "ordersource", "series")),
        Col("ship_state", "str", False, "CRD1.State / OINV address", "Ship-to state or province", ("state", "shiptostate", "region", "province")),
        Col("ship_country", "str", False, "CRD1.Country", "Ship-to country (ISO 2 letter)", ("country", "shiptocountry", "countrycode")),
        Col("sales_employee", "str", False, "OINV.SlpCode / OSLP.SlpName", "Sales employee", ("slpcode", "slpname", "salesperson", "salesrep", "rep")),
        Col("canceled", "bool", False, "OINV.CANCELED", "Y if document canceled", ("canceled", "cancelled", "iscanceled")),
    ),
))

_t(Table(
    key="returns", title="Returns and credit memos", sap_source="ORIN/RIN1, ORDN/RDN1", problem="BP1, BP2", essential=False,
    desc="Customer returns. Netted out of demand when settings say so.",
    cols=(
        Col("doc_date", "date", True, "ORIN.DocDate", "Return date", ("docdate", "date", "returndate")),
        Col("item_code", "str", True, "RIN1.ItemCode", "Item code", ("itemcode", "sku")),
        Col("quantity", "float", True, "RIN1.Quantity", "Units returned (positive)", ("quantity", "qty")),
        Col("customer_code", "str", False, "ORIN.CardCode", "Customer code", ("cardcode", "customer")),
        Col("warehouse", "str", False, "RIN1.WhsCode", "Warehouse code", ("whscode", "warehousecode")),
        Col("line_total", "float", False, "RIN1.LineTotal", "Credited amount (USD)", ("linetotal", "amount")),
        Col("reason", "str", False, "U_ field / Comments", "Return reason", ("reason", "returnreason", "comments")),
    ),
))

_t(Table(
    key="stock", title="Stock by warehouse", sap_source="OITW", problem="BP1", essential=True,
    desc="Current balance per item and warehouse, as of the export date.",
    cols=(
        Col("item_code", "str", True, "OITW.ItemCode", "Item code", ("itemcode", "sku")),
        Col("warehouse", "str", True, "OITW.WhsCode", "Warehouse code", ("whscode", "warehousecode", "whs")),
        Col("on_hand", "float", True, "OITW.OnHand", "On hand units", ("onhand", "instock", "qtyonhand")),
        Col("committed", "float", False, "OITW.IsCommited", "Committed to open sales orders", ("iscommited", "iscommitted", "committed")),
        Col("on_order", "float", False, "OITW.OnOrder", "Ordered from suppliers, not received", ("onorder", "ordered")),
        Col("min_stock", "float", False, "OITW.MinStock", "Current minimum (SAP)", ("minstock", "min", "reorderpoint")),
        Col("max_stock", "float", False, "OITW.MaxStock", "Current maximum (SAP)", ("maxstock", "max")),
    ),
))

_t(Table(
    key="purchases", title="Purchase order lines", sap_source="OPOR/POR1", problem="BP1", essential=True,
    desc="All PO lines in the history window, open and closed.",
    cols=(
        Col("po_entry", "str", True, "OPOR.DocEntry", "PO key", ("docentry", "ponumber", "docnum", "po")),
        Col("po_date", "date", True, "OPOR.DocDate", "PO creation date", ("docdate", "orderdate", "podate", "createdate")),
        Col("item_code", "str", True, "POR1.ItemCode", "Item code", ("itemcode", "sku")),
        Col("supplier_code", "str", True, "OPOR.CardCode", "Supplier code", ("cardcode", "vendor", "vendorcode", "supplier")),
        Col("quantity", "float", True, "POR1.Quantity", "Ordered units", ("quantity", "qty", "orderedqty")),
        Col("line_num", "int", False, "POR1.LineNum", "Line number", ("linenum", "line")),
        Col("unit_price", "float", False, "POR1.Price", "Unit price (document currency)", ("price", "unitprice")),
        Col("currency", "str", False, "POR1.Currency", "Currency", ("currency", "doccur")),
        Col("warehouse", "str", False, "POR1.WhsCode", "Ship-to warehouse", ("whscode", "warehousecode")),
        Col("promised_date", "date", False, "POR1.ShipDate / OPOR.DocDueDate", "Promised delivery date", ("shipdate", "docduedate", "duedate", "promiseddate", "deliverydate")),
        Col("open_qty", "float", False, "POR1.OpenQty", "Units still open", ("openqty", "openquantity", "remaining")),
        Col("status", "str", False, "POR1.LineStatus", "O = open, C = closed", ("linestatus", "docstatus", "status")),
    ),
))

_t(Table(
    key="receipts", title="Goods receipt lines", sap_source="OPDN/PDN1", problem="BP1", essential=True,
    desc="Receipts against POs. BaseEntry and BaseLine link back to the PO line.",
    cols=(
        Col("receipt_date", "date", True, "OPDN.DocDate", "Receipt date", ("docdate", "receiptdate", "date", "grpodate")),
        Col("item_code", "str", True, "PDN1.ItemCode", "Item code", ("itemcode", "sku")),
        Col("quantity", "float", True, "PDN1.Quantity", "Units received", ("quantity", "qty")),
        Col("po_entry", "str", False, "PDN1.BaseEntry", "Linked PO key", ("baseentry", "ponumber", "baseref")),
        Col("po_line", "int", False, "PDN1.BaseLine", "Linked PO line", ("baseline", "poline")),
        Col("supplier_code", "str", False, "OPDN.CardCode", "Supplier code", ("cardcode", "vendor", "supplier")),
        Col("warehouse", "str", False, "PDN1.WhsCode", "Warehouse code", ("whscode", "warehousecode")),
    ),
))

_t(Table(
    key="suppliers", title="Supplier master", sap_source="OCRD (CardType S)", problem="BP1", essential=False,
    desc="One row per supplier.",
    cols=(
        Col("supplier_code", "str", True, "OCRD.CardCode", "Supplier code", ("cardcode", "vendor", "vendorcode")),
        Col("supplier_name", "str", False, "OCRD.CardName (masked is fine)", "Supplier name or alias", ("cardname", "name", "vendorname")),
        Col("country", "str", False, "CRD1.Country", "Supplier country", ("country", "countrycode")),
    ),
))

_t(Table(
    key="item_suppliers", title="Item sourcing options", sap_source="ITM2, OSCN, special prices", problem="BP1", essential=False,
    desc="Which suppliers can supply each item, at what price and minimums. Enables multi-supplier recommendations.",
    cols=(
        Col("item_code", "str", True, "ITM2.ItemCode", "Item code", ("itemcode", "sku")),
        Col("supplier_code", "str", True, "ITM2.VendorCode", "Supplier code", ("vendorcode", "cardcode", "vendor")),
        Col("unit_price", "float", False, "Last PO price or price list", "Unit cost from this supplier (USD)", ("price", "unitprice", "cost")),
        Col("moq", "float", False, "Supplier MOQ", "Minimum order quantity", ("minordrqty", "minorderqty")),
        Col("order_multiple", "float", False, "Supplier pack size", "Order multiple", ("ordrmulti", "packsize")),
        Col("lead_time_days", "float", False, "Quoted lead time", "Quoted lead time (days)", ("leadtime",)),
        Col("landed_cost_pct", "float", False, "OIPF/IPF1 history", "Freight, duty and broker as % of price", ("freightpct", "landedpct", "dutypct")),
    ),
))

_t(Table(
    key="open_orders", title="Open sales orders / backorders", sap_source="ORDR/RDR1 open lines", problem="BP1", essential=False,
    desc="Customer orders not yet shipped. Used in the 1 to 8 week risk projection.",
    cols=(
        Col("item_code", "str", True, "RDR1.ItemCode", "Item code", ("itemcode", "sku")),
        Col("open_qty", "float", True, "RDR1.OpenQty", "Units still to ship", ("openqty", "quantity", "qty")),
        Col("warehouse", "str", False, "RDR1.WhsCode", "Warehouse code", ("whscode",)),
        Col("due_date", "date", False, "RDR1.ShipDate", "Promised ship date", ("shipdate", "docduedate", "duedate")),
        Col("customer_code", "str", False, "ORDR.CardCode", "Customer code", ("cardcode", "customer")),
    ),
))

_t(Table(
    key="customers", title="Customer master", sap_source="OCRD (CardType C), OCRG, CRD1", problem="BP2", essential=False,
    desc="One row per customer. Anonymized codes are fine.",
    cols=(
        Col("customer_code", "str", True, "OCRD.CardCode", "Customer code", ("cardcode", "customer", "customerid")),
        Col("customer_group", "str", False, "OCRG.GroupName", "Customer group", ("groupname", "groupcode", "group", "segment")),
        Col("customer_type", "str", False, "U_ field / industry", "Type (fleet, repair shop, dealer, export)", ("industry", "type", "customertype")),
        Col("city", "str", False, "CRD1.City", "City", ("city",)),
        Col("state", "str", False, "CRD1.State", "State or province", ("state", "region", "province")),
        Col("country", "str", False, "CRD1.Country", "Country (ISO 2 letter)", ("country", "countrycode")),
        Col("zip", "str", False, "CRD1.ZipCode", "Postal code", ("zipcode", "postalcode", "zip")),
        Col("first_purchase_date", "date", False, "Derived", "First purchase date", ("createdate", "firstpurchase", "customersince")),
        Col("price_tier", "str", False, "OCRD.ListNum", "Price list or tier", ("listnum", "pricelist", "tier")),
    ),
))

_t(Table(
    key="web_searches", title="Web store searches (FocusPoint)", sap_source="FocusPoint search log (outside SAP)", problem="BP1, BP2", essential=False,
    desc="Searches on fortpro.com and the app. Searches with no result are demand SAP never records.",
    cols=(
        Col("search_date", "date", True, "FocusPoint log", "Search date", ("date", "searchdate", "timestamp")),
        Col("search_term", "str", True, "FocusPoint log", "What the buyer typed", ("term", "query", "searchterm", "keyword")),
        Col("results_count", "float", False, "FocusPoint log", "Number of results shown", ("results", "resultcount", "hits")),
        Col("item_code", "str", False, "FocusPoint log", "Item clicked, if any", ("itemcode", "sku", "clickeditem")),
        Col("customer_code", "str", False, "FocusPoint account", "Account, if logged in", ("cardcode", "customer", "account")),
    ),
))

_t(Table(
    key="market_potential", title="External market potential", sap_source="Group 5 public data (not from FORTPRO)", problem="BP2", essential=False,
    desc="A market size measure per region, e.g. registered heavy trucks or trucking establishments.",
    cols=(
        Col("country", "str", True, "Public data", "Country (ISO 2 letter)", ("countrycode",)),
        Col("region", "str", True, "Public data", "State or region code; use the country code for whole countries", ("state", "province")),
        Col("potential", "float", True, "Public data", "Market size measure (any unit, same unit for all rows)", ("value", "marketsize", "trucks", "establishments")),
        Col("source", "str", False, "Public data", "Where the number came from", ("note",)),
    ),
))

_t(Table(
    key="events", title="Unusual demand events", sap_source="Usually outside SAP", problem="BP1, BP2", essential=False,
    desc="One-off orders, promotions or projects. Lines inside an exclude event are removed from the forecast history.",
    cols=(
        Col("item_code", "str", False, "", "Item code; blank means all items", ("itemcode", "sku")),
        Col("customer_code", "str", False, "", "Customer code; blank means all customers", ("cardcode", "customer")),
        Col("start_date", "date", True, "", "First day of the event", ("start", "from")),
        Col("end_date", "date", True, "", "Last day of the event", ("end", "to")),
        Col("action", "str", False, "", "exclude (default) or keep", ("type",)),
        Col("note", "str", False, "", "What happened", ("description", "comment")),
    ),
))
