/* =====================================================================================
   FORTPRO AI Planner: SAP Business One export queries (Microsoft SQL Server syntax)
   -------------------------------------------------------------------------------------
   For Daniel Chavez (IT Systems Manager). Run each query, save the result as CSV with the
   file name shown, and put the files in one folder. Column names are the standard SAP
   field names, which the app maps automatically. Nothing here writes to SAP.

   * Change the start date in each WHERE clause if you want more or less history.
   * Customer and supplier names are NOT exported. CardCode is enough. If CardCode itself
     contains names, run tools/anonymize.py on the folder afterwards.
   * SAP HANA: put double quotes around table and field names, e.g. T0."DocDate".
   * Custom U_ fields: add them to the SELECT lists. U_Channel, for example, is picked up
     as the sales channel automatically.
   * Please confirm with a test before relying on the price list number (PriceList = 1)
     and the address joins; these vary between SAP setups.
   ===================================================================================== */


/* 1. OITM.csv  - product master ------------------------------------------------------ */
SELECT T0.ItemCode, T0.ItemName, T1.ItmsGrpNam, T2.FirmName,
       T0.AvgPrice, P.Price AS ListPrice,
       T0.validFor, T0.MinOrdrQty, T0.OrdrMulti, T0.LeadTime, T0.CardCode,
       T0.InvntItem, T0.CreateDate
FROM OITM T0
LEFT JOIN OITB T1 ON T1.ItmsGrpCod = T0.ItmsGrpCod
LEFT JOIN OMRC T2 ON T2.FirmCode = T0.FirmCode
LEFT JOIN ITM1 P  ON P.ItemCode = T0.ItemCode AND P.PriceList = 1;


/* 2. INV1.csv  - A/R invoice lines (demand and customer purchase history) ------------- */
SELECT T0.DocEntry, T1.LineNum, T0.DocDate, T1.ItemCode, T1.Quantity, T1.Price,
       T1.DiscPrcnt, T1.LineTotal, T1.GrssProfit, T0.CardCode, T1.WhsCode,
       S.SlpName, T0.CANCELED,
       A.State, A.Country
       /* , T0.U_Channel   add channel or other U_ fields here if FORTPRO uses them */
FROM OINV T0
JOIN INV1 T1 ON T1.DocEntry = T0.DocEntry
LEFT JOIN OSLP S ON S.SlpCode = T0.SlpCode
LEFT JOIN CRD1 A ON A.CardCode = T0.CardCode AND A.Address = T0.ShipToCode AND A.AdresType = 'S'
WHERE T0.DocDate >= '20230101'
  AND T0.CANCELED = 'N'
  AND T1.ItemCode IS NOT NULL;


/* 3. RIN1.csv  - A/R credit memo lines (customer returns) -------------------------- */
SELECT T0.DocEntry, T1.LineNum, T0.DocDate, T1.ItemCode, T1.Quantity, T1.LineTotal,
       T0.CardCode, T1.WhsCode
FROM ORIN T0
JOIN RIN1 T1 ON T1.DocEntry = T0.DocEntry
WHERE T0.DocDate >= '20230101' AND T0.CANCELED = 'N' AND T1.ItemCode IS NOT NULL;


/* 4. OITW.csv  - stock by warehouse, today ------------------------------------------- */
SELECT ItemCode, WhsCode, OnHand, IsCommited, OnOrder, MinStock, MaxStock
FROM OITW
WHERE OnHand <> 0 OR IsCommited <> 0 OR OnOrder <> 0 OR MinStock <> 0 OR MaxStock <> 0;


/* 5. POR1.csv  - purchase order lines, open and closed -------------------------------- */
SELECT T0.DocEntry, T1.LineNum, T0.DocDate, T1.ItemCode, T0.CardCode, T1.Quantity,
       T1.Price, T1.Currency, T1.WhsCode, T1.ShipDate, T1.OpenQty, T1.LineStatus
FROM OPOR T0
JOIN POR1 T1 ON T1.DocEntry = T0.DocEntry
WHERE T0.DocDate >= '20230101' AND T0.CANCELED = 'N' AND T1.ItemCode IS NOT NULL;


/* 6. PDN1.csv  - goods receipt PO lines, linked to the PO line they received ---------- */
SELECT T0.DocEntry, T0.DocDate, T1.ItemCode, T1.Quantity,
       CASE WHEN T1.BaseType = 22 THEN T1.BaseEntry END AS BaseEntry,
       CASE WHEN T1.BaseType = 22 THEN T1.BaseLine  END AS BaseLine,
       T0.CardCode, T1.WhsCode
FROM OPDN T0
JOIN PDN1 T1 ON T1.DocEntry = T0.DocEntry
WHERE T0.DocDate >= '20230101' AND T0.CANCELED = 'N' AND T1.ItemCode IS NOT NULL;


/* 7. suppliers.csv - supplier master (no names needed) --------------------------------- */
SELECT CardCode, Country
FROM OCRD
WHERE CardType = 'S';


/* 8. ITM2.csv  - extra approved vendors per item (enables multi-supplier choice) ------- */
SELECT ItemCode, VendorCode
FROM ITM2;
/* If a supplier price list exists per vendor, add a Price column. Otherwise the app uses
   each supplier's last PO price for that item. */


/* 9. open_orders.csv - open sales order lines ----------------------------------------- */
SELECT T1.ItemCode, T1.OpenQty, T1.WhsCode, T1.ShipDate, T0.CardCode
FROM ORDR T0
JOIN RDR1 T1 ON T1.DocEntry = T0.DocEntry
WHERE T1.LineStatus = 'O' AND T0.CANCELED = 'N' AND T1.OpenQty > 0;


/* 10. customers.csv - customer master (no names, phones or emails) -------------------- */
SELECT T0.CardCode, G.GroupName, T0.City, T0.State1 AS State, T0.Country, T0.ZipCode,
       T0.CreateDate, T0.ListNum
FROM OCRD T0
LEFT JOIN OCRG G ON G.GroupCode = T0.GroupCode
WHERE T0.CardType = 'C';
