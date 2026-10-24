# Consistency check of the discovery documents

Checked against each other and against the two spreadsheets (October 19, 2026). Counts were recomputed from
`FORTPRO_Web_Product_Catalog_Index.xlsx` (3,169 products) and `FORTPRO_Public_Web_Product_Data_Audit.xlsx` (1,195 pages).

## Corrections needed

| # | Document | Says | What the data shows | Suggested fix |
|---|---|---|---|---|
| 1 | Website Audit Summary, Business and Product Mix #2 | "Seven brands supply about 280 products, led by Spicer (43) and Lucas Oil (34)." | Six third-party brands have products: Spicer 43, Lucas Oil 34, Road Choice 22, Mahle 15, Bendix 11, SAF-HOLLAND 6 = 131 tagged. Donaldson and NTN pages are empty. The 280 non F-number products are those 131, plus 145 untagged (82 of them RL- floor mats), plus 4 FORTPRO-tagged items in other formats. | "Six third-party brands tag 131 products, led by Spicer (43) and Lucas Oil (34). About 145 more non-FORTPRO SKUs carry no brand, mostly RL- floor mats." |
| 2 | Website Audit, SKU pattern table; Summary #1 | "F + 6 digits: 2,889" | 2,889 SKUs start with F plus 6 digits, but only 2,479 are exactly that. 410 carry a suffix (E, EX, -A, -010, -STD and so on), such as F224748E and F224748EX. | Say "F + 6 digits, sometimes with a variant suffix", and ask Jose what each suffix means (the Data Requirements doc already asks about E and EX). |
| 3 | Data Requirements, Other items #5 | "About 12% of the web catalog (mainly lighting, air brake hoses, and wheels) falls under federal DOT standards" | Lighting alone is 376 of 3,169 = 11.9%. Adding air brake hoses (about 16 by name) and wheels (about 29) gives roughly 13%. | "About 13% (lighting alone is 12%)", or "at least 12%". |
| 4 | Website Audit Summary, Related Files; Data Audit workbook, Method and Limits | "FORTPRO Product Catalog Index.xlsx" and "FORTPRO Product Data Audit.xlsx" | The files are named "FORTPRO Web Product Catalog Index.xlsx" and "FORTPRO Public Web Product Data Audit.xlsx". | Use the exact file names so readers can find them. |
| 5 | Website Audit and Summary vs Data Requirements | PS1 / PS2 | The Data Requirements doc calls them BP1 / BP2. | Pick one label across all documents. BP matches the request FORTPRO will read. |

## Checked and consistent

* Problem statements: the Data Requirements doc reproduces the proposal text word for word.
* 3,169 products; 3,062 in a category plus 107 on brand pages only.
* 21,173 sitemap URLs = 17,787 tag pages + 3,386 other pages.
* 185 category pages = 163 with products + 22 empty; 53 categories with 1 to 10 products.
* Five lines (brakes, engine, suspension, lighting, driveline) = 2,241 products = 70.7%, rounded to 71%.
* 569 products with no brand page; FORTPRO tagged 2,469, of which 2,465 use F-numbers, plus 424 untagged F-numbers.
* Audit percentages tie to 1,195 pages: no description 100%, no specs 432 (36%), no cross-references 501 (42%),
  no brand 307 (26%), no category 127 (11%), "Customers also bought" 731 (61%), all pages reporting "in stock".
* SAP Business One table names in the data request (ORDR/RDR1, ODLN/DLN1, OINV/INV1, ORDN/RDN1, ORIN/RIN1, OITM, OITB,
  OMRC, OITW, OINM, OIVL/IVL1, OWTR/WTR1, OPOR/POR1, OPDN/PDN1, OCRD, ITM2, OSCN, OCRG, CRD1, OSLP, OTER, OPLN/ITM1,
  OSPP/SPP1, OFCT/FCT1, ORPD/RPD1, OIPF/IPF1, OQUT/QUT1, OOPR/OPR1, OCPN/CPN1, OCLG) are the standard names.

## Worth noting

* The audit sampled pages in sitemap order, not at random (stated in the workbook). The 38% sample percentages may
  not hold for the whole catalog; say "in the pages checked" when quoting them.
* The Oct 17 class briefing said scope "spans two problem statements". The proposal confirms FORTPRO sent both, so
  the one-sentence project statement can now be written.
