"""Replace customer and supplier codes with stable anonymous IDs across every file in a folder.

The same code always becomes the same ID (keyed by a secret phrase you choose), so files
still join. Keep the phrase private; without it the IDs cannot be traced back.

Usage:
    python tools/anonymize.py --src C:/exports/raw --dst C:/exports/anon --secret "a phrase only FORTPRO knows"

Columns treated as codes: CardCode, VendorCode, customer_code, supplier_code, Account.
Columns removed if present: CardName, Phone1, Phone2, Cellular, E_Mail, LicTradNum, Address, Street.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
from pathlib import Path

import pandas as pd

CODE_COLS = {"cardcode", "vendorcode", "customer_code", "supplier_code", "account", "customer", "supplier", "preferred_supplier"}
DROP_COLS = {"cardname", "phone1", "phone2", "cellular", "e_mail", "email", "lictradnum", "address", "street", "cntctprsn"}


def anon(value: str, secret: bytes, prefix: str = "BP") -> str:
    if value is None or str(value).strip() in ("", "nan", "None"):
        return value
    h = hmac.new(secret, str(value).strip().encode("utf-8"), hashlib.sha256).hexdigest()[:10].upper()
    return f"{prefix}{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--secret", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    key = a.secret.encode("utf-8")
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in (".csv", ".txt", ".tsv", ".xlsx", ".xls"):
            continue
        if p.suffix.lower() in (".xlsx", ".xls"):
            sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        else:
            sheets = {None: pd.read_csv(p, dtype=str, keep_default_na=False, sep=None, engine="python")}
        out = {}
        for name, df in sheets.items():
            df = df[[c for c in df.columns if c.strip().lower() not in DROP_COLS]].copy()
            for c in df.columns:
                if c.strip().lower() in CODE_COLS:
                    # one prefix for all business partners: SAP CardCodes are unique across customers and
                    # suppliers, and the same code must map to the same ID in every file
                    df[c] = df[c].map(lambda v: anon(v, key, "BP"))
            out[name] = df
        target = dst / p.name
        if None in out:
            out[None].to_csv(target, index=False)
        else:
            with pd.ExcelWriter(target) as xw:
                for name, df in out.items():
                    df.to_excel(xw, sheet_name=name, index=False)
        print("wrote", target)


if __name__ == "__main__":
    main()
