"""Write blank CSV templates and the data dictionary to data/templates."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.schema import TABLES  # noqa: E402


def main(out: Path | None = None):
    out = out or ROOT / "data" / "templates"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, t in TABLES.items():
        pd.DataFrame(columns=[c.name for c in t.cols]).to_csv(out / f"{key}.csv", index=False)
        for c in t.cols:
            rows.append({"File": f"{key}.csv", "Table": t.title, "Needed for": t.problem,
                         "Priority": "Essential" if t.essential else "Optional", "Column": c.name,
                         "Required": "Yes" if c.required else "", "Type": c.dtype, "SAP B1 source": c.sap,
                         "Description": c.desc, "Also accepted as": ", ".join(c.synonyms)})
    dd = pd.DataFrame(rows)
    dd.to_csv(out / "DATA_DICTIONARY.csv", index=False)
    return dd


if __name__ == "__main__":
    print(len(main()), "columns documented")
