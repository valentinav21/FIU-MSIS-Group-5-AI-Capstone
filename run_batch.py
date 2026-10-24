"""Run the planner without the app, for a scheduled weekly job.

    python run_batch.py --folder C:/FORTPRO/exports --out C:/FORTPRO/results

Writes FORTPRO_AI_Planner_<date>.xlsx and refreshes the results the app shows.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import io, pipeline, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None, help="data folder (default: the one in settings)")
    ap.add_argument("--out", default="output", help="folder for the Excel workbook")
    a = ap.parse_args()
    s = io.load_settings()
    res = pipeline.run(a.folder or s["data"]["folder"], s, progress=lambda f, m: print(f"{f:5.0%}  {m}", flush=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if res.as_of is not None:
        p = out / f"FORTPRO_AI_Planner_{res.as_of:%Y%m%d}.xlsx"
        p.write_bytes(report.workbook(res))
        print("Wrote", p)
    for e in res.errors:
        print("ERROR:", e)
    return 1 if res.errors else 0


if __name__ == "__main__":
    sys.exit(main())
