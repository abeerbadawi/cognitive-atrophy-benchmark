"""
Single-turn User Input Analysis — UIRI on CounselChat + PAIR

Computes the User Input Risk Index (UIRI) on the merged single-turn datasets.

Inputs (must sit alongside this script):
  counselchat_human_eval.csv
  pair_human_eval.csv

Outputs (printed to stdout, also saved to results.json):
  - per-attribute marginals per dataset and combined
  - UIRI summary statistics
  - per-reviewer breakdown using IRR-table codes (R1..R6 + Gold)

Methodology
-----------
Each user turn is coded once on five attributes (U1..U5).  Raw values are
normalised by their range maxima (k1..k5 = 2,2,1,1,2) so each attribute lands
in [0, 1] with 1 indicating the higher-demand end.  UIRI is the equal-weighted
mean of the five normalised values:

    UIRI(t) = (1/5) [ s_U1 + s_U2 + s_U3 + s_U4 + s_U5 ]

Bands: Low  if UIRI < 0.30
       Medium if 0.30 <= UIRI < 0.60
       High if UIRI >= 0.60
       (pre-specified, descriptive stratification — Boateng et al., 2018)

Reviewer codes follow the inter-rater reliability table:
  R1 = R1, R2 = R2, R3 = R3,
  R4 = R4, R5 = R5, R6 = R6,
  Gold = gold_standard  (R7 = a hypothetical 7th reviewer is excluded and is not in either single-turn block).
"""

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
DATASETS = {
    "CounselChat": DATA / "counselchat_human_eval.csv",
    "PAIR":        DATA / "pair_human_eval.csv",
}

NAME_TO_CODE = {
    "R1":     "R1",
    "R2":    "R2",
    "R3":  "R3",
    "R4": "R4",
    "R5":      "R5",
    "R6":     "R6",
    "gold_standard":    "Gold",
}


def normalise(row):
    U1 = int(float(row["user_typicality"]))
    U2 = int(float(row["user_evocative"]))
    U3 = int(float(row["user_sensitivity"]))
    U4 = int(float(row["user_request_info"]))
    U5 = int(float(row["user_underlying"]))
    s = (U1 / 2, U2 / 2, U3, U4, U5 / 2)
    UIRI = sum(s) / 5
    return U1, U2, U3, U4, U5, UIRI


def band(u):
    if u < 0.30:  return "Low"
    if u < 0.60:  return "Medium"
    return "High"


def load_dataset(name, path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        U1, U2, U3, U4, U5, UIRI = normalise(r)
        out.append({
            "dataset":  name,
            "reviewer": r["reviewer"],
            "code":     NAME_TO_CODE.get(r["reviewer"], r["reviewer"]),
            "prompt_id": int(float(r["prompt_id"])),
            "label":    r.get("label", ""),
            "U1": U1, "U2": U2, "U3": U3, "U4": U4, "U5": U5,
            "UIRI": UIRI,
            "band": band(UIRI),
        })
    return out


def per_attribute_marginals(rows):
    out = {}
    for attr in ["U1", "U2", "U3", "U4", "U5"]:
        c = Counter([r[attr] for r in rows])
        out[attr] = {str(k): v for k, v in sorted(c.items())}
    return out


def per_reviewer_summary(rows):
    by_code = defaultdict(list)
    for r in rows:
        by_code[r["code"]].append(r)

    order = ["Gold", "R1", "R2", "R3", "R4", "R5", "R6"]
    out = []
    for code in order:
        if code not in by_code:
            continue
        rs = by_code[code]
        names = sorted(set(r["reviewer"] for r in rs))
        prompts = sorted(set(r["prompt_id"] for r in rs))
        bands = Counter([r["band"] for r in rs])
        out.append({
            "code":    code,
            "name":    " / ".join(names),
            "n":       len(rs),
            "block":   f"{min(prompts)}–{max(prompts)}",
            "mean":    round(statistics.mean(r["UIRI"] for r in rs), 3),
            "low":     bands.get("Low", 0),
            "medium":  bands.get("Medium", 0),
            "high":    bands.get("High", 0),
        })
    return out


def panel_summary(rows):
    vals = [r["UIRI"] for r in rows]
    bands = Counter([r["band"] for r in rows])
    return {
        "n":      len(rows),
        "n_codings": len(rows) * 5,
        "mean":   round(statistics.mean(vals), 3),
        "sd":     round(statistics.stdev(vals), 3),
        "min":    round(min(vals), 3),
        "max":    round(max(vals), 3),
        "low":    bands.get("Low", 0),
        "medium": bands.get("Medium", 0),
        "high":   bands.get("High", 0),
    }


def main():
    all_rows = []
    per_dataset = {}
    for name, path in DATASETS.items():
        rows = load_dataset(name, path)
        per_dataset[name] = rows
        all_rows.extend(rows)

    results = {
        "per_dataset": {
            name: {
                "n_rows": len(rs),
                "marginals": per_attribute_marginals(rs),
                "panel": panel_summary(rs),
                "per_reviewer": per_reviewer_summary(rs),
            } for name, rs in per_dataset.items()
        },
        "combined": {
            "n_rows": len(all_rows),
            "marginals": per_attribute_marginals(all_rows),
            "panel": panel_summary(all_rows),
            "per_reviewer": per_reviewer_summary(all_rows),
        },
    }

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- pretty print ----
    print(f"=== Single-turn user input analysis ===\n")
    for name in DATASETS:
        d = results["per_dataset"][name]
        print(f"--- {name}  (n = {d['n_rows']} prompts, {d['n_rows']*5} codings) ---")
        for attr in ["U1", "U2", "U3", "U4", "U5"]:
            c = d["marginals"][attr]
            parts = ", ".join(f"{lvl}={cnt} ({100*cnt/d['n_rows']:.0f}%)" for lvl, cnt in sorted(c.items()))
            print(f"  {attr}: {parts}")
        p = d["panel"]
        print(f"  UIRI: mean={p['mean']:.3f}  sd={p['sd']:.3f}  range [{p['min']:.3f}, {p['max']:.3f}]")
        print(f"  Bands: Low={p['low']}, Medium={p['medium']}, High={p['high']}")
        print(f"  Per reviewer:")
        for row in d["per_reviewer"]:
            print(f"    {row['code']:<6} {row['name']:<12} n={row['n']:>2}  block {row['block']:<6}  mean={row['mean']:.3f}  L/M/H = {row['low']}/{row['medium']}/{row['high']}")
        print()

    c = results["combined"]
    print(f"--- COMBINED  (n = {c['n_rows']} prompts, {c['n_rows']*5} codings) ---")
    for attr in ["U1", "U2", "U3", "U4", "U5"]:
        cc = c["marginals"][attr]
        parts = ", ".join(f"{lvl}={cnt} ({100*cnt/c['n_rows']:.0f}%)" for lvl, cnt in sorted(cc.items()))
        print(f"  {attr}: {parts}")
    p = c["panel"]
    print(f"  UIRI: mean={p['mean']:.3f}  sd={p['sd']:.3f}  range [{p['min']:.3f}, {p['max']:.3f}]")
    print(f"  Bands: Low={p['low']}, Medium={p['medium']}, High={p['high']} ({100*p['low']/c['n_rows']:.0f}% / {100*p['medium']/c['n_rows']:.0f}% / {100*p['high']/c['n_rows']:.0f}%)")
    print(f"  Per reviewer:")
    for row in c["per_reviewer"]:
        print(f"    {row['code']:<6} {row['name']:<12} n={row['n']:>2}  block {row['block']:<6}  mean={row['mean']:.3f}  L/M/H = {row['low']}/{row['medium']}/{row['high']}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
