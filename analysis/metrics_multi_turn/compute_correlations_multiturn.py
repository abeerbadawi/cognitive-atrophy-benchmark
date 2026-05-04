"""
Multi-Turn User Input × LLM Response Correlation Analysis
==========================================================
Mirrors the single-turn correlations analysis but adds the per-turn dimension.

For each LLM model m and each (U_i, R_j) attribute pair, three scopes:

  1. POOLED (across all 720 turn-units): one Spearman ρ per (m, i, j).
     Equivalent to "treat every turn as an independent observation."
     This is the headline for §5 / §6 in the report.

  2. PER-TURN (at each turn t = 1..10): n_c = 72 obs per (m, t, i, j).
     Lets us watch how the U-R coupling evolves as the conversation
     progresses. Reported as small-multiples in §7.

  3. PER-DATASET (HOPE / CareBench, pooled across turns): per-(m, ds, i, j)
     ρ over the 360 (HOPE) or 360 (CareBench) turn-units.

BH-FDR correction applied within (model, scope).

Outputs (in ../data/):
  per_pair_pooled.csv          - 5 models × 5 U × 10 R = 250 rows
  per_pair_per_turn.csv        - 5 × 10 turns × 5 × 10 = 2500 rows
  per_pair_per_dataset.csv     - 5 × 2 × 5 × 10 = 500 rows
  significant_cells.csv        - all (model, scope) cells passing |ρ|>=0.20 AND q<0.05
  results_summary.json
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA_OUT = HERE.parent / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR = HERE.parent / "data"
HOPE_CSV = DATA_DIR / "hope_human_eval.csv"
CARE_CSV = DATA_DIR / "carebench_human_eval.csv"

LLM_NAMES = {1: "Qwen", 2: "Llama", 3: "GPT", 4: "Claude", 5: "Gemini"}

USER_ATTRS = [
    ("U1", "user_typicality"),
    ("U2", "user_evocative"),
    ("U3", "user_sensitivity"),
    ("U4", "user_request_info"),
    ("U5", "user_underlying"),
]

# 10 R-attributes (R1=SEN binary, R2-R10 ordinal)
RESP_ATTRS = [
    ("SEN",  "S_score"),    # binary
    ("AUR",  "AUR_score"),
    ("TD",   "TD_score"),
    ("FIX",  "FIX_score"),
    ("RT",   "RT_score"),
    ("TN",   "TN_score"),
    ("QOC",  "QOC_score"),
    ("LM",   "LM_score"),
    ("ME",   "ME_score"),
    ("EMP",  "EMP_score"),
]

# ---------- helpers ----------
def topic_num(filename):
    m = re.search(r"topic(\d+)", filename or "")
    return int(m.group(1)) if m else None

def to_float(x):
    s = (x or "").strip()
    if not s:
        return None
    if "|" in s:
        # multi-coded: take max
        try:
            return max(float(p) for p in s.split("|"))
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None

def load(path, dataset_label):
    with open(path) as f:
        first = f.readline().strip()
        if "," in first:
            f.seek(0)
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        topic = topic_num(r.get("filename", ""))
        turn = to_float(r.get("turn"))
        if topic is None or turn is None:
            continue
        # Spread r first, then override / add our derived fields (NOT vice versa)
        rec = dict(r)
        rec["dataset"] = dataset_label
        rec["topic"] = topic
        rec["turn"] = int(turn)
        rec["conv_id"] = f"{dataset_label}_topic{topic:02d}"
        rec["annotator"] = (r.get("reviewer") or "").strip().lower()
        out.append(rec)
    return out

# BH-FDR
def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values for a list of raw p-values."""
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        orig_i, p = indexed[rank]
        adj = min(p * n / (rank + 1), 1.0) if (rank + 1) > 0 else 1.0
        prev = min(prev, adj)
        q[orig_i] = prev
    return q

# ---------- 1. load ----------
hope_rows = load(HOPE_CSV, "HOPE")
care_rows = load(CARE_CSV, "CareBench")
all_rows = hope_rows + care_rows
print(f"HOPE rows: {len(hope_rows)}  CareBench rows: {len(care_rows)}  total: {len(all_rows)}")

# ---------- 2. build per-(LLM × turn × conv) records with U + R values ----------
records = []
for r in all_rows:
    u_vals = {short: to_float(r.get(col)) for short, col in USER_ATTRS}
    for slot, name in LLM_NAMES.items():
        r_vals = {short: to_float(r.get(f"Response {slot}_{col}")) for short, col in RESP_ATTRS}
        records.append({
            "dataset": r["dataset"], "topic": r["topic"], "turn": r["turn"], "conv_id": r["conv_id"],
            "annotator": r["annotator"], "llm_slot": slot, "llm_name": name,
            **{f"U_{k}": v for k, v in u_vals.items()},
            **{f"R_{k}": v for k, v in r_vals.items()},
        })
print(f"Records (LLM × turn × conv): {len(records)}  (expected {len(all_rows) * 5})")

# ---------- 3. compute correlations per scope ----------
def corr_one(xs, ys):
    """Filter (None, None) pairs, return (rho, p, n) or (None, None, n_kept)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 8:
        return None, None, len(pairs)
    xs2 = [p[0] for p in pairs]; ys2 = [p[1] for p in pairs]
    if len(set(xs2)) < 2 or len(set(ys2)) < 2:
        return None, None, len(pairs)
    rho, p = spearmanr(xs2, ys2)
    if hasattr(rho, "item"): rho = rho.item()
    if hasattr(p, "item"): p = p.item()
    return rho, p, len(pairs)

# 3a. POOLED across all turns
print("\n=== Pooled correlations (per LLM × U × R, across all 720 turn-units) ===")
pooled_rows = []
for slot, name in LLM_NAMES.items():
    sub = [r for r in records if r["llm_slot"] == slot]
    raw_p_list = []
    rho_list = []
    cells = []
    for u_short, _ in USER_ATTRS:
        for r_short, _ in RESP_ATTRS:
            xs = [r[f"U_{u_short}"] for r in sub]
            ys = [r[f"R_{r_short}"] for r in sub]
            rho, p, n = corr_one(xs, ys)
            cells.append({"u": u_short, "r": r_short, "rho": rho, "p": p, "n": n})
    # BH-FDR within model (50 cells)
    p_vec = [c["p"] if c["p"] is not None else 1.0 for c in cells]
    q_vec = bh_fdr(p_vec)
    for c, q in zip(cells, q_vec):
        c["q"] = q
        c["model"] = name
        c["scope"] = "pooled"
        c["sig"] = (c["rho"] is not None and abs(c["rho"]) >= 0.20 and q < 0.05)
        pooled_rows.append(c)

# Save
out = DATA_OUT / "per_pair_pooled.csv"
fields = ["model", "u", "r", "rho", "p", "q", "n", "sig", "scope"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for c in pooled_rows:
        w.writerow({k: c.get(k) for k in fields})
print(f"Wrote {out.relative_to(HERE.parent)}  ({len(pooled_rows)} rows)")

# 3b. PER-TURN
print("\n=== Per-turn correlations (per LLM × turn × U × R, n_c = 72 conv per cell) ===")
per_turn_rows = []
for slot, name in LLM_NAMES.items():
    for t in range(1, 11):
        sub = [r for r in records if r["llm_slot"] == slot and r["turn"] == t]
        cells = []
        for u_short, _ in USER_ATTRS:
            for r_short, _ in RESP_ATTRS:
                xs = [r[f"U_{u_short}"] for r in sub]
                ys = [r[f"R_{r_short}"] for r in sub]
                rho, p, n = corr_one(xs, ys)
                cells.append({"u": u_short, "r": r_short, "rho": rho, "p": p, "n": n,
                              "model": name, "turn": t, "scope": "per_turn"})
        p_vec = [c["p"] if c["p"] is not None else 1.0 for c in cells]
        q_vec = bh_fdr(p_vec)
        for c, q in zip(cells, q_vec):
            c["q"] = q
            c["sig"] = (c["rho"] is not None and abs(c["rho"]) >= 0.20 and q < 0.05)
            per_turn_rows.append(c)

out = DATA_OUT / "per_pair_per_turn.csv"
fields2 = ["model", "turn", "u", "r", "rho", "p", "q", "n", "sig", "scope"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields2)
    w.writeheader()
    for c in per_turn_rows:
        w.writerow({k: c.get(k) for k in fields2})
print(f"Wrote {out.relative_to(HERE.parent)}  ({len(per_turn_rows)} rows)")

# 3c. PER-DATASET
print("\n=== Per-dataset correlations (pooled across that dataset's 360 turn-units) ===")
per_ds_rows = []
for slot, name in LLM_NAMES.items():
    for ds in ["HOPE", "CareBench"]:
        sub = [r for r in records if r["llm_slot"] == slot and r["dataset"] == ds]
        cells = []
        for u_short, _ in USER_ATTRS:
            for r_short, _ in RESP_ATTRS:
                xs = [r[f"U_{u_short}"] for r in sub]
                ys = [r[f"R_{r_short}"] for r in sub]
                rho, p, n = corr_one(xs, ys)
                cells.append({"u": u_short, "r": r_short, "rho": rho, "p": p, "n": n,
                              "model": name, "dataset": ds, "scope": "per_dataset"})
        p_vec = [c["p"] if c["p"] is not None else 1.0 for c in cells]
        q_vec = bh_fdr(p_vec)
        for c, q in zip(cells, q_vec):
            c["q"] = q
            c["sig"] = (c["rho"] is not None and abs(c["rho"]) >= 0.20 and q < 0.05)
            per_ds_rows.append(c)

out = DATA_OUT / "per_pair_per_dataset.csv"
fields3 = ["model", "dataset", "u", "r", "rho", "p", "q", "n", "sig", "scope"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields3)
    w.writeheader()
    for c in per_ds_rows:
        w.writerow({k: c.get(k) for k in fields3})
print(f"Wrote {out.relative_to(HERE.parent)}  ({len(per_ds_rows)} rows)")

# ---------- 4. significant cells (pooled) ----------
sig_pooled = [c for c in pooled_rows if c["sig"]]
sig_pooled.sort(key=lambda c: (-abs(c["rho"]), c["model"]))
out = DATA_OUT / "significant_cells.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "u", "r", "rho", "p", "q", "n"])
    w.writeheader()
    for c in sig_pooled:
        w.writerow({k: c[k] for k in ["model", "u", "r", "rho", "p", "q", "n"]})
print(f"Wrote {out.relative_to(HERE.parent)}  ({len(sig_pooled)} significant cells)")

# ---------- 5. summary ----------
def fmt(x): return None if x is None else round(x, 4)
summary = {
    "n_records": len(records),
    "n_turn_units": len(all_rows),
    "n_conversations": 72,
    "n_models": 5,
    "model_names": list(LLM_NAMES.values()),
    "scopes": {
        "pooled": {"n_cells": len(pooled_rows), "n_significant": len(sig_pooled)},
        "per_turn": {"n_cells": len(per_turn_rows),
                     "n_significant": sum(1 for c in per_turn_rows if c["sig"])},
        "per_dataset": {"n_cells": len(per_ds_rows),
                        "n_significant": sum(1 for c in per_ds_rows if c["sig"])},
    },
    "per_model_pooled_significant_count": {
        name: sum(1 for c in pooled_rows if c["model"] == name and c["sig"])
        for name in LLM_NAMES.values()
    },
}

# Cross-model consistency: count of models where each (u,r) cell is significant in pooled
consistency = {}
for u_short, _ in USER_ATTRS:
    for r_short, _ in RESP_ATTRS:
        models_sig = sorted([c["model"] for c in pooled_rows
                             if c["u"] == u_short and c["r"] == r_short and c["sig"]])
        rhos = [c["rho"] for c in pooled_rows if c["u"] == u_short and c["r"] == r_short and c["rho"] is not None]
        consistency[f"{u_short}_{r_short}"] = {
            "n_models_sig": len(models_sig),
            "models_sig": models_sig,
            "mean_rho_across_models": fmt(sum(rhos) / len(rhos)) if rhos else None,
        }
summary["cross_model_consistency_pooled"] = consistency

with open(DATA_OUT / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote results_summary.json")

# ---------- print headlines ----------
print("\n=========== HEADLINE ===========")
for name in LLM_NAMES.values():
    print(f"{name:<8}: {summary['per_model_pooled_significant_count'][name]} significant cells (pooled)")
print(f"\nTop 5 cells with most cross-model consistency:")
sorted_cons = sorted(consistency.items(), key=lambda x: -x[1]["n_models_sig"])
for k, v in sorted_cons[:8]:
    print(f"  {k}: {v['n_models_sig']}/5 models  mean ρ = {v['mean_rho_across_models']}")
