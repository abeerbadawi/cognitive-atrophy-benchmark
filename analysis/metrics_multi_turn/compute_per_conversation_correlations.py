"""
Per-conversation correlations: for each (conversation, LLM, U_i, R_j) compute
Spearman ρ across that conversation's 10 turns.

This is a 4th scope on top of the three already in compute_correlations_multiturn.py
(pooled / per-turn / per-dataset). It answers the question "within this single
conversation, do U_i and R_j move together as the turns progress?"

n_obs per cell = T_c (10 for almost all conversations; 9 for HOPE topic 7).

Total cells: 72 conv × 5 LLMs × 5 U × 10 R = 18,000.

Outputs (in ../data/):
  per_conversation_correlations.csv     - all 18,000 cells
  per_conversation_summary.json         - per-conv counts + per-cell distributions
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA_OUT = HERE.parent / "data"
DATA_DIR = HERE.parent / "data"
HOPE_CSV = DATA_DIR / "hope_human_eval.csv"
CARE_CSV = DATA_DIR / "carebench_human_eval.csv"

LLM_NAMES = {1: "Qwen", 2: "Llama", 3: "GPT", 4: "Claude", 5: "Gemini"}
USER_ATTRS = [("U1", "user_typicality"), ("U2", "user_evocative"),
              ("U3", "user_sensitivity"), ("U4", "user_request_info"),
              ("U5", "user_underlying")]
RESP_ATTRS = [("SEN", "S_score"), ("AUR", "AUR_score"), ("TD", "TD_score"),
              ("FIX", "FIX_score"), ("RT", "RT_score"), ("TN", "TN_score"),
              ("QOC", "QOC_score"), ("LM", "LM_score"), ("ME", "ME_score"),
              ("EMP", "EMP_score")]

def topic_num(fn):
    m = re.search(r"topic(\d+)", fn or "")
    return int(m.group(1)) if m else None
def to_float(x):
    s = (x or "").strip()
    if not s: return None
    if "|" in s:
        try: return max(float(p) for p in s.split("|"))
        except: return None
    try: return float(s)
    except: return None

def load(path, ds):
    with open(path) as f:
        first = f.readline().strip()
        if "," in first: f.seek(0)
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        topic = topic_num(r.get("filename", ""))
        turn = to_float(r.get("turn"))
        if topic is None or turn is None: continue
        rec = dict(r)
        rec["dataset"] = ds; rec["topic"] = topic; rec["turn"] = int(turn)
        rec["conv_id"] = f"{ds}_topic{topic:02d}"
        out.append(rec)
    return out

# ---------- 1. load ----------
all_rows = load(HOPE_CSV, "HOPE") + load(CARE_CSV, "CareBench")
print(f"Loaded {len(all_rows)} turn-units")

# Group by conversation
by_conv = defaultdict(list)
for r in all_rows:
    by_conv[r["conv_id"]].append(r)
for c in by_conv:
    by_conv[c].sort(key=lambda r: r["turn"])
print(f"Conversations: {len(by_conv)}")

# ---------- 2. compute per-conversation ρ ----------
out_rows = []
for conv_id in sorted(by_conv):
    rs = by_conv[conv_id]
    T = len(rs)
    ds = rs[0]["dataset"]
    topic = rs[0]["topic"]
    annotator = (rs[0].get("reviewer") or "").strip().lower()

    # User attribute time series (one shared series per attribute, since U coded per turn)
    u_series = {short: [to_float(r.get(col)) for r in rs] for short, col in USER_ATTRS}

    for slot, name in LLM_NAMES.items():
        # Response attribute time series for this LLM in this conversation
        r_series = {short: [to_float(r.get(f"Response {slot}_{col}")) for r in rs]
                    for short, col in RESP_ATTRS}

        for u_short, _ in USER_ATTRS:
            xs_full = u_series[u_short]
            for r_short, _ in RESP_ATTRS:
                ys_full = r_series[r_short]
                # Filter aligned non-None pairs
                pairs = [(x, y) for x, y in zip(xs_full, ys_full)
                         if x is not None and y is not None]
                rho, p, n_pairs = None, None, len(pairs)
                if len(pairs) >= 4 and len({p[0] for p in pairs}) >= 2 and len({p[1] for p in pairs}) >= 2:
                    xs2 = [p[0] for p in pairs]; ys2 = [p[1] for p in pairs]
                    rho, p = spearmanr(xs2, ys2)
                    if hasattr(rho, "item"): rho = rho.item()
                    if hasattr(p, "item"): p = p.item()
                out_rows.append({
                    "conv_id": conv_id, "dataset": ds, "topic": topic,
                    "annotator": annotator, "llm_slot": slot, "llm_name": name,
                    "u": u_short, "r": r_short, "n_pairs": n_pairs,
                    "rho": rho, "p": p, "T": T,
                })

print(f"Per-conversation cells computed: {len(out_rows)}")

# Save
fields = ["conv_id", "dataset", "topic", "annotator", "llm_slot", "llm_name",
          "u", "r", "n_pairs", "rho", "p", "T"]
out_csv = DATA_OUT / "per_conversation_correlations.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in out_rows:
        w.writerow(row)
print(f"Wrote {out_csv.relative_to(HERE.parent)}")

# ---------- 3. summary ----------
# For each (model, U, R) cell, report: N convs with computable ρ; mean ρ across convs; SD across convs;
# median ρ; fraction of convs with |ρ|>=0.3 ('practically large')
import statistics
summary = {"per_cell": {}, "n_cells": 0}
by_cell = defaultdict(list)
for r in out_rows:
    if r["rho"] is None: continue
    by_cell[(r["llm_name"], r["u"], r["r"])].append(r["rho"])
for (m, u, rj), rhos in by_cell.items():
    if not rhos: continue
    n = len(rhos)
    summary["per_cell"][f"{m}|{u}|{rj}"] = {
        "n_convs": n,
        "mean": round(statistics.mean(rhos), 4),
        "median": round(statistics.median(rhos), 4),
        "sd": round(statistics.stdev(rhos), 4) if n > 1 else 0,
        "frac_abs_ge_0.3": round(sum(1 for x in rhos if abs(x) >= 0.3) / n, 3),
        "frac_pos_ge_0.3": round(sum(1 for x in rhos if x >= 0.3) / n, 3),
        "frac_neg_le_-0.3": round(sum(1 for x in rhos if x <= -0.3) / n, 3),
        "min": round(min(rhos), 3),
        "max": round(max(rhos), 3),
    }
summary["n_cells"] = len(by_cell)

# Per-conversation richness: how many cells per (conv, LLM) have |ρ|>=0.3
by_conv_llm = defaultdict(int)
total_by_conv_llm = defaultdict(int)
for r in out_rows:
    key = (r["conv_id"], r["llm_name"])
    if r["rho"] is None: continue
    total_by_conv_llm[key] += 1
    if abs(r["rho"]) >= 0.3:
        by_conv_llm[key] += 1
summary["per_conv_llm_richness"] = [
    {"conv_id": c, "llm": l, "n_cells_strong": by_conv_llm[(c, l)],
     "n_cells_total": total_by_conv_llm[(c, l)]}
    for (c, l) in sorted(total_by_conv_llm)
]

with open(DATA_OUT / "per_conversation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote per_conversation_summary.json")

# ---------- print headlines ----------
print("\n=== Within-conversation distribution for top pooled cells ===")
for cell in ["U3|SEN", "U1|SEN", "U1|AUR", "U3|EMP", "U1|LM"]:
    print(f"\n{cell}:")
    for m in LLM_NAMES.values():
        info = summary["per_cell"].get(f"{m}|{cell}")
        if info:
            print(f"  {m:<8}  n={info['n_convs']:3d}  mean={info['mean']:+.3f}  "
                  f"median={info['median']:+.3f}  sd={info['sd']:.3f}  "
                  f"|ρ|≥0.3 in {info['frac_abs_ge_0.3']*100:.0f}% of convs")
