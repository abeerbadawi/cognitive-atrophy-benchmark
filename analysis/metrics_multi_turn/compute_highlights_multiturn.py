"""
Compute the 5 trajectory statics per (LLM x attribute) for the multi-turn
highlights analysis.

Five statics, computed from the response-side risk-transformed values:
  open  = mean Risk at turn 1
  peak  = max  over turns 1..10 of mean Risk
  mean  = mean Risk over all 10 turns
  delta = mean Risk at turn 10 - mean Risk at turn 1
  sigma = pooled within-conversation SD across turns
          (sqrt of mean per-conversation variance over turns)

Plus: per-turn mean (10 values), peak_turn, panel z-score, raw 0/1/2 score
distribution per (LLM, attr, turn) for the score-distribution report.

Inputs:
  ../../multi_turn_llm_response_analysis/data/per_turn_llm_attrs.csv
  ../../multi_turn_llm_response_analysis/data/per_attr_summary.json
  ../../multi_turn_llm_response_analysis/data/per_model_behavior.json

Outputs (multi_turn_llm_highlights/data/):
  per_model_statics.csv      (5 LLMs * 10 attrs = 50 rows; 5 statics + per-turn)
  per_model_statics.json     (same data, JSON keyed [llm][attr])
  panel_per_attr.json        (per-attr panel mean of each static + SD across LLMs)
  score_distribution.json    (per (LLM, attr, turn) raw 0/1/2 share + count)
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DATA = HERE.parent / "data"
RESP_DATA = HERE.parent / "data"  # produced by compute_response_multiturn.py

LLMS = ["Qwen", "Llama", "GPT", "Claude", "Gemini"]
ATTRS = ["SEN", "AUR", "TD", "FIX", "RECT", "EMP", "LMT", "MEN", "TSH", "QOC"]
TURNS = list(range(1, 11))

# Map raw 0/1/2 to risk for each attribute (matching response analysis transforms)
HIGH_IS_RISK = {"FIX", "RECT"}     # x/2
LOW_IS_RISK  = {"AUR", "TD", "EMP", "LMT", "MEN", "TSH", "QOC"}  # 1 - x/2
BINARY      = {"SEN"}             # x


# ---------- 1. Load per-turn raw + risk values ----------
rows = []
with open(RESP_DATA / "per_turn_llm_attrs.csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
print(f"loaded {len(rows)} (conv, turn, llm) rows")

# Index: per (llm, attr, turn) -> list of risk values
by_lat = defaultdict(list)         # (llm, attr, turn) -> [risk]
by_la  = defaultdict(list)         # (llm, attr)        -> [risk]   (whole panel)
by_la_conv = defaultdict(lambda: defaultdict(list))  # (llm, attr) -> {(conv, turn) -> risk}; for sigma
by_la_raw_per_turn = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
# (llm, attr) -> {turn -> [count_0, count_1, count_2]}; SEN gets [count_0, count_1, 0]

for r in rows:
    llm = r["llm_name"]
    if llm not in LLMS:
        continue
    turn = int(r["turn"])
    if turn not in TURNS:
        continue
    conv = r["conv_id"]
    for attr in ATTRS:
        rk = r.get(f"risk_{attr}", "")
        rw = r.get(f"raw_{attr}", "")
        if rk in ("", None):
            continue
        v = float(rk)
        by_lat[(llm, attr, turn)].append(v)
        by_la[(llm, attr)].append(v)
        by_la_conv[(llm, attr)][(conv, turn)] = v

        if rw not in ("", None):
            raw = int(round(float(rw)))
            if attr in BINARY:
                # SEN raw 0/1
                idx = max(0, min(1, raw))
                by_la_raw_per_turn[(llm, attr)][turn][idx] += 1
            else:
                # ordinal 0/1/2
                idx = max(0, min(2, raw))
                by_la_raw_per_turn[(llm, attr)][turn][idx] += 1


# ---------- 2. Compute per-(LLM, attr) trajectory statics ----------
def safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


per_model_statics = {m: {} for m in LLMS}
csv_rows = []

for m in LLMS:
    for a in ATTRS:
        per_turn = [safe_mean(by_lat[(m, a, t)]) for t in TURNS]
        opn   = per_turn[0]
        end   = per_turn[-1]
        peak  = float(np.nanmax(per_turn))
        mn    = float(np.nanmean(per_turn))
        dlta  = end - opn
        # Pooled within-conv-pair SD across turns:
        #   group risk values by conv; for each conv compute SD across its turns;
        #   then pool by averaging variance and taking sqrt.
        conv_to_vals = defaultdict(list)
        for (conv, t), v in by_la_conv[(m, a)].items():
            conv_to_vals[conv].append(v)
        variances = [float(np.var(vs, ddof=0)) for vs in conv_to_vals.values() if len(vs) >= 2]
        sigma = float(np.sqrt(np.mean(variances))) if variances else 0.0

        peak_turn = int(np.argmax(per_turn)) + 1
        min_turn = int(np.argmin(per_turn)) + 1

        per_model_statics[m][a] = {
            "open": opn, "end": end,
            "peak": peak, "mean": mn,
            "delta": dlta, "sigma": sigma,
            "peak_turn": peak_turn, "min_turn": min_turn,
            "per_turn": per_turn,
        }
        csv_rows.append({
            "llm": m, "attr": a,
            "open": round(opn, 4), "end": round(end, 4),
            "peak": round(peak, 4), "mean": round(mn, 4),
            "delta": round(dlta, 4), "sigma": round(sigma, 4),
            "peak_turn": peak_turn, "min_turn": min_turn,
        })

# Panel stats per attr per static (across the 5 LLMs)
panel = {}
for a in ATTRS:
    panel[a] = {}
    for stat in ("open", "end", "peak", "mean", "delta", "sigma"):
        vals = [per_model_statics[m][a][stat] for m in LLMS]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        panel[a][stat] = {"panel_mean": mu, "panel_sd": sd,
                          "min": float(np.min(vals)), "max": float(np.max(vals))}

# Add per-LLM z-scores per static
for m in LLMS:
    for a in ATTRS:
        for stat in ("open", "end", "peak", "mean", "delta", "sigma"):
            ps = panel[a][stat]
            v = per_model_statics[m][a][stat]
            z = (v - ps["panel_mean"]) / ps["panel_sd"] if ps["panel_sd"] > 0 else 0.0
            per_model_statics[m][a][f"z_{stat}"] = z

# ---------- 3. Score distribution ----------
score_dist = {}
for m in LLMS:
    score_dist[m] = {}
    for a in ATTRS:
        score_dist[m][a] = {}
        for t in TURNS:
            counts = by_la_raw_per_turn[(m, a)][t][:]
            n = sum(counts)
            shares = [c / n if n else 0.0 for c in counts]
            score_dist[m][a][t] = {
                "counts": counts,
                "shares": [round(s, 4) for s in shares],
                "n": n,
            }

# ---------- 4. Save outputs ----------
OUT_DATA.mkdir(exist_ok=True, parents=True)

with open(OUT_DATA / "per_model_statics.csv", "w", newline="") as f:
    fieldnames = list(csv_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(csv_rows)
print(f"wrote per_model_statics.csv ({len(csv_rows)} rows)")

with open(OUT_DATA / "per_model_statics.json", "w") as f:
    json.dump({"statics": per_model_statics, "panel": panel,
               "llms": LLMS, "attrs": ATTRS, "turns": TURNS}, f, indent=2)
print("wrote per_model_statics.json")

with open(OUT_DATA / "panel_per_attr.json", "w") as f:
    json.dump(panel, f, indent=2)
print("wrote panel_per_attr.json")

with open(OUT_DATA / "score_distribution.json", "w") as f:
    json.dump({"score_dist": score_dist, "llms": LLMS,
               "attrs": ATTRS, "turns": TURNS,
               "binary_attrs": list(BINARY)}, f, indent=2)
print("wrote score_distribution.json")

# ---------- 5. Print headline ----------
print("\n=== Per-LLM 5-statics summary (mean across attrs) ===")
print(f"{'LLM':<8}  open  peak  mean   delta  sigma")
for m in LLMS:
    o = np.mean([per_model_statics[m][a]["open"] for a in ATTRS])
    p = np.mean([per_model_statics[m][a]["peak"] for a in ATTRS])
    mn = np.mean([per_model_statics[m][a]["mean"] for a in ATTRS])
    d = np.mean([per_model_statics[m][a]["delta"] for a in ATTRS])
    s = np.mean([per_model_statics[m][a]["sigma"] for a in ATTRS])
    print(f"{m:<8}  {o:.2f}  {p:.2f}  {mn:.2f}  {d:+.3f}  {s:.3f}")
