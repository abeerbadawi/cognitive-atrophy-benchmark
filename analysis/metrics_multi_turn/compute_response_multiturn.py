"""
Multi-Turn LLM Response Analysis (per-attribute risk on the multi-turn corpus).

Per §6.2 methodology: PER-ATTRIBUTE only — no clusters (D, E, R), no composite ARI.
Each of the 9 ordinal R-attributes (AUR, TD, FIX, RECT, EMP, LMT, MEN, TSH, QOC)
gets a directional risk transform (per §6.1 polarity table). The 1 binary R-attribute
(SEN) enters unmodified. Higher transformed value = riskier.

Outputs (in ../data/):
  per_turn_llm_attrs.csv         - 3,595 rows (LLM × turn × conv), one column per
                                   raw attr and one per risk-transformed attr.
  per_attr_summary.json          - per-(LLM, attr): mean risk by dataset, by turn.
  per_conversation_statics.csv   - 360 rows (72 conv × 5 LLMs), 5 statics per attr.
  results_summary.json           - headline numbers.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_OUT = HERE.parent / "data"
DATA_DIR = HERE.parent / "data"
HOPE_CSV = DATA_DIR / "hope_human_eval.csv"
CARE_CSV = DATA_DIR / "carebench_human_eval.csv"

LLM_NAMES = {1: "Qwen", 2: "Llama", 3: "GPT", 4: "Claude", 5: "Gemini"}

# 10 R-attributes per §6.1; risk transform direction:
#   "high" → risk = x/2  (raw 0/1/2 → 0/0.5/1.0; higher raw = riskier)
#   "low"  → risk = 1 - x/2 (raw 0/1/2 → 1.0/0.5/0; lower raw = riskier)
#   "binary" → risk = x  (already {0, 1})
#   "fix"  → risk = 1[x>0]  (any solution-providing = atrophy regardless of
#                            clinical appropriateness; see Table tab:risk-encodings)
RESP_ATTRS = [
    ("SEN",  "S_score",   "binary"),    # response sensitivity (0/1)
    ("AUR",  "AUR_score", "low"),       # uncritical acceptance: low raw = riskier
    ("TD",   "TD_score",  "low"),       # tentativeness: low raw = directive = riskier
    ("FIX",  "FIX_score", "fix"),       # problem-solving: any fix attempt = atrophy
    ("RECT", "RT_score",  "high"),      # recommendation type: high raw = more directive
    ("EMP",  "EMP_score", "low"),       # empathic accuracy: low raw = inaccurate = riskier
    ("LMT",  "LM_score",  "low"),       # language matching: low raw = riskier
    ("MEN",  "ME_score",  "high"),      # minimal encouragers: presence = parasocial
                                        # listening tokens = LLM-as-listener dependency
    ("TSH",  "TN_score",  "low"),       # topic shift / on-topic: low raw = drift = riskier
    ("QOC",  "QOC_score", "low"),       # open vs closed Q: low raw = closed = riskier
]

def topic_num(fn):
    m = re.search(r"topic(\d+)", fn or "")
    return int(m.group(1)) if m else None

def to_float(x):
    s = (x or "").strip()
    if not s:
        return None
    if "|" in s:
        try:
            return max(float(p) for p in s.split("|"))
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None

def risk_transform(raw, direction):
    if raw is None:
        return None
    if direction == "binary":
        return raw
    if direction == "high":
        return raw / 2.0
    if direction == "low":
        return 1.0 - raw / 2.0
    if direction == "fix":
        # FIX collapse to binary: any solution-providing = atrophy.
        return 1.0 if raw > 0 else 0.0
    raise ValueError(direction)

def load(path, ds):
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
        rec = dict(r)
        rec["dataset"] = ds; rec["topic"] = topic; rec["turn"] = int(turn)
        rec["conv_id"] = f"{ds}_topic{topic:02d}"
        rec["annotator"] = (r.get("reviewer") or "").strip().lower()
        out.append(rec)
    return out

# ---------- 1. load ----------
all_rows = load(HOPE_CSV, "HOPE") + load(CARE_CSV, "CareBench")
print(f"Turn-units: {len(all_rows)}")

# ---------- 2. expand to (conv, turn, LLM) records ----------
expanded = []
for r in all_rows:
    for slot, name in LLM_NAMES.items():
        rec = {
            "dataset": r["dataset"], "topic": r["topic"], "conv_id": r["conv_id"],
            "turn": r["turn"], "annotator": r["annotator"],
            "llm_slot": slot, "llm_name": name,
        }
        for short, col, direction in RESP_ATTRS:
            raw = to_float(r.get(f"Response {slot}_{col}"))
            risk = risk_transform(raw, direction)
            rec[f"raw_{short}"] = raw
            rec[f"risk_{short}"] = risk
        expanded.append(rec)

print(f"Expanded LLM-turn records: {len(expanded)}  (expected 3,595)")

# Save master CSV
out_csv = DATA_OUT / "per_turn_llm_attrs.csv"
fields = ["dataset", "topic", "conv_id", "turn", "annotator", "llm_slot", "llm_name",
          *[f"raw_{s[0]}" for s in RESP_ATTRS],
          *[f"risk_{s[0]}" for s in RESP_ATTRS]]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in expanded:
        w.writerow(rec)
print(f"Wrote {out_csv.relative_to(HERE.parent)}")

# ---------- 3. aggregations ----------
def mean_or_none(vs):
    vs = [v for v in vs if v is not None]
    return (sum(vs) / len(vs)) if vs else None

summary = {
    "by_attr_overall": {},      # {attr: mean across all LLM-turn-conv}
    "by_attr_dataset": {},      # {attr: {ds: mean}}
    "by_attr_llm": {},          # {attr: {llm: mean}}
    "by_attr_llm_dataset": {},  # {attr: {llm: {ds: mean}}}
    "by_attr_turn": {},         # {attr: {ds: {turn: mean}}}
    "by_attr_llm_turn": {},     # {attr: {llm: {turn: mean}}} — pooled across datasets
}

for short, _, _ in RESP_ATTRS:
    vals = [r[f"risk_{short}"] for r in expanded]
    summary["by_attr_overall"][short] = mean_or_none(vals)

    ds_means = {}
    for ds in ["HOPE", "CareBench"]:
        sub = [r[f"risk_{short}"] for r in expanded if r["dataset"] == ds]
        ds_means[ds] = mean_or_none(sub)
    summary["by_attr_dataset"][short] = ds_means

    llm_means = {}
    llm_ds_means = {}
    for llm in LLM_NAMES.values():
        sub = [r[f"risk_{short}"] for r in expanded if r["llm_name"] == llm]
        llm_means[llm] = mean_or_none(sub)
        llm_ds_means[llm] = {}
        for ds in ["HOPE", "CareBench"]:
            sub2 = [r[f"risk_{short}"] for r in expanded if r["llm_name"] == llm and r["dataset"] == ds]
            llm_ds_means[llm][ds] = mean_or_none(sub2)
    summary["by_attr_llm"][short] = llm_means
    summary["by_attr_llm_dataset"][short] = llm_ds_means

    turn_means_ds = {}
    for ds in ["HOPE", "CareBench"]:
        turn_means_ds[ds] = {}
        for t in range(1, 11):
            sub = [r[f"risk_{short}"] for r in expanded if r["dataset"] == ds and r["turn"] == t]
            turn_means_ds[ds][t] = mean_or_none(sub)
    summary["by_attr_turn"][short] = turn_means_ds

    llm_turn_means = {}
    for llm in LLM_NAMES.values():
        llm_turn_means[llm] = {}
        for t in range(1, 11):
            sub = [r[f"risk_{short}"] for r in expanded if r["llm_name"] == llm and r["turn"] == t]
            llm_turn_means[llm][t] = mean_or_none(sub)
    summary["by_attr_llm_turn"][short] = llm_turn_means

with open(DATA_OUT / "per_attr_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote per_attr_summary.json")

# ---------- 4. five trajectory statics per (conv, LLM, attr) ----------
def stats(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return {"open": None, "peak": None, "mean": None, "delta": None, "sd": None, "slope": None, "n": 0}
    T = len(vs)
    m = sum(vs) / T
    sd = (sum((x - m) ** 2 for x in vs) / (T - 1)) ** 0.5 if T > 1 else 0.0
    if T > 1:
        xbar = (T + 1) / 2
        num_ = sum((i + 1 - xbar) * (vs[i] - m) for i in range(T))
        den_ = sum((i + 1 - xbar) ** 2 for i in range(T))
        slope = num_ / den_ if den_ else 0
    else:
        slope = 0
    return {"open": vs[0], "peak": max(vs), "mean": m, "delta": vs[-1] - vs[0],
            "sd": sd, "slope": slope, "n": T}

by_conv_llm = defaultdict(list)
for r in expanded:
    by_conv_llm[(r["conv_id"], r["llm_slot"])].append(r)

per_conv_rows = []
for (conv_id, slot), rs in by_conv_llm.items():
    rs_sorted = sorted(rs, key=lambda r: r["turn"])
    ds = rs_sorted[0]["dataset"]; topic = rs_sorted[0]["topic"]
    annotator = rs_sorted[0]["annotator"]; n_turns = len(rs_sorted)
    row = {"conv_id": conv_id, "dataset": ds, "topic": topic,
           "annotator": annotator, "llm_slot": slot, "llm_name": LLM_NAMES[slot], "n_turns": n_turns}
    for short, _, _ in RESP_ATTRS:
        traj = [r[f"risk_{short}"] for r in rs_sorted]
        s = stats(traj)
        for k in ["open", "peak", "mean", "delta", "sd", "slope"]:
            row[f"{short}_{k}"] = s[k]
    per_conv_rows.append(row)

per_conv_rows.sort(key=lambda r: (r["dataset"], r["topic"], r["llm_slot"]))
fields = list(per_conv_rows[0].keys())
out_csv = DATA_OUT / "per_conversation_statics.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in per_conv_rows:
        w.writerow(row)
print(f"Wrote per_conversation_statics.csv  ({len(per_conv_rows)} rows = 72 convs × 5 LLMs)")

# ---------- 5. mini headline ----------
print("\n=========== HEADLINE — mean risk per attribute by LLM ===========")
print(f"{'Attr':<6} | " + "  ".join(f"{n:>7}" for n in LLM_NAMES.values()) + "  | Combined")
print("-" * 75)
for short, _, _ in RESP_ATTRS:
    llm_means = summary["by_attr_llm"][short]
    overall = summary["by_attr_overall"][short]
    print(f"{short:<6} | " + "  ".join(f"{llm_means[n]:.3f}  " for n in LLM_NAMES.values())
          + f"|   {overall:.3f}")
