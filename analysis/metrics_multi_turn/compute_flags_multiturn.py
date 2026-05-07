"""
Multi-Turn Binary Flags Analysis (5 flags × 5 LLMs across 720 turns).
Mirrors single-turn binary flags analysis but adds the per-turn dimension.

Per-attribute (no clusters): each of the 5 flags is treated independently.
The per-response burden F(r_t) = mean of the 5 flags is also computed per
(conversation, LLM, turn) since it appears in §6.1 and is useful as a
single-line summary signal for trajectory visualisation.

Outputs:
  per_turn_flags.csv             - 3,595 rows (719 turn-units × 5 LLMs)
  per_flag_firing_rates.json     - per-flag firing rates by (dataset, LLM, turn)
  per_conversation_flag_statics.csv - 72 conversations × 5 LLMs × {5 flags + F-burden} × 5 statics
  results_summary.json           - headline numbers
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
HOPE_CSV = DATA_DIR / "hope_human_eval.csv"
CARE_CSV = DATA_DIR / "carebench_human_eval.csv"

# 5 binary flags as defined in single-turn schema
FLAGS = [
    ("F1_decisive",   "yn_decisive",   "Directive"),
    ("F2_assumes",    "yn_assumes",    "Assumes underlying"),
    ("F3_introduces", "yn_introduces", "Introduces new content"),
    ("F4_harmful",    "yn_harmful",    "Harmful validation"),
    ("F5_incoherent", "yn_incoherent", "Incoherent"),
]
LLM_SLOTS = list(range(1, 6))   # Response 1..5
LLM_NAMES = {1: "Qwen", 2: "Llama", 3: "GPT", 4: "Claude", 5: "Gemini"}

def topic_num(filename):
    m = re.search(r"topic(\d+)", filename or "")
    return int(m.group(1)) if m else None

def to_int(x):
    s = (x or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None

def load(path, dataset_label):
    with open(path) as f:
        first = f.readline().strip()
        if "," in first:
            f.seek(0)
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_dataset"] = dataset_label
        r["_topic"] = topic_num(r.get("filename", ""))
        r["_turn"] = to_int(r.get("turn"))
        r["_annotator"] = (r.get("reviewer") or "").strip().lower()
    return rows

# ---------- 1. load ----------
hope_rows = load(HOPE_CSV, "HOPE")
care_rows = load(CARE_CSV, "CareBench")
all_rows = hope_rows + care_rows

print(f"HOPE rows:      {len(hope_rows)}")
print(f"CareBench rows: {len(care_rows)}")
print(f"Total turn-units: {len(all_rows)}  (× 5 LLMs = {len(all_rows) * 5} turn-LLM units)")

# ---------- 2. expand to (conv, turn, LLM) records with all 5 flag values ----------
expanded = []
for r in all_rows:
    if r["_topic"] is None or r["_turn"] is None:
        continue
    conv_id = f"{r['_dataset']}_topic{r['_topic']:02d}"
    for slot in LLM_SLOTS:
        rec = {
            "dataset": r["_dataset"],
            "topic": r["_topic"],
            "conv_id": conv_id,
            "turn": r["_turn"],
            "annotator": r["_annotator"],
            "llm_slot": slot,
            "llm_name": LLM_NAMES[slot],
        }
        flag_vals = []
        any_missing = False
        for short, col, _label in FLAGS:
            raw = to_int(r.get(f"Response {slot}_{col}"))
            if raw is None:
                any_missing = True
                rec[short] = None
            else:
                rec[short] = raw
                flag_vals.append(raw)
        if any_missing or not flag_vals:
            rec["F_burden"] = None
        else:
            rec["F_burden"] = sum(flag_vals) / 5.0
        expanded.append(rec)

print(f"Expanded records: {len(expanded)}  (719 × 5 = 3,595 expected)")

# Save the per-turn-LLM master CSV
out_csv = DATA_DIR / "per_turn_flags.csv"
with open(out_csv, "w", newline="") as f:
    fields = ["dataset", "topic", "conv_id", "turn", "annotator", "llm_slot", "llm_name",
              *[s[0] for s in FLAGS], "F_burden"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in expanded:
        w.writerow(rec)
print(f"Wrote {out_csv.name}")

# ---------- 3. per-flag firing rates ----------
def firing_rate(records, flag_short):
    vals = [r[flag_short] for r in records if r[flag_short] is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)

rates = {
    "by_dataset": {},        # {dataset: {flag: {rate, n}}}
    "by_llm": {},            # {llm_slot: {flag: {rate, n}}}
    "by_dataset_llm": {},    # {dataset: {llm_slot: {flag: {rate, n}, F_burden_mean}}}
    "by_turn": {},           # {dataset: {turn: {flag: {rate, n}}}}
    "by_dataset_combined": {},   # {flag: {rate, n}}
}

for ds in ["HOPE", "CareBench"]:
    sub = [r for r in expanded if r["dataset"] == ds]
    rates["by_dataset"][ds] = {}
    for short, _col, _label in FLAGS:
        rate, n = firing_rate(sub, short)
        rates["by_dataset"][ds][short] = {"rate": rate, "n": n}
    # F-burden mean
    burdens = [r["F_burden"] for r in sub if r["F_burden"] is not None]
    rates["by_dataset"][ds]["F_burden_mean"] = sum(burdens) / len(burdens) if burdens else None
    rates["by_dataset"][ds]["n_responses"] = len(burdens)

# combined across both datasets
for short, _col, _label in FLAGS:
    rate, n = firing_rate(expanded, short)
    rates["by_dataset_combined"][short] = {"rate": rate, "n": n}
burdens = [r["F_burden"] for r in expanded if r["F_burden"] is not None]
rates["by_dataset_combined"]["F_burden_mean"] = sum(burdens) / len(burdens)
rates["by_dataset_combined"]["n_responses"] = len(burdens)

# by LLM (combined across datasets)
for slot in LLM_SLOTS:
    sub = [r for r in expanded if r["llm_slot"] == slot]
    rates["by_llm"][slot] = {}
    for short, _col, _label in FLAGS:
        rate, n = firing_rate(sub, short)
        rates["by_llm"][slot][short] = {"rate": rate, "n": n}
    burdens = [r["F_burden"] for r in sub if r["F_burden"] is not None]
    rates["by_llm"][slot]["F_burden_mean"] = sum(burdens) / len(burdens) if burdens else None
    rates["by_llm"][slot]["n_responses"] = len(burdens)

# by dataset × LLM
for ds in ["HOPE", "CareBench"]:
    rates["by_dataset_llm"][ds] = {}
    for slot in LLM_SLOTS:
        sub = [r for r in expanded if r["dataset"] == ds and r["llm_slot"] == slot]
        rates["by_dataset_llm"][ds][slot] = {}
        for short, _col, _label in FLAGS:
            rate, n = firing_rate(sub, short)
            rates["by_dataset_llm"][ds][slot][short] = {"rate": rate, "n": n}
        burdens = [r["F_burden"] for r in sub if r["F_burden"] is not None]
        rates["by_dataset_llm"][ds][slot]["F_burden_mean"] = sum(burdens) / len(burdens) if burdens else None
        rates["by_dataset_llm"][ds][slot]["n_responses"] = len(burdens)

# by dataset × turn
for ds in ["HOPE", "CareBench"]:
    rates["by_turn"][ds] = {}
    for t in range(1, 11):
        sub = [r for r in expanded if r["dataset"] == ds and r["turn"] == t]
        rates["by_turn"][ds][t] = {}
        for short, _col, _label in FLAGS:
            rate, n = firing_rate(sub, short)
            rates["by_turn"][ds][t][short] = {"rate": rate, "n": n}
        burdens = [r["F_burden"] for r in sub if r["F_burden"] is not None]
        rates["by_turn"][ds][t]["F_burden_mean"] = sum(burdens) / len(burdens) if burdens else None
        rates["by_turn"][ds][t]["n_responses"] = len(burdens)

with open(DATA_DIR / "per_flag_firing_rates.json", "w") as f:
    json.dump(rates, f, indent=2)
print(f"Wrote per_flag_firing_rates.json")

# ---------- 4. per-conversation 5 trajectory statics ----------
# Per (conversation, LLM): 10-turn trajectory of each flag and F-burden.

def stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"open": None, "peak": None, "mean": None, "delta": None, "sd": None, "n": 0, "slope": None}
    T = len(vals); m = sum(vals) / T
    sd = (sum((x - m) ** 2 for x in vals) / (T - 1)) ** 0.5 if T > 1 else 0.0
    # slope (OLS, on x=turn-index)
    if T > 1:
        xbar = (T + 1) / 2
        num_ = sum((i + 1 - xbar) * (vals[i] - m) for i in range(T))
        den_ = sum((i + 1 - xbar) ** 2 for i in range(T))
        slope = num_ / den_ if den_ else 0
    else:
        slope = 0
    return {"open": vals[0], "peak": max(vals), "mean": m, "delta": vals[-1] - vals[0],
            "sd": sd, "n": T, "slope": slope}

per_conv_llm = []
by_conv_llm = defaultdict(list)
for r in expanded:
    by_conv_llm[(r["conv_id"], r["llm_slot"])].append(r)
for (conv_id, slot), rs in by_conv_llm.items():
    rs = sorted(rs, key=lambda r: r["turn"])
    ds = rs[0]["dataset"]
    topic = rs[0]["topic"]
    annotator = rs[0]["annotator"]
    n_turns = len(rs)
    row = {"conv_id": conv_id, "dataset": ds, "topic": topic,
           "annotator": annotator, "llm_slot": slot, "llm_name": LLM_NAMES[slot],
           "n_turns": n_turns}
    # Each flag
    for short, _col, _label in FLAGS:
        vals = [r[short] for r in rs]
        s = stats(vals)
        for k in ["open", "peak", "mean", "delta", "sd", "slope"]:
            row[f"{short}_{k}"] = s[k]
    # F-burden
    vals = [r["F_burden"] for r in rs]
    s = stats(vals)
    for k in ["open", "peak", "mean", "delta", "sd", "slope"]:
        row[f"F_burden_{k}"] = s[k]
    per_conv_llm.append(row)

per_conv_llm.sort(key=lambda r: (r["dataset"], r["topic"], r["llm_slot"]))
fields = list(per_conv_llm[0].keys())
with open(DATA_DIR / "per_conversation_flag_statics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in per_conv_llm:
        w.writerow(row)
print(f"Wrote per_conversation_flag_statics.csv  ({len(per_conv_llm)} rows = 72 convs × 5 LLMs)")

# ---------- 5. summary ----------
def fmt(x): return None if x is None else round(x, 4)
summary = {
    "datasets": {},
    "n_conversations_total": 72,
    "n_turn_units": len(all_rows),
    "n_response_units": len(expanded),
    "flags": {short: label for short, _col, label in FLAGS},
}
for ds in ["HOPE", "CareBench"]:
    summary["datasets"][ds] = {
        "n_responses": rates["by_dataset"][ds]["n_responses"],
        "F_burden_mean": fmt(rates["by_dataset"][ds]["F_burden_mean"]),
        "per_flag_firing_rate": {short: fmt(rates["by_dataset"][ds][short]["rate"]) for short, _, _ in FLAGS},
    }
summary["combined"] = {
    "n_responses": rates["by_dataset_combined"]["n_responses"],
    "F_burden_mean": fmt(rates["by_dataset_combined"]["F_burden_mean"]),
    "per_flag_firing_rate": {short: fmt(rates["by_dataset_combined"][short]["rate"]) for short, _, _ in FLAGS},
}

with open(DATA_DIR / "flags_results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote results_summary.json")

# ---------- print headlines ----------
print("\n=========== HEADLINE ===========")
for ds in ["HOPE", "CareBench"]:
    s = summary["datasets"][ds]
    print(f"{ds:<10}: F̄ = {s['F_burden_mean']:.3f}  |  per-flag rates: " +
          "  ".join(f"{short.split('_')[0]}={s['per_flag_firing_rate'][short]*100:.0f}%" for short, _, _ in FLAGS))
sc = summary["combined"]
print(f"Combined  : F̄ = {sc['F_burden_mean']:.3f}  |  per-flag rates: " +
      "  ".join(f"{short.split('_')[0]}={sc['per_flag_firing_rate'][short]*100:.0f}%" for short, _, _ in FLAGS))
