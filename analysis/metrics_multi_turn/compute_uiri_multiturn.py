"""
Multi-Turn User Input Analysis (UIRI on HOPE and CareBench)
============================================================
Mirrors the single-turn analysis (compute_uiri.py) but operates on
the 72 multi-turn conversations × 10 turns each (HOPE 36 + CareBench 36).

Pipeline:
  1. Load both merged-reviewer CSVs (HOPE, CareBench).
  2. Normalise reviewer casing.
  3. For every (conversation, turn) compute:
       - per-attribute normalised user score U_i(t) / k_i in [0, 1]
       - UIRI(t) = mean of the five normalised attributes
       - UIRI band (Low / Med / High)
  4. For every conversation, compute the 5 trajectory statics
     (open / peak / mean / Δ / σ) for UIRI(t) and for each U_i(t).
  5. Per-attribute marginals: count + % of (conversation, turn) cells
     at each level, broken down per dataset, per conversation, then
     averaged across conversations per dataset.
  6. UIRI band counts per conversation; averaged across conversations
     per dataset.
  7. Save:
       - per_turn_uiri.csv           (720 rows: every (conv, turn))
       - per_conversation_statics.csv (72 rows × 5 statics × 6 signals)
       - per_attribute_marginals.json
       - uiri_band_distributions.json
       - results_summary.json
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

# ---------- paths ----------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = HERE
HOPE_CSV = DATA_DIR / "hope_human_eval.csv"
CARE_CSV = DATA_DIR / "carebench_human_eval.csv"

# ---------- attribute spec ----------
USER_ATTRS = [
    ("U1_typicality", "user_typicality", 2, "typical · less typical · atypical"),
    ("U2_evocative", "user_evocative", 2, "not emotional · somewhat · very evocative"),
    ("U3_sensitivity", "user_sensitivity", 1, "no overt risk · self-harm/suicide"),
    ("U4_fix_seeking", "user_request_info", 1, "not asking · asks for fix"),
    ("U5_underlying", "user_underlying", 2, "no latent · possible · clear"),
]

LEVEL_LABELS = {
    "U1_typicality": ["typical", "less typical", "atypical"],
    "U2_evocative": ["not emotional", "somewhat", "very evocative"],
    "U3_sensitivity": ["no overt risk", "self-harm/suicide"],
    "U4_fix_seeking": ["not asking", "asks for fix"],
    "U5_underlying": ["no latent", "possible", "clear"],
}

BAND_CUT_LOW, BAND_CUT_HIGH = 0.30, 0.60

# ---------- helpers ----------
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

def band(uiri):
    if uiri < BAND_CUT_LOW:
        return "Low"
    if uiri < BAND_CUT_HIGH:
        return "Medium"
    return "High"

def load(path, dataset_label):
    """Load merged CSV; handle a custom-style pre-header line if present."""
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

# Verify totals
print(f"Loaded HOPE rows:      {len(hope_rows)}")
print(f"Loaded CareBench rows: {len(care_rows)}")
print(f"Total turn-units:      {len(all_rows)}")

# ---------- 2. per-(conv, turn) compute ----------
per_turn = []
for r in all_rows:
    ds = r["_dataset"]
    topic = r["_topic"]
    turn = r["_turn"]
    if topic is None or turn is None:
        continue
    # Composite conversation key
    conv_id = f"{ds}_topic{topic:02d}"

    rec = {"dataset": ds, "topic": topic, "conv_id": conv_id, "turn": turn,
           "annotator": r["_annotator"]}
    raw, norm = {}, {}
    for short, col, k, _ in USER_ATTRS:
        raw_v = to_int(r.get(col))
        if raw_v is None:
            norm_v = None
        else:
            norm_v = raw_v / k
        raw[short] = raw_v
        norm[short] = norm_v
        rec[f"raw_{short}"] = raw_v
        rec[f"norm_{short}"] = norm_v

    if any(v is None for v in norm.values()):
        rec["UIRI"] = None
        rec["band"] = None
    else:
        u = sum(norm.values()) / 5.0
        rec["UIRI"] = u
        rec["band"] = band(u)
    per_turn.append(rec)

print(f"\nPer-turn records built: {len(per_turn)}")

# ---------- 3. save per_turn_uiri.csv ----------
per_turn_csv = OUT_DIR / "per_turn_uiri.csv"
with open(per_turn_csv, "w", newline="") as f:
    fields = ["dataset", "topic", "conv_id", "turn", "annotator",
              *[f"raw_{s[0]}" for s in USER_ATTRS],
              *[f"norm_{s[0]}" for s in USER_ATTRS],
              "UIRI", "band"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in per_turn:
        w.writerow(rec)
print(f"Wrote {per_turn_csv.name} ({len(per_turn)} rows)")

# ---------- 4. per-attribute marginals ----------
# Per-conversation: count + % at each level (over 10 turns), per attribute, per dataset.
# Then average across conversations per dataset.

def compute_marginals(per_turn_records):
    """Return nested dict:
       marginals[dataset][attr]['per_conv'] = [{conv_id, level_counts: {0: n, 1: n, ...}, n_turns}]
       marginals[dataset][attr]['per_dataset_total_counts']  - sum across all conv
       marginals[dataset][attr]['per_dataset_avg_pct']       - mean across conversations of the per-conv pct
    """
    out = {}
    by_ds_conv = defaultdict(lambda: defaultdict(list))
    for r in per_turn_records:
        by_ds_conv[r["dataset"]][r["conv_id"]].append(r)

    for ds in by_ds_conv:
        out[ds] = {}
        for attr_short, _col, k, _meanings in USER_ATTRS:
            n_levels = k + 1   # 0..k inclusive
            per_conv_entries = []
            sum_total = [0] * n_levels
            sum_pct_across_convs = [0.0] * n_levels
            n_convs_with_data = 0
            for conv_id, rs in by_ds_conv[ds].items():
                level_counts = [0] * n_levels
                n_obs = 0
                for r in rs:
                    raw_v = r.get(f"raw_{attr_short}")
                    if raw_v is None:
                        continue
                    level_counts[raw_v] += 1
                    sum_total[raw_v] += 1
                    n_obs += 1
                if n_obs == 0:
                    continue
                pct = [c / n_obs * 100.0 for c in level_counts]
                per_conv_entries.append({
                    "conv_id": conv_id,
                    "n_turns": n_obs,
                    "level_counts": level_counts,
                    "level_pct": pct,
                })
                for i in range(n_levels):
                    sum_pct_across_convs[i] += pct[i]
                n_convs_with_data += 1
            avg_pct = [s / n_convs_with_data for s in sum_pct_across_convs] if n_convs_with_data else [0]*n_levels
            out[ds][attr_short] = {
                "per_conv": per_conv_entries,
                "per_dataset_total_counts": sum_total,
                "per_dataset_total_n": sum(sum_total),
                "per_dataset_total_pct": [c / sum(sum_total) * 100.0 for c in sum_total] if sum(sum_total) else [0]*n_levels,
                "avg_per_conv_pct": avg_pct,
            }
    return out

marginals = compute_marginals(per_turn)
with open(OUT_DIR / "per_attribute_marginals.json", "w") as f:
    json.dump(marginals, f, indent=2)
print(f"Wrote per_attribute_marginals.json")

# ---------- 5. UIRI band distribution per conversation, per dataset, averaged ----------
def compute_band_distribution(per_turn_records):
    """Per conversation: count of turns in Low/Med/High, plus mean UIRI.
       Per dataset: sum of per-conv band counts, plus mean across conversations of per-conv pct."""
    out = {}
    by_ds_conv = defaultdict(lambda: defaultdict(list))
    for r in per_turn_records:
        if r["UIRI"] is None:
            continue
        by_ds_conv[r["dataset"]][r["conv_id"]].append(r)

    for ds in by_ds_conv:
        per_conv_entries = []
        sum_band = {"Low": 0, "Medium": 0, "High": 0}
        sum_pct = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
        sum_mean_uiri = 0.0
        n_convs = 0
        for conv_id, rs in by_ds_conv[ds].items():
            counts = {"Low": 0, "Medium": 0, "High": 0}
            uiris = []
            for r in rs:
                counts[r["band"]] += 1
                uiris.append(r["UIRI"])
            n_t = len(rs)
            pct = {b: counts[b] / n_t * 100.0 for b in counts}
            mu = mean(uiris)
            per_conv_entries.append({
                "conv_id": conv_id,
                "n_turns": n_t,
                "band_counts": counts,
                "band_pct": pct,
                "mean_uiri": mu,
                "min_uiri": min(uiris),
                "max_uiri": max(uiris),
            })
            for b in counts:
                sum_band[b] += counts[b]
                sum_pct[b] += pct[b]
            sum_mean_uiri += mu
            n_convs += 1
        out[ds] = {
            "per_conv": per_conv_entries,
            "n_conversations": n_convs,
            "total_band_counts": sum_band,
            "total_band_pct": {b: sum_band[b] / sum(sum_band.values()) * 100.0 for b in sum_band},
            "avg_per_conv_band_pct": {b: sum_pct[b] / n_convs for b in sum_pct},
            "avg_per_conv_mean_uiri": sum_mean_uiri / n_convs,
        }
    return out

band_dist = compute_band_distribution(per_turn)
with open(OUT_DIR / "uiri_band_distributions.json", "w") as f:
    json.dump(band_dist, f, indent=2)
print(f"Wrote uiri_band_distributions.json")

# ---------- 6. five trajectory statics per conversation, for UIRI and each U_i ----------
def stats(values):
    if not values or any(v is None for v in values):
        return {"open": None, "peak": None, "mean": None, "delta": None, "sd": None, "n": 0}
    T = len(values)
    m = sum(values) / T
    sd = (sum((x - m) ** 2 for x in values) / (T - 1)) ** 0.5 if T > 1 else 0.0
    return {
        "open": values[0],
        "peak": max(values),
        "mean": m,
        "delta": values[-1] - values[0],
        "sd": sd,
        "n": T,
    }

def compute_per_conversation_statics(per_turn_records):
    by_conv = defaultdict(list)
    for r in per_turn_records:
        by_conv[r["conv_id"]].append(r)
    rows = []
    for conv_id in sorted(by_conv.keys()):
        recs = sorted(by_conv[conv_id], key=lambda r: r["turn"])
        ds = recs[0]["dataset"]
        topic = recs[0]["topic"]
        annotator = recs[0]["annotator"]
        row = {"conv_id": conv_id, "dataset": ds, "topic": topic,
               "annotator": annotator, "n_turns": len(recs)}
        # UIRI trajectory
        uiri_traj = [r["UIRI"] for r in recs]
        s = stats(uiri_traj)
        for key in ["open", "peak", "mean", "delta", "sd"]:
            row[f"UIRI_{key}"] = s[key]
        # Each per-attribute trajectory (normalised)
        for short, _col, _k, _meanings in USER_ATTRS:
            traj = [r[f"norm_{short}"] for r in recs]
            s = stats(traj)
            for key in ["open", "peak", "mean", "delta", "sd"]:
                row[f"{short}_{key}"] = s[key]
        rows.append(row)
    return rows

per_conv_statics = compute_per_conversation_statics(per_turn)
out_csv = OUT_DIR / "per_conversation_statics.csv"
with open(out_csv, "w", newline="") as f:
    fieldnames = list(per_conv_statics[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in per_conv_statics:
        w.writerow(row)
print(f"Wrote per_conversation_statics.csv ({len(per_conv_statics)} conversations × {len(fieldnames)} columns)")

# ---------- 7. dataset-level summary ----------
summary = {
    "datasets": {},
    "overall": {},
    "n_conversations_total": len(per_conv_statics),
}
def fmt(x):
    return None if x is None else round(x, 4)

for ds in ["HOPE", "CareBench"]:
    convs = [r for r in per_conv_statics if r["dataset"] == ds]
    uiris = [r["UIRI_mean"] for r in convs]
    opens = [r["UIRI_open"] for r in convs]
    peaks = [r["UIRI_peak"] for r in convs]
    deltas = [r["UIRI_delta"] for r in convs]
    sds_within = [r["UIRI_sd"] for r in convs]
    summary["datasets"][ds] = {
        "n_conversations": len(convs),
        "n_turns_total": sum(r["n_turns"] for r in convs),
        "UIRI_open_mean":  fmt(mean(opens)),
        "UIRI_peak_mean":  fmt(mean(peaks)),
        "UIRI_mean_mean":  fmt(mean(uiris)),
        "UIRI_delta_mean": fmt(mean(deltas)),
        "UIRI_sigma_mean_within": fmt(mean(sds_within)),
        "UIRI_mean_min":   fmt(min(uiris)),
        "UIRI_mean_max":   fmt(max(uiris)),
        "band_total_pct":  band_dist[ds]["total_band_pct"],
        "band_avg_per_conv_pct": band_dist[ds]["avg_per_conv_band_pct"],
        "avg_per_conv_mean_uiri":  fmt(band_dist[ds]["avg_per_conv_mean_uiri"]),
    }

# Overall (combined HOPE+CareBench)
convs = per_conv_statics
uiris = [r["UIRI_mean"] for r in convs]
opens = [r["UIRI_open"] for r in convs]
peaks = [r["UIRI_peak"] for r in convs]
deltas = [r["UIRI_delta"] for r in convs]
sds_within = [r["UIRI_sd"] for r in convs]
total_bands = {"Low": 0, "Medium": 0, "High": 0}
for r in per_turn:
    if r["band"]:
        total_bands[r["band"]] += 1
n_total = sum(total_bands.values())
summary["overall"] = {
    "n_conversations": len(convs),
    "n_turns_total": sum(r["n_turns"] for r in convs),
    "UIRI_open_mean":  fmt(mean(opens)),
    "UIRI_peak_mean":  fmt(mean(peaks)),
    "UIRI_mean_mean":  fmt(mean(uiris)),
    "UIRI_delta_mean": fmt(mean(deltas)),
    "UIRI_sigma_mean_within": fmt(mean(sds_within)),
    "band_total_pct": {b: round(total_bands[b] / n_total * 100.0, 2) for b in total_bands},
}

with open(OUT_DIR / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote results_summary.json")

# ---------- print headline numbers ----------
print("\n=========== HEADLINE ===========")
for ds in ["HOPE", "CareBench"]:
    s = summary["datasets"][ds]
    print(f"{ds:<10}: {s['n_conversations']} convs / {s['n_turns_total']} turns, "
          f"mean UIRI per conv = {s['UIRI_mean_mean']:.3f}, "
          f"avg band L/M/H = {s['band_avg_per_conv_pct']['Low']:.1f}%/{s['band_avg_per_conv_pct']['Medium']:.1f}%/{s['band_avg_per_conv_pct']['High']:.1f}%")
print(f"Overall   : mean UIRI per conv = {summary['overall']['UIRI_mean_mean']:.3f}, "
      f"total turn-band L/M/H = {summary['overall']['band_total_pct']['Low']:.1f}%/{summary['overall']['band_total_pct']['Medium']:.1f}%/{summary['overall']['band_total_pct']['High']:.1f}%")
