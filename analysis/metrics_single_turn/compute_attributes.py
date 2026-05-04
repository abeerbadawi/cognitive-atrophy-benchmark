"""
Single-turn LLM response attribute analysis.

For each of the 5 LLMs (Response 1=Qwen, 2=Llama, 3=GPT, 4=Claude, 5=Gemini)
and each of the 10 ordinal response attributes
(S, AUR, TD, FIX, RT, TN, QOC, LM, ME, EMP), compute combined-dataset
descriptive statistics: marginal counts (0/1/2), mean, SD, n.

Inputs:  counselchat_human_eval.csv
         pair_human_eval.csv
Outputs: results.json, results_summary.csv

Notes
-----
* Each prompt is rated by exactly one reviewer (Gold = gold_standard for prompts 1-2;
  R1-R6 for blocks of 8). So per (model, attribute) we have one rating per
  prompt — n = 100 codings combined across CounselChat + PAIR.
* S_score and ME_score are binary (0/1); the remaining 8 attributes are
  ordinal 0/1/2.
* RT_score has compound entries (e.g. "1|2") where the rater chose two RT
  categories. These are excluded from the ordinal statistics (n=441 instead
  of 500). The compound count is reported separately.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

SLOT_TO_MODEL = {1: "Qwen", 2: "Llama", 3: "GPT", 4: "Claude", 5: "Gemini"}
ATTRS = ["S", "AUR", "TD", "FIX", "RT", "TN", "QOC", "LM", "ME", "EMP"]
ATTR_LONG = {
    "S":   "Sensitivity (binary)",
    "AUR": "Authentic / Universal Response",
    "TD":  "Tone — Directive",
    "FIX": "Fix-it tendency",
    "RT":  "Response Type",
    "TN":  "Tone (general)",
    "QOC": "Question / Open–Closed style",
    "LM":  "Length / Magnitude",
    "ME":  "Mental-health Education (binary)",
    "EMP": "Empathy",
}
BINARY_ATTRS = {"S", "ME"}


def load_combined() -> pd.DataFrame:
    cc = pd.read_csv(DATA / "counselchat_human_eval.csv")
    pa = pd.read_csv(DATA / "pair_human_eval.csv")
    return pd.concat(
        [cc.assign(dataset="CounselChat"), pa.assign(dataset="PAIR")],
        ignore_index=True,
    )


def attr_stats(values: pd.Series, is_binary: bool) -> dict:
    """Compute marginals + mean/SD on a Series of ordinal codings."""
    coerced = pd.to_numeric(values, errors="coerce")
    n_total = int(values.notna().sum())
    n_compound = int(values.notna().sum() - coerced.notna().sum())
    clean = coerced.dropna()
    n = int(clean.shape[0])

    levels = [0, 1] if is_binary else [0, 1, 2]
    marginals = {str(int(lvl)): int((clean == lvl).sum()) for lvl in levels}

    if n == 0:
        return {
            "n": 0, "n_compound": n_compound, "n_total": n_total,
            "mean": None, "sd": None, "marginals": marginals,
        }
    return {
        "n": n,
        "n_compound": n_compound,
        "n_total": n_total,
        "mean": float(round(clean.mean(), 4)),
        "sd": float(round(clean.std(ddof=1), 4)) if n > 1 else 0.0,
        "marginals": marginals,
    }


def compute() -> dict:
    df = load_combined()

    # prompt_id ranges 1..50 inside *each* dataset, so the global count is rows
    out = {
        "n_prompts_combined": int(len(df)),
        "n_codings_total": int(len(df)),
        "n_prompts_counselchat": int((df["dataset"] == "CounselChat").sum()),
        "n_prompts_pair": int((df["dataset"] == "PAIR").sum()),
        "models": {},
        "by_attribute": {},
    }

    rows_long = []  # for the CSV summary

    # Per-model, per-attribute
    for slot, model in SLOT_TO_MODEL.items():
        model_block = {}
        for a in ATTRS:
            col = f"Response {slot}_{a}_score"
            stats = attr_stats(df[col], is_binary=(a in BINARY_ATTRS))
            model_block[a] = stats
            rows_long.append({
                "model": model,
                "attribute": a,
                "attribute_long": ATTR_LONG[a],
                "is_binary": a in BINARY_ATTRS,
                "n": stats["n"],
                "n_compound": stats["n_compound"],
                "mean": stats["mean"],
                "sd": stats["sd"],
                **{f"count_{k}": v for k, v in stats["marginals"].items()},
            })
        out["models"][model] = model_block

    # Cross-model aggregate per attribute (pooled across the 5 models)
    for a in ATTRS:
        pooled = []
        n_compound_total = 0
        for slot in SLOT_TO_MODEL:
            col = f"Response {slot}_{a}_score"
            coerced = pd.to_numeric(df[col], errors="coerce").dropna()
            pooled.extend(coerced.tolist())
            n_compound_total += int(df[col].notna().sum() - coerced.notna().sum())
        pooled = np.asarray(pooled)
        levels = [0, 1] if a in BINARY_ATTRS else [0, 1, 2]
        out["by_attribute"][a] = {
            "long_name": ATTR_LONG[a],
            "is_binary": a in BINARY_ATTRS,
            "n": int(pooled.size),
            "n_compound": n_compound_total,
            "mean": float(round(pooled.mean(), 4)) if pooled.size else None,
            "sd": float(round(pooled.std(ddof=1), 4)) if pooled.size > 1 else 0.0,
            "marginals": {str(lvl): int((pooled == lvl).sum()) for lvl in levels},
        }

    # Save outputs
    with open(DATA / "descriptive_results.json", "w") as f:
        json.dump(out, f, indent=2)

    summary = pd.DataFrame(rows_long)
    # Stable column ordering
    leading = ["model", "attribute", "attribute_long", "is_binary",
               "n", "n_compound", "mean", "sd"]
    rest = [c for c in summary.columns if c not in leading]
    summary = summary[leading + rest]
    summary.to_csv(DATA / "descriptive_results_summary.csv", index=False)

    return out


if __name__ == "__main__":
    out = compute()
    print(f"n_prompts: {out['n_prompts_combined']}, total rows: {out['n_codings_total']}")
    print("\nPer-attribute pooled means (5 models pooled):")
    for a, blk in out["by_attribute"].items():
        scale = "0/1" if blk["is_binary"] else "0/1/2"
        print(f"  {a:<5} ({scale:<5})  n={blk['n']:>3}  mean={blk['mean']:.3f}  sd={blk['sd']:.3f}  "
              f"counts={blk['marginals']}")
    print("\nPer-model means by attribute:")
    print(f"{'attr':<5} " + "".join(f"{m:>10}" for m in SLOT_TO_MODEL.values()))
    for a in ATTRS:
        cells = [f"{out['models'][m][a]['mean']:>10.3f}" for m in SLOT_TO_MODEL.values()]
        print(f"{a:<5} " + "".join(cells))
