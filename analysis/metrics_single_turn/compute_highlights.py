"""
Single-turn LLM highlight analysis.

For each of the 5 LLMs (Response 1=Qwen, 2=Llama, 3=GPT, 4=Claude, 5=Gemini)
and each of the 20 highlight categories
(SEN, AUR, TEN, DIR, FIX, RECT, TSH, QOP, QCL, LMT, MEN,
 VIN, NIN, ASIN, SIN, VAC, NAC, ASAC, SAC, INC),
compute three per-response metrics:

  1. Span count   — number of pipe-separated text spans the rater extracted
  2. Token count  — total tokens (cl100k_base) inside those spans
  3. Tokens/span  — derived: token count / span count (length of each excerpt)

The 20 highlights are then rolled up under 10 attributes (see ATTR_GROUPS),
plus INC kept as a separate "21st" category for incoherence.

Outputs (in ../data/):
  - per_response_long.csv       row-level long-format with all metrics
  - per_model_summary.csv       per (model, metric, attribute) means + SDs
  - per_model_highlights.csv    per (model, metric, individual highlight) means
  - distinctiveness_zscores.csv z-scores of model means against pooled mean
  - within_attr_asymmetry.csv   TEN/DIR, QOP/QCL, accurate/inaccurate ratios
  - tokens_per_span.csv         derived intensity-per-instance per model × attr
  - results.json                full nested results dictionary

Inputs (in ../data/):
  counselchat_human_eval.csv
  pair_human_eval.csv
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import tiktoken

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SLOT_TO_MODEL = {1: "Qwen", 2: "Llama", 3: "GPT", 4: "Claude", 5: "Gemini"}
MODELS = list(SLOT_TO_MODEL.values())

HIGHLIGHTS = [
    "SEN", "AUR", "TEN", "DIR", "FIX", "RECT", "TSH",
    "QOP", "QCL", "LMT", "MEN",
    "VIN", "NIN", "ASIN", "SIN",
    "VAC", "NAC", "ASAC", "SAC",
    "INC",
]

# 20 highlights -> 10 attributes (+ INC kept separate)
ATTR_GROUPS = {
    "S":   ["SEN"],
    "AUR": ["AUR"],
    "TD":  ["TEN", "DIR"],          # Tentativeness / Directness
    "FIX": ["FIX"],
    "RT":  ["RECT"],
    "TN":  ["TSH"],                 # Topic shift
    "QOC": ["QOP", "QCL"],          # Open / Closed questions
    "LM":  ["LMT"],
    "ME":  ["MEN"],
    "EMP": ["VIN", "NIN", "ASIN", "SIN", "VAC", "NAC", "ASAC", "SAC"],
}
ATTRS = list(ATTR_GROUPS.keys())  # 10 ordered attribute keys

ATTR_LONG = {
    "S":   "Sensitivity",
    "AUR": "Assumption of User Accuracy",
    "TD":  "Tentativeness / Directness",
    "FIX": "Fix-It Tendency",
    "RT":  "Recommendation Type",
    "TN":  "Topic Shift",
    "QOC": "Open / Closed Questions",
    "LM":  "Language Matching",
    "ME":  "Minimal Encouragers",
    "EMP": "Empathy & Matching",
}

# Within-attribute asymmetry decompositions
ASYMMETRY = {
    "TD":  ("TEN", "DIR"),                                    # tentative vs directive
    "QOC": ("QOP", "QCL"),                                    # open vs closed
    # EMP gets a special compound: accurate (4) vs inaccurate (4)
}
EMP_ACCURATE   = ["VAC", "NAC", "ASAC", "SAC"]
EMP_INACCURATE = ["VIN", "NIN", "ASIN", "SIN"]

ENC = tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------- #
# Cell-level metrics
# --------------------------------------------------------------------------- #
def _split_spans(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def span_count(cell) -> int:
    return len(_split_spans(cell))


def token_count(cell) -> int:
    return sum(len(ENC.encode(p)) for p in _split_spans(cell))


def word_count(cell) -> int:
    return sum(len(p.split()) for p in _split_spans(cell))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_combined() -> pd.DataFrame:
    cc = pd.read_csv(DATA / "counselchat_human_eval.csv")
    pa = pd.read_csv(DATA / "pair_human_eval.csv")
    return pd.concat(
        [cc.assign(dataset="CounselChat"), pa.assign(dataset="PAIR")],
        ignore_index=True,
    )


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-format frame keyed by (dataset, reviewer, prompt_id, model)
    with per-highlight span / token / word counts."""
    pieces = []
    for slot, model in SLOT_TO_MODEL.items():
        block = pd.DataFrame({
            "dataset": df["dataset"],
            "reviewer": df["reviewer"],
            "prompt_id": df["prompt_id"],
            "model": model,
        })
        for h in HIGHLIGHTS:
            col = f"Response {slot}_{h}"
            block[f"n_{h}"]    = df[col].apply(span_count)
            block[f"tok_{h}"]  = df[col].apply(token_count)
            block[f"word_{h}"] = df[col].apply(word_count)
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def aggregate_to_attribute(long: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Sum highlight columns within each attribute group on a per-row basis."""
    out = pd.DataFrame({
        "dataset": long["dataset"],
        "reviewer": long["reviewer"],
        "prompt_id": long["prompt_id"],
        "model": long["model"],
    })
    for a, hls in ATTR_GROUPS.items():
        out[a] = long[[f"{prefix}_{h}" for h in hls]].sum(axis=1)
    out["INC"] = long[f"{prefix}_INC"]
    return out


def per_model_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Per (model, metric, attribute): mean & SD across responses."""
    rows = []
    for prefix, metric in [("n", "span_count"), ("tok", "token_count"), ("word", "word_count")]:
        attr_long = aggregate_to_attribute(long, prefix)
        for model in MODELS:
            sub = attr_long.loc[attr_long.model == model]
            for a in ATTRS + ["INC"]:
                vals = sub[a]
                rows.append({
                    "model": model,
                    "metric": metric,
                    "attribute": a,
                    "attribute_long": ATTR_LONG.get(a, "Incoherence"),
                    "n_responses": int(len(vals)),
                    "mean": float(round(vals.mean(), 4)),
                    "sd": float(round(vals.std(ddof=1), 4)) if len(vals) > 1 else 0.0,
                    "sum": int(vals.sum()),
                    "n_nonzero": int((vals > 0).sum()),
                })
    return pd.DataFrame(rows)


def per_model_per_highlight(long: pd.DataFrame) -> pd.DataFrame:
    """Per (model, metric, individual highlight): mean across responses."""
    rows = []
    for prefix, metric in [("n", "span_count"), ("tok", "token_count"), ("word", "word_count")]:
        for model in MODELS:
            sub = long.loc[long.model == model]
            for h in HIGHLIGHTS:
                vals = sub[f"{prefix}_{h}"]
                rows.append({
                    "model": model,
                    "metric": metric,
                    "highlight": h,
                    # Map to its attribute
                    "attribute": next((a for a, hls in ATTR_GROUPS.items() if h in hls), "INC"),
                    "n_responses": int(len(vals)),
                    "mean": float(round(vals.mean(), 4)),
                    "sd": float(round(vals.std(ddof=1), 4)) if len(vals) > 1 else 0.0,
                    "sum": int(vals.sum()),
                })
    return pd.DataFrame(rows)


def distinctiveness_zscores(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (metric, attribute), z-score each model's mean against the
    cross-model distribution of means (z = (mean_m - mean_pooled) / sd_pooled).
    This identifies which model is the *outlier* on each attribute."""
    rows = []
    for metric in summary["metric"].unique():
        for a in ATTRS + ["INC"]:
            block = summary[(summary.metric == metric) & (summary.attribute == a)]
            mu = block["mean"].mean()
            sigma = block["mean"].std(ddof=1)
            for _, r in block.iterrows():
                z = (r["mean"] - mu) / sigma if sigma and sigma > 0 else 0.0
                rows.append({
                    "metric": metric,
                    "attribute": a,
                    "attribute_long": r["attribute_long"],
                    "model": r["model"],
                    "model_mean": r["mean"],
                    "cross_model_mean": float(round(mu, 4)),
                    "cross_model_sd": float(round(sigma, 4)) if sigma else 0.0,
                    "z": float(round(z, 4)),
                })
    return pd.DataFrame(rows)


def within_attribute_asymmetry(long: pd.DataFrame) -> pd.DataFrame:
    """For TD (TEN vs DIR), QOC (QOP vs QCL), EMP (accurate vs inaccurate),
    compute the per-response ratio averaged within model, on both span count
    and token count.  Reported as: mean of pole A, mean of pole B, ratio A/B,
    proportion A/(A+B), n_responses_with_any."""
    rows = []
    for prefix, metric in [("n", "span_count"), ("tok", "token_count")]:
        for model in MODELS:
            sub = long.loc[long.model == model]
            # TD
            ten = sub[f"{prefix}_TEN"]; dir_ = sub[f"{prefix}_DIR"]
            rows.append({
                "metric": metric, "model": model, "attribute": "TD",
                "pole_a": "TEN (tentative)", "pole_b": "DIR (directive)",
                "mean_a": float(round(ten.mean(), 4)),
                "mean_b": float(round(dir_.mean(), 4)),
                "ratio_a_over_b": float(round(ten.sum() / dir_.sum(), 4)) if dir_.sum() else None,
                "share_a": float(round(ten.sum() / (ten.sum() + dir_.sum()), 4)) if (ten.sum() + dir_.sum()) else None,
                "n_responses_with_any": int(((ten + dir_) > 0).sum()),
            })
            # QOC
            qop = sub[f"{prefix}_QOP"]; qcl = sub[f"{prefix}_QCL"]
            rows.append({
                "metric": metric, "model": model, "attribute": "QOC",
                "pole_a": "QOP (open)", "pole_b": "QCL (closed)",
                "mean_a": float(round(qop.mean(), 4)),
                "mean_b": float(round(qcl.mean(), 4)),
                "ratio_a_over_b": float(round(qop.sum() / qcl.sum(), 4)) if qcl.sum() else None,
                "share_a": float(round(qop.sum() / (qop.sum() + qcl.sum()), 4)) if (qop.sum() + qcl.sum()) else None,
                "n_responses_with_any": int(((qop + qcl) > 0).sum()),
            })
            # EMP: accurate vs inaccurate
            acc = sub[[f"{prefix}_{h}" for h in EMP_ACCURATE]].sum(axis=1)
            ina = sub[[f"{prefix}_{h}" for h in EMP_INACCURATE]].sum(axis=1)
            rows.append({
                "metric": metric, "model": model, "attribute": "EMP",
                "pole_a": "accurate (VAC+NAC+ASAC+SAC)",
                "pole_b": "inaccurate (VIN+NIN+ASIN+SIN)",
                "mean_a": float(round(acc.mean(), 4)),
                "mean_b": float(round(ina.mean(), 4)),
                "ratio_a_over_b": float(round(acc.sum() / ina.sum(), 4)) if ina.sum() else None,
                "share_a": float(round(acc.sum() / (acc.sum() + ina.sum()), 4)) if (acc.sum() + ina.sum()) else None,
                "n_responses_with_any": int(((acc + ina) > 0).sum()),
            })
    return pd.DataFrame(rows)


def attribute_score_means(df_combined: pd.DataFrame) -> pd.DataFrame:
    """Per (model, attribute) mean and SD of the ordinal `_X_score` column.

    The CSV's per-attribute ordinal scores live in columns named
    ``Response {slot}_{attr}_score``. RT_score has compound entries (e.g.
    "1|2") that are not numeric — these are dropped (coerced to NaN) for the
    mean.

    Returns a long-format frame with one row per (model, attribute).
    """
    rows = []
    for slot, model in SLOT_TO_MODEL.items():
        for attr in ATTRS:
            col = f"Response {slot}_{attr}_score"
            if col not in df_combined.columns:
                continue
            s = pd.to_numeric(df_combined[col], errors="coerce")
            n = int(s.notna().sum())
            rows.append({
                "model": model,
                "attribute": attr,
                "attribute_long": ATTR_LONG.get(attr, attr),
                "n": n,
                "mean_score": float(round(s.mean(), 4)) if n else None,
                "sd_score":   float(round(s.std(ddof=1), 4)) if n > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def attribute_cofiring(long: pd.DataFrame) -> pd.DataFrame:
    """Per-model 10x10 attribute co-firing matrix.

    For each model, take the per-response span counts at the *attribute* level
    (10 attributes; INC excluded) and compute Spearman rank correlations
    between every attribute pair. ρ measures whether highlight categories
    tend to co-fire on the same response — a behavioural-architecture signal.
    """
    rows = []
    # First, build per-row attribute-level span counts inside `long`'s schema
    attr_long = aggregate_to_attribute(long, "n")[
        ["dataset", "reviewer", "prompt_id", "model"] + ATTRS
    ]
    for model in MODELS:
        sub = attr_long.loc[attr_long.model == model][ATTRS]
        rho = sub.corr(method="spearman")
        for a1 in ATTRS:
            for a2 in ATTRS:
                rows.append({
                    "model": model,
                    "attribute_a": a1,
                    "attribute_b": a2,
                    "rho": float(rho.loc[a1, a2]) if pd.notna(rho.loc[a1, a2]) else None,
                })
    return pd.DataFrame(rows)


def cofiring_top_pairs(cofiring: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Top-N strongest co-firing pairs per model (excluding self-correlations)."""
    rows = []
    for model in MODELS:
        sub = cofiring[(cofiring.model == model) &
                       (cofiring.attribute_a < cofiring.attribute_b)]
        sub = sub.assign(abs_rho=sub["rho"].abs()).sort_values("abs_rho", ascending=False)
        for i, (_, r) in enumerate(sub.head(top_n).iterrows()):
            rows.append({
                "model": model,
                "rank": i + 1,
                "pair": f"{r['attribute_a']}~{r['attribute_b']}",
                "rho": float(round(r["rho"], 4)) if pd.notna(r["rho"]) else None,
            })
    return pd.DataFrame(rows)


def tokens_per_span_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Derive tokens-per-span from (model, attribute) totals."""
    span = summary[summary.metric == "span_count"].set_index(["model", "attribute"])["sum"]
    tok  = summary[summary.metric == "token_count"].set_index(["model", "attribute"])["sum"]
    rows = []
    for (model, attr), s in span.items():
        t = tok.loc[(model, attr)]
        rows.append({
            "model": model,
            "attribute": attr,
            "attribute_long": ATTR_LONG.get(attr, "Incoherence"),
            "total_spans": int(s),
            "total_tokens": int(t),
            "tokens_per_span": float(round(t / s, 4)) if s else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    df = load_combined()
    print(f"Loaded combined data: {len(df)} rows "
          f"({(df.dataset=='CounselChat').sum()} CC + {(df.dataset=='PAIR').sum()} PAIR)")
    print(f"Reviewers: {sorted(df.reviewer.unique().tolist())}")

    long = build_long(df)
    summary = per_model_summary(long)
    per_hl  = per_model_per_highlight(long)
    z_table = distinctiveness_zscores(summary)
    asym    = within_attribute_asymmetry(long)
    tps     = tokens_per_span_table(summary)
    cofiring = attribute_cofiring(long)
    cofiring_top = cofiring_top_pairs(cofiring, top_n=5)
    score_means = attribute_score_means(df)

    long.to_csv(DATA / "per_response_long.csv", index=False)
    summary.to_csv(DATA / "per_model_summary.csv", index=False)
    per_hl.to_csv(DATA / "per_model_highlights.csv", index=False)
    z_table.to_csv(DATA / "distinctiveness_zscores.csv", index=False)
    asym.to_csv(DATA / "within_attr_asymmetry.csv", index=False)
    tps.to_csv(DATA / "tokens_per_span.csv", index=False)
    cofiring.to_csv(DATA / "cofiring_matrices.csv", index=False)
    cofiring_top.to_csv(DATA / "cofiring_top_pairs.csv", index=False)
    score_means.to_csv(DATA / "attribute_score_means.csv", index=False)

    # JSON: nested results
    nested = {
        "n_responses_per_model": int(len(long) // len(MODELS)),
        "models": MODELS,
        "attributes": ATTRS,
        "highlights_per_attribute": ATTR_GROUPS,
        "summary": json.loads(summary.to_json(orient="records")),
        "per_highlight": json.loads(per_hl.to_json(orient="records")),
        "distinctiveness_zscores": json.loads(z_table.to_json(orient="records")),
        "within_attribute_asymmetry": json.loads(asym.to_json(orient="records")),
        "tokens_per_span": json.loads(tps.to_json(orient="records")),
        "cofiring_matrices": json.loads(cofiring.to_json(orient="records")),
        "cofiring_top_pairs": json.loads(cofiring_top.to_json(orient="records")),
        "attribute_score_means": json.loads(score_means.to_json(orient="records")),
    }
    with open(DATA / "results.json", "w") as f:
        json.dump(nested, f, indent=2)

    # Pretty-print main tables
    print("\n=== Span count means (per attribute × model) ===")
    pivot_n = (summary[summary.metric == "span_count"]
               .pivot(index="attribute", columns="model", values="mean")
               .reindex(ATTRS + ["INC"]))
    print(pivot_n[MODELS].round(2))

    print("\n=== Token count means (per attribute × model) ===")
    pivot_t = (summary[summary.metric == "token_count"]
               .pivot(index="attribute", columns="model", values="mean")
               .reindex(ATTRS + ["INC"]))
    print(pivot_t[MODELS].round(1))

    print("\n=== Tokens-per-span (intensity per instance) ===")
    pivot_ps = tps.pivot(index="attribute", columns="model", values="tokens_per_span").reindex(ATTRS + ["INC"])
    print(pivot_ps[MODELS].round(1))

    print("\n=== Top co-firing attribute pairs per model (Spearman ρ on span counts) ===")
    for m in MODELS:
        sub = cofiring_top[cofiring_top.model == m]
        line = ", ".join(f"{r['pair']}={r['rho']:+.2f}" for _, r in sub.iterrows())
        print(f"  {m:<7} {line}")

    print("\n=== Top distinctive (z >= 1.5) cells ===")
    standout = z_table[z_table["z"].abs() >= 1.5].sort_values(
        ["metric", "attribute", "z"], ascending=[True, True, False]
    )
    for _, r in standout.iterrows():
        sign = "+" if r["z"] > 0 else "−"
        print(f"  [{r['metric']:<11}] {r['model']:<7} {r['attribute']:<5} "
              f"mean={r['model_mean']:.2f}  pooled={r['cross_model_mean']:.2f}  z={sign}{abs(r['z']):.2f}")

    print("\nAll outputs written to:", DATA)


if __name__ == "__main__":
    main()
