"""
Single-turn correlation analysis — User input × LLM response, per model.

Computes Spearman rho (with two-sided p-value) for every (U_i, R_j) pair on the
combined single-turn data (CounselChat + PAIR), separately for each of the five
evaluated LLMs.  Applies Benjamini-Hochberg FDR correction within each model
(50 tests per model) and reports per-cell results.

Inputs (must sit alongside this script):
  counselchat_human_eval.csv
  pair_human_eval.csv

Outputs:
  results.json   — full numeric results
  results_summary.csv — flat per-cell table (model x U x R x rho x p x q x sig)

Methodology
-----------
Primary:  Spearman rho per (model, U_i, R_j).
P-values: scipy.stats.spearmanr two-sided.
FDR:      Benjamini-Hochberg within each model panel of 50 tests; q = 0.05.
Effect:   |rho| >= 0.20 AND q < 0.05 marks a strong cell.

Slot -> Model mapping
---------------------
  Response 1 -> Qwen
  Response 2 -> Llama
  Response 3 -> GPT
  Response 4 -> Claude
  Response 5 -> Gemini

Reviewer codes (from IRR convention):
  R1 = R1, R2 = R2, R3 = R3, R4 = R4, R5 = R5, R6 = R6,
  Gold = gold_standard.
"""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
FIG  = HERE.parent / "figures"

DATASETS = {
    "CounselChat": DATA / "counselchat_human_eval.csv",
    "PAIR":        DATA / "pair_human_eval.csv",
}

USER_ATTRS = ["U1", "U2", "U3", "U4", "U5"]
USER_COLS  = {
    "U1": "user_typicality",
    "U2": "user_evocative",
    "U3": "user_sensitivity",
    "U4": "user_request_info",
    "U5": "user_underlying",
}

# Response ordinals as defined in the framework.
RESP_ATTRS = ["SEN", "AUR", "TD", "FIX", "RT", "TN", "QOC", "LM", "ME", "EMP"]
RESP_COL_SUFFIX = {
    "SEN": "_S_score",
    "AUR": "_AUR_score",
    "TD":  "_TD_score",
    "FIX": "_FIX_score",
    "RT":  "_RT_score",
    "TN":  "_TN_score",
    "QOC": "_QOC_score",
    "LM":  "_LM_score",
    "ME":  "_ME_score",
    "EMP": "_EMP_score",
}

SLOT_TO_MODEL = {
    1: "Qwen",
    2: "Llama",
    3: "GPT",
    4: "Claude",
    5: "Gemini",
}
MODEL_ORDER = ["GPT", "Claude", "Gemini", "Llama", "Qwen"]


def parse_int(value):
    s = (value or "").strip()
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_all_rows():
    rows = []
    for ds_name, path in DATASETS.items():
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                # User-side attributes
                user = {ua: parse_int(r[USER_COLS[ua]]) for ua in USER_ATTRS}
                # Response-side per slot
                for slot in range(1, 6):
                    resp = {ra: parse_int(r[f"Response {slot}{RESP_COL_SUFFIX[ra]}"]) for ra in RESP_ATTRS}
                    rows.append({
                        "dataset":  ds_name,
                        "reviewer": r["reviewer"],
                        "prompt_id": int(float(r["prompt_id"])),
                        "slot":     slot,
                        "model":    SLOT_TO_MODEL[slot],
                        **{f"u_{k}": v for k, v in user.items()},
                        **{f"r_{k}": v for k, v in resp.items()},
                    })
    return rows


def benjamini_hochberg(pvals, q=0.05):
    """Return (qvals, reject) arrays. qvals are the BH-adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    # BH-adjusted p-values: q_i = min over j>=i of (n / j) * p_j
    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = ranked[i] * n / rank
        cumulative_min = min(cumulative_min, adj)
        adjusted[i] = cumulative_min
    qvals = np.empty(n, dtype=float)
    qvals[order] = adjusted
    reject = qvals < q
    return qvals, reject


def main():
    rows = load_all_rows()
    print(f"Loaded {len(rows)} (prompt, model) pairs (= {len(rows)//5} prompts × 5 models)")
    n_per_model = len(rows) // 5

    # Per-model correlation matrix
    per_model = {}
    flat = []  # for CSV summary
    for model in MODEL_ORDER:
        model_rows = [x for x in rows if x["model"] == model]
        cells = []
        for ui, uname in enumerate(USER_ATTRS):
            for rj, rname in enumerate(RESP_ATTRS):
                u_arr = np.array([x[f"u_{uname}"] for x in model_rows], dtype=float)
                r_arr = np.array([x[f"r_{rname}"] for x in model_rows], dtype=float)
                # Drop NaNs pairwise
                mask = ~(np.isnan(u_arr) | np.isnan(r_arr))
                if mask.sum() < 5 or np.std(u_arr[mask]) == 0 or np.std(r_arr[mask]) == 0:
                    cells.append({"U": uname, "R": rname, "n": int(mask.sum()),
                                  "rho": None, "p": None})
                    continue
                res = spearmanr(u_arr[mask], r_arr[mask])
                cells.append({"U": uname, "R": rname, "n": int(mask.sum()),
                              "rho": float(res.correlation), "p": float(res.pvalue)})

        # BH-FDR within model on cells with p defined
        valid_idx = [i for i, c in enumerate(cells) if c["p"] is not None]
        pvals = [cells[i]["p"] for i in valid_idx]
        qvals, reject = benjamini_hochberg(pvals, q=0.05)
        for k, i in enumerate(valid_idx):
            cells[i]["q"] = float(qvals[k])
            cells[i]["sig"] = bool(reject[k] and abs(cells[i]["rho"]) >= 0.20)

        per_model[model] = cells
        for c in cells:
            flat.append({
                "model": model,
                "U": c["U"], "R": c["R"],
                "n": c["n"],
                "rho": c.get("rho"),
                "p":   c.get("p"),
                "q":   c.get("q"),
                "sig": c.get("sig"),
            })

    # Save outputs
    with open(DATA / "results.json", "w") as f:
        json.dump({"n_per_model": n_per_model, "models": per_model}, f, indent=2)

    with open(DATA / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","U","R","n","rho","p","q","sig"])
        w.writeheader()
        for r in flat:
            w.writerow(r)

    # ---- pretty print ----
    print(f"\n=== Spearman rho per model · n = {n_per_model} responses each ===")
    print(f"FDR threshold q = 0.05; cells marked '*' meet |rho| >= 0.20 AND q < 0.05.\n")
    header = f"{'U':<4}|" + "".join(f"{r:>9}" for r in RESP_ATTRS)
    for model in MODEL_ORDER:
        cells = per_model[model]
        # nest cells in 5x10 grid
        grid = {(c["U"], c["R"]): c for c in cells}
        print(f"--- {model} ---")
        print(header)
        for ui, uname in enumerate(USER_ATTRS):
            line = f"{uname:<4}|"
            for rj, rname in enumerate(RESP_ATTRS):
                c = grid[(uname, rname)]
                if c["rho"] is None:
                    line += f"{'   --   ':>9}"
                else:
                    star = "*" if c.get("sig") else " "
                    line += f"{c['rho']:>+7.2f}{star} "
            print(line)
        print()

    print(f"\nWrote results.json and results_summary.csv to {DATA}\n")


if __name__ == "__main__":
    main()
