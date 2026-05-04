#!/usr/bin/env python3
"""
csv_to_json.py — convert a CSV of LLM responses into the JSON shape that the
                  annotation UI expects.

The single-turn evaluator (single_turn_evaluator.html) consumes:
    [
      {
        "questionID": "...",
        "prompt": "...",
        "Qwen Output": "...",
        "Llama Output": "...",
        "GPT Output": "...",
        "Claude Output": "...",
        "Gemini Output": "...",
        ...any other columns are passed through and ignored by the UI
      },
      ...
    ]

The multi-turn evaluator (multi_turn_evaluator.html) consumes:
    [
      {
        "filename": "<conversation id>",
        "turns": [ <one row per turn, same shape as a single-turn row>, ... ]
      },
      ...
    ]

Usage
-----
Single-turn (one CSV → one JSON):
    python csv_to_json.py single \\
        --input  my_responses.csv \\
        --output my_responses.json

Multi-turn (one CSV with a Conversation column that groups rows):
    python csv_to_json.py multi \\
        --input  my_conversations.csv \\
        --output my_conversations.json \\
        --group-by Conversation        # column that identifies each conversation

Multi-turn (one folder of per-conversation CSVs):
    python csv_to_json.py multi-folder \\
        --input-dir  my_conversations/ \\
        --output     my_conversations.json

The script pretty-prints the JSON. The UI loads it via the "Load Dataset"
button at the top of the page, or from the empty-state screen on first launch.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("This helper needs pandas. Install with:  pip install pandas")


def csv_rows(path: Path) -> list[dict]:
    """Read a CSV and return a list of plain dicts (NaN -> empty string)."""
    df = pd.read_csv(path)
    df = df.where(df.notnull(), "")
    return df.to_dict(orient="records")


def cmd_single(args: argparse.Namespace) -> None:
    rows = csv_rows(Path(args.input))
    Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Wrote {len(rows)} prompts → {args.output}")


def cmd_multi(args: argparse.Namespace) -> None:
    rows = csv_rows(Path(args.input))
    if args.group_by not in (rows[0] if rows else {}):
        sys.exit(f"--group-by column '{args.group_by}' not found in CSV header")

    by_conv: dict[str, list[dict]] = {}
    for r in rows:
        key = str(r[args.group_by])
        by_conv.setdefault(key, []).append(r)

    out = [{"filename": name, "turns": turns} for name, turns in by_conv.items()]
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} conversations ({sum(len(c['turns']) for c in out)} turns) → {args.output}")


def cmd_multi_folder(args: argparse.Namespace) -> None:
    in_dir = Path(args.input_dir)
    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"No .csv files found in {in_dir}")

    out = []
    for csv_path in csvs:
        turns = csv_rows(csv_path)
        out.append({"filename": csv_path.stem, "turns": turns})

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} conversations ({sum(len(c['turns']) for c in out)} turns) → {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="single-turn CSV → JSON")
    p_single.add_argument("--input",  required=True, help="path to single-turn CSV")
    p_single.add_argument("--output", required=True, help="path to write JSON")
    p_single.set_defaults(func=cmd_single)

    p_multi = sub.add_parser("multi", help="multi-turn CSV (one file, grouped by column) → JSON")
    p_multi.add_argument("--input",    required=True, help="path to multi-turn CSV")
    p_multi.add_argument("--output",   required=True, help="path to write JSON")
    p_multi.add_argument("--group-by", default="Conversation",
                         help="column that identifies each conversation (default: Conversation)")
    p_multi.set_defaults(func=cmd_multi)

    p_folder = sub.add_parser("multi-folder", help="folder of per-conversation CSVs → JSON")
    p_folder.add_argument("--input-dir", required=True, help="folder containing one CSV per conversation")
    p_folder.add_argument("--output",    required=True, help="path to write JSON")
    p_folder.set_defaults(func=cmd_multi_folder)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
