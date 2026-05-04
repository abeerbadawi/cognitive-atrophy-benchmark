"""
5-Model AI Response Generator — HOPE Dataset
─────────────────────────────────────────────
Models  : Qwen3.5-35B (vec-inf), Llama-4-Maverick (HF), GPT-5.3, Claude Sonnet, Gemini 3 Flash
Input   : HOPE xlsx files (Type=P patient turns, Type=T therapist turns)
Strategy: Option A — AI responds to each patient turn using its own previous
          responses as history (no original therapist responses used)
Output  : one CSV per conversation + all_results.csv
"""

import os
import re
import time
import glob
import pandas as pd
from openai import OpenAI
import anthropic
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  ←  only section you need to edit
# ─────────────────────────────────────────────────────────────────

# --- Qwen (vec-inf cluster) ---
QWEN_API_KEY  = "EMPTY"
QWEN_BASE_URL = "ADD_YOUR_QWEN_BASE_URL_HERE"        # ← update port from vec-inf status
QWEN_MODEL_ID = "Qwen3.5-35B-A3B-FP8"    # exact name from vec-inf

# --- Llama / MetaAI (Hugging Face Inference Providers) ---
LLAMA_API_KEY  = os.environ.get("HF_TOKEN", "ADD_YOUR_HF_TOKEN_HERE")
LLAMA_BASE_URL = "https://router.huggingface.co/v1"
LLAMA_MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# --- ChatGPT (OpenAI API) ---
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "ADD_YOUR_OPENAI_API_KEY_HERE")
OPENAI_MODEL_ID = "gpt-5.3-chat-latest"

# --- Claude (Anthropic API) ---
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "ADD_YOUR_ANTHROPIC_API_KEY_HERE")
ANTHROPIC_MODEL_ID = "claude-sonnet-4-6"

# --- Gemini (Google API) ---
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "ADD_YOUR_GEMINI_API_KEY_HERE")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_ID = "gemini-3-flash-preview"


# --- Folders ---
INPUT_DIR  = "ADD_YOUR_INPUT_DIR_HERE"
OUTPUT_DIR = "ADD_YOUR_OUTPUT_DIR_HERE"

# --- Shared params ---
MAX_TOKENS        = 2048
GPT_MAX_TOKENS    = 2048
MAX_WORDS         = 300
MAX_HISTORY_TURNS = 10
TEMPERATURE       = 1.0
TOP_P             = 1.0

# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Read the user's message carefully and provide a response."
)

# ─────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────
def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*",          "", text, flags=re.DOTALL)
    return text.strip()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\u2026": "...",
        "\u00e2\u0080\u0099": "'",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

def trim_response(text: str) -> str:
    if not isinstance(text, str) or text.startswith("[ERROR]") or text.startswith("[EMPTY]"):
        return text
    words = text.split()
    if len(words) <= MAX_WORDS:
        return text
    trimmed = " ".join(words[:MAX_WORDS])
    for punct in [".", "!", "?"]:
        last = trimmed.rfind(punct)
        if last > len(trimmed) * 0.6:
            return trimmed[:last + 1]
    return trimmed

# ─────────────────────────────────────────────────────────────────
# LOAD ONE HOPE CONVERSATION FILE
# Extracts only patient (P) turns as inputs
# Stores original therapist (T) turns for reference only
# ─────────────────────────────────────────────────────────────────
def load_conversation(filepath: str) -> list[dict]:
    df   = pd.read_excel(filepath)
    rows = df[df["Type"].str.strip().isin(["P", "T"])].reset_index(drop=True)
    turns = []
    for i, row in rows.iterrows():
        if str(row["Type"]).strip() == "P":
            # original therapist = what came after this patient turn
            orig_t = ""
            if i + 1 < len(rows) and str(rows.iloc[i+1]["Type"]).strip() == "T":
                orig_t = str(rows.iloc[i+1]["Utterance"]).strip()
            turns.append({
                "patient"       : str(row["Utterance"]).strip(),
                "orig_therapist": orig_t,
            })
    return turns

# ─────────────────────────────────────────────────────────────────
# BUILD CONVERSATION HISTORY AS MESSAGES
# Option A: history uses AI's own previous responses (same as CareBench)
# ─────────────────────────────────────────────────────────────────
def build_messages(history: list[dict], user_input: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_input})
    return messages

def build_messages_claude(history: list[dict], user_input: str) -> list[dict]:
    messages = []
    for h in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_input})
    return messages

# ─────────────────────────────────────────────────────────────────
# MODEL CALL FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def call_qwen(client, user_input, history):
    try:
        r = client.chat.completions.create(
            model       = QWEN_MODEL_ID,
            messages    = build_messages(history, user_input),
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            top_p       = TOP_P,
            extra_body  = {"chat_template_kwargs": {"enable_thinking": False}},
        )
        return trim_response(clean_text(strip_thinking(r.choices[0].message.content)))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_llama(client, user_input, history):
    try:
        r = client.chat.completions.create(
            model       = LLAMA_MODEL_ID,
            messages    = build_messages(history, user_input),
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            top_p       = TOP_P,
        )
        return trim_response(clean_text(r.choices[0].message.content.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_openai(client, user_input, history):
    try:
        r = client.chat.completions.create(
            model                 = OPENAI_MODEL_ID,
            messages              = build_messages(history, user_input),
            max_completion_tokens = GPT_MAX_TOKENS,
            temperature           = TEMPERATURE,
            top_p                 = TOP_P,
        )
        content       = r.choices[0].message.content
        finish_reason = r.choices[0].finish_reason
        if not content or not content.strip():
            return f"[EMPTY] finish_reason={finish_reason}"
        return trim_response(clean_text(content.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_claude(client, user_input, history):
    try:
        r = client.messages.create(
            model      = ANTHROPIC_MODEL_ID,
            max_tokens = MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = build_messages_claude(history, user_input),
        )
        return trim_response(clean_text(r.content[0].text.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_gemini(client, user_input, history):
    try:
        r = client.chat.completions.create(
            model       = GEMINI_MODEL_ID,
            messages    = build_messages(history, user_input),
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            top_p       = TOP_P,
        )
        return trim_response(clean_text(r.choices[0].message.content.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

# ─────────────────────────────────────────────────────────────────
# PROCESS ONE CONVERSATION
# ─────────────────────────────────────────────────────────────────
def process_conversation(conv_idx, filepath, clients, model_config):
    conv_name  = os.path.splitext(os.path.basename(filepath))[0]
    output_csv = os.path.join(OUTPUT_DIR, f"{conv_name}_results.csv")

    # Skip if already done
    if os.path.exists(output_csv):
        print(f"  ⏭  {conv_name} already done — skipping")
        return pd.read_csv(output_csv)

    print(f"\n{'='*65}")
    print(f"  Conversation {conv_idx}: {conv_name}")
    print(f"{'='*65}")

    turns = load_conversation(filepath)
    if not turns:
        print(f"  ⚠️  No patient turns found — skipping")
        return pd.DataFrame()

    histories = {"qwen": [], "llama": [], "openai": [], "claude": [], "gemini": []}
    results   = []
    conv_start = time.time()

    for turn_idx, turn in enumerate(turns, start=1):
        patient  = turn["patient"]
        orig_t   = turn["orig_therapist"]
        turn_start = time.time()

        print(f"  Turn {turn_idx:02d}/{len(turns)} | {time.strftime('%H:%M:%S')} | {patient[:60].replace(chr(10),' ')}...")

        row = {
            "Conversation"      : conv_name,
            "Turn"              : turn_idx,
            "Patient Input"     : patient,
            "Original Therapist": orig_t,
        }

        # Call all 5 models in parallel
        def call_model(args):
            model_name, call_fn, client, hkey = args
            response = call_fn(client, patient, histories[hkey])
            return model_name, hkey, response

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = list(executor.map(call_model, model_config))

        for model_name, hkey, response in futures:
            # History uses patient input + AI's own response (Option A)
            histories[hkey].append({
                "turn"     : turn_idx,
                "user"     : patient,
                "assistant": response,
            })
            row[f"{model_name} Output"] = response

        turn_elapsed         = round(time.time() - turn_start, 2)
        row["Turn Time (s)"] = turn_elapsed
        row["History Depth"] = turn_idx
        results.append(row)

    # Save this conversation's CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    conv_elapsed = round(time.time() - conv_start, 2)
    print(f"  ✅ {conv_name} done — {len(turns)} turns in {conv_elapsed}s → {output_csv}")
    return df

# ─────────────────────────────────────────────────────────────────
# VALIDATE API KEYS
# ─────────────────────────────────────────────────────────────────
def validate_keys():
    missing = []
    if LLAMA_API_KEY     == "ADD_YOUR_HF_TOKEN_HERE":      missing.append("HF_TOKEN")
    if OPENAI_API_KEY    == "ADD_YOUR_OPENAI_API_KEY_HERE":    missing.append("OPENAI_API_KEY")
    if ANTHROPIC_API_KEY == "ADD_YOUR_ANTHROPIC_API_KEY_HERE": missing.append("ANTHROPIC_API_KEY")
    if GEMINI_API_KEY    == "ADD_YOUR_GEMINI_API_KEY_HERE":    missing.append("GEMINI_API_KEY")
    if missing:
        print("❌ Missing API keys:")
        for m in missing:
            print(f"   - {m}")
        print("\nSet them with:")
        print("   export HF_TOKEN='hf_...'")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print("   export GEMINI_API_KEY='...'")
        return False
    return True

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  5-Model Batch Generator  |  HOPE Dataset")
    print("  Qwen | Llama | GPT-5.3 | Claude Sonnet | Gemini-2.5-Flash")
    print(f"  max_tokens={MAX_TOKENS}, max_words={MAX_WORDS}, history={MAX_HISTORY_TURNS} turns")
    print("="*65 + "\n")

    if not validate_keys():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all xlsx files sorted numerically by the number in filename
    files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.xlsx")),
        key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
    )

    if not files:
        print(f"❌ No xlsx files found in {INPUT_DIR}")
        return

    print(f"📂 Found {len(files)} conversation files in {INPUT_DIR}")
    print(f"💾 Results will be saved to {OUTPUT_DIR}\n")

    # Initialise clients once — reuse across all conversations
    clients = {
        "qwen"   : OpenAI(api_key=QWEN_API_KEY,      base_url=QWEN_BASE_URL),
        "llama"  : OpenAI(api_key=LLAMA_API_KEY,      base_url=LLAMA_BASE_URL),
        "openai" : OpenAI(api_key=OPENAI_API_KEY),
        "claude" : anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
        "gemini" : OpenAI(api_key=GEMINI_API_KEY,     base_url=GEMINI_BASE_URL),
    }

    model_config = [
        ("Qwen",   call_qwen,   clients["qwen"],   "qwen"),
        ("Llama",  call_llama,  clients["llama"],  "llama"),
        ("GPT",    call_openai, clients["openai"], "openai"),
        ("Claude", call_claude, clients["claude"], "claude"),
        ("Gemini", call_gemini, clients["gemini"], "gemini"),
    ]

    all_dfs     = []
    total_start = time.time()

    for conv_idx, filepath in enumerate(files, start=1):
        df = process_conversation(conv_idx, filepath, clients, model_config)
        if not df.empty:
            all_dfs.append(df)

    # Combine all into one master CSV
    if all_dfs:
        combined_csv = os.path.join(OUTPUT_DIR, "all_results.csv")
        pd.concat(all_dfs, ignore_index=True).to_csv(combined_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ Combined CSV → {combined_csv}")

    total_elapsed = time.time() - total_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)

    print(f"\n{'='*65}")
    print(f"  All done!")
    print(f"  Conversations : {len(all_dfs)}")
    print(f"  Total time    : {total_mins}m {total_secs}s")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
