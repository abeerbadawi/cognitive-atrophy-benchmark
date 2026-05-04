"""
5-Model AI Response Generator — CounselChat Dataset
─────────────────────────────────────────────────────
Dataset : Single-turn counseling Q&A
          questionText → AI responds (no history, no turns)
Models  : Qwen, Llama, GPT, Claude, Gemini
Output  : counselchat_results.csv
"""

import os
import re
import time
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


# --- Files ---
INPUT_FILE = "ADD_YOUR_INPUT_FILE_HERE"
OUTPUT_DIR = "ADD_YOUR_OUTPUT_DIR_HERE"
OUTPUT_CSV      = os.path.join(OUTPUT_DIR, "counselchat_results.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "counselchat_checkpoint.csv")

# --- Shared params ---
MAX_TOKENS     = 2048
GPT_MAX_TOKENS = 2048
MAX_WORDS      = 350
TEMPERATURE    = 1.0
TOP_P          = 1.0

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
# BUILD MESSAGES — single turn, no history
# ─────────────────────────────────────────────────────────────────
def build_messages(user_input: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_input},
    ]

def build_messages_claude(user_input: str) -> list[dict]:
    return [
        {"role": "user", "content": user_input},
    ]

# ─────────────────────────────────────────────────────────────────
# MODEL CALL FUNCTIONS — no history parameter
# ─────────────────────────────────────────────────────────────────
def call_qwen(client, user_input):
    try:
        r = client.chat.completions.create(
            model       = QWEN_MODEL_ID,
            messages    = build_messages(user_input),
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            top_p       = TOP_P,
            extra_body  = {"chat_template_kwargs": {"enable_thinking": False}},
        )
        return trim_response(clean_text(strip_thinking(r.choices[0].message.content)))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_llama(client, user_input):
    try:
        r = client.chat.completions.create(
            model       = LLAMA_MODEL_ID,
            messages    = build_messages(user_input),
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            top_p       = TOP_P,
        )
        return trim_response(clean_text(r.choices[0].message.content.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_openai(client, user_input):
    try:
        r = client.chat.completions.create(
            model                 = OPENAI_MODEL_ID,
            messages              = build_messages(user_input),
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

def call_claude(client, user_input):
    try:
        r = client.messages.create(
            model      = ANTHROPIC_MODEL_ID,
            max_tokens = MAX_TOKENS,
            system     = SYSTEM_PROMPT,
            messages   = build_messages_claude(user_input),
        )
        return trim_response(clean_text(r.content[0].text.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

def call_gemini(client, user_input):
    try:
        r = client.chat.completions.create(
            model       = GEMINI_MODEL_ID,
            messages    = build_messages(user_input),
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            top_p       = TOP_P,
        )
        return trim_response(clean_text(r.choices[0].message.content.strip()))
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"

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
    print("  5-Model Generator  |  CounselChat  |  Single-Turn")
    print("  Qwen | Llama | GPT | Claude | Gemini")
    print(f"  max_tokens={MAX_TOKENS}, max_words={MAX_WORDS}")
    print("="*65 + "\n")

    if not validate_keys():
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    df_input = pd.read_excel(INPUT_FILE)
    print(f"📂 Loaded {len(df_input)} rows from {INPUT_FILE}")

    # Check for existing checkpoint to resume
    if os.path.exists(CHECKPOINT_FILE):
        done_df  = pd.read_csv(CHECKPOINT_FILE)
        done_ids = set(done_df["questionID"].tolist())
        print(f"♻️  Resuming — {len(done_ids)} already done")
        results  = done_df.to_dict("records")
    else:
        done_ids = set()
        results  = []

    # Initialise clients
    clients = {
        "qwen"   : OpenAI(api_key=QWEN_API_KEY,     base_url=QWEN_BASE_URL),
        "llama"  : OpenAI(api_key=LLAMA_API_KEY,     base_url=LLAMA_BASE_URL),
        "openai" : OpenAI(api_key=OPENAI_API_KEY),
        "claude" : anthropic.Anthropic(api_key=ANTHROPIC_API_KEY),
        "gemini" : OpenAI(api_key=GEMINI_API_KEY,    base_url=GEMINI_BASE_URL),
    }

    MODEL_CONFIG = [
        ("Qwen",   call_qwen,   clients["qwen"]),
        ("Llama",  call_llama,  clients["llama"]),
        ("GPT",    call_openai, clients["openai"]),
        ("Claude", call_claude, clients["claude"]),
        ("Gemini", call_gemini, clients["gemini"]),
    ]

    total      = len(df_input)
    start_time = time.time()

    for idx, input_row in df_input.iterrows():
        qid          = input_row["questionID"]
        question     = str(input_row["questionText"]).strip()
        answer       = str(input_row["answerText"]).strip()
        label        = str(input_row.get("label", ""))
                # Skip if already done
        if qid in done_ids:
            continue

        row_start = time.time()
        print(f"\n  [{idx+1}/{total}] Q{qid} | {question[:70].replace(chr(10),' ')}...")

        row = {
            "questionID"    : qid,
            "questionText"  : question,
            "answerText"    : answer,
            "label"         : label,
        }

        # Call all 5 models in parallel
        def call_model(args):
            model_name, call_fn, client = args
            response = call_fn(client, question)
            return model_name, response

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = list(executor.map(call_model, MODEL_CONFIG))

        for model_name, response in futures:
            print(f"  {model_name:<8}: {response[:90].replace(chr(10),' ')}...")
            row[f"{model_name} Output"] = response

        row["Turn Time (s)"] = round(time.time() - row_start, 2)
        results.append(row)

        # Checkpoint after every row
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")

    # Save final CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    elapsed    = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    print(f"\n{'='*65}")
    print(f"  All done!")
    print(f"  Rows processed : {len(results)}")
    print(f"  Total time     : {mins}m {secs}s")
    print(f"  Output CSV     : {OUTPUT_CSV}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()