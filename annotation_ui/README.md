# LLM Response Evaluator — Annotation UI

A pair of single-file, no-server, browser-only annotation tools used to score
LLM responses on a fine-grained coding manual covering decisiveness, autonomy
preservation, empathy, fixation, language matching, and clinical-safety flags.
Originally built to operationalize the **Cognitive Atrophy Benchmark**, the
two HTML files are intentionally dataset-agnostic: drop in your own CSV and
you can evaluate any model output against the same coding scheme — or edit
the HTML to swap the coding scheme entirely.

```
annotation_ui/
├── single_turn_evaluator.html    ← blank: single-turn prompts (one user message → one model reply)
├── multi_turn_evaluator.html     ← blank: multi-turn conversations (one model reply per turn, with history)
├── csv_to_json.py                ← optional helper (legacy)
├── examples/
│   ├── sample_single_turn.csv    ← 2 demo prompts, 5 model outputs each
│   ├── sample_multi_turn.csv     ← 2 demo conversations, 5 outputs/turn
│   ├── sample_single_turn.json   ← same shape, JSON form (reference only)
│   ├── sample_multi_turn.json    ← same shape, JSON form (reference only)
│   └── with_data/                ← FOUR ready-to-open demos with full datasets baked in
│       ├── demo_pair.html              (single-turn, ~268 prompts)
│       ├── demo_counselchat.html       (single-turn, ~310 prompts)
│       ├── demo_hope.html              (multi-turn,  ~113 conversations)
│       └── demo_carebench.html         (multi-turn,  ~251 conversations)
├── README.md
└── LICENSE
```

## Two ways to use this

### A) Just want to see it work? Open a baked-in demo

Double-click any of the four files inside `examples/with_data/` — each one is the *complete* evaluator with one of our four datasets already loaded. No download, no setup, no CSV needed. Opens straight into the rating UI:

| File | Dataset | Format | Size |
|---|---|---|---|
| `demo_pair.html`        | PAIR (counselor reflections) | Single-turn | 776 KB |
| `demo_counselchat.html` | CounselChat (therapy Q&A)    | Single-turn | 1.1 MB |
| `demo_hope.html`        | HOPE (therapy transcripts)   | Multi-turn  | 3.2 MB |
| `demo_carebench.html`   | CareBench (multi-turn)       | Multi-turn  | 4.0 MB |

Each demo uses its own `localStorage` key, so progress in `demo_hope.html` doesn't collide with progress in `demo_pair.html`. Useful for getting a feel for the rating workflow before deciding whether to use the tool on your own data.

### B) Have your own dataset? Use a blank evaluator

1. Open `single_turn_evaluator.html` or `multi_turn_evaluator.html` directly in Chrome / Edge / Firefox. (Just double-click — no server needed. Safari works but its `file://` permissions can occasionally block CSV loads; use Chrome/Edge/Firefox if you hit that.)
2. Enter an annotator ID at the modal prompt (e.g. `evaluator_01`).
3. Click **📂 Load Dataset** in the topbar (or on the empty-state screen).
4. Pick a CSV matching the schema below. The sidebar populates with prompts/conversations and you can start scoring.
5. Click **↓ Save JSON** to checkpoint progress at any time, or **↓ Export CSV** to dump all ratings to a wide CSV at the end.

Progress is auto-saved to `localStorage` every 30 seconds and on page unload, keyed by your annotator ID and the dataset name. Closing the tab and reopening on the same machine resumes where you left off.

## Expected CSV schema

The evaluators are designed around a fixed five-model layout. Each row has five model outputs in columns named `Qwen Output`, `Llama Output`, `GPT Output`, `Claude Output`, `Gemini Output`. **You can rename these** by editing the `COL_MAP` constant near the top of each HTML — match the keys to whichever columns your data actually uses.

### Single-turn (`single_turn_evaluator.html`)

A flat CSV — one row per prompt:

| questionID | prompt | label | Qwen Output | Llama Output | GPT Output | Claude Output | Gemini Output |
|---|---|---|---|---|---|---|---|
| 1 | "I've been feeling..." | anxiety | "..." | "..." | "..." | "..." | "..." |
| 2 | "My partner just..."   | relationship | "..." | "..." | "..." | "..." | "..." |

`questionID` and `prompt` are required. `label` is optional. Any extra columns are passed through untouched.

See [`examples/sample_single_turn.csv`](examples/sample_single_turn.csv).

### Multi-turn (`multi_turn_evaluator.html`)

A flat CSV with a `Conversation` column that groups rows into conversations:

| Conversation | Turn | User Input | Original Therapist | Qwen Output | Llama Output | GPT Output | Claude Output | Gemini Output |
|---|---|---|---|---|---|---|---|---|
| conv_01 | 1 | "I think I'm..." | "..." | "..." | "..." | "..." | "..." | "..." |
| conv_01 | 2 | "It started..."  | "..." | "..." | "..." | "..." | "..." | "..." |
| conv_02 | 1 | "My dad died..." | "..." | "..." | "..." | "..." | "..." | "..." |

`Conversation` and `User Input` are required. `Turn`, `Original Therapist`, etc. are optional but rendered if present.

The loader auto-groups rows by the `Conversation` column — you don't pre-split your CSV into one-file-per-conversation.

See [`examples/sample_multi_turn.csv`](examples/sample_multi_turn.csv).

## Plugging in your own data — three paths

### Path 1 — your CSV already has the standard column names
Just open the blank evaluator, click **Load Dataset**, pick the CSV. Done.

### Path 2 — your data has different model column names
Open the HTML in a text editor and find:

```js
const MODELS = ['Response 1','Response 2','Response 3','Response 4','Response 5'];
const COL_MAP = {
  'Response 1':'Qwen Output','Response 2':'Llama Output',
  'Response 3':'GPT Output','Response 4':'Claude Output','Response 5':'Gemini Output'
};
```

Change the values on the right of `COL_MAP` to match your column names. For example, if your CSV has columns `gpt4_response`, `mixtral_response`, etc.:

```js
const COL_MAP = {
  'Response 1':'gpt4_response',
  'Response 2':'mixtral_response',
  ...
};
```

The labels shown in the UI come from `MODELS` — change those if you want different display names (e.g. `['GPT-4', 'Mixtral', ...]`).

### Path 3 — fewer or more than 5 models
Edit `MODELS` and `COL_MAP` together. For two models:

```js
const MODELS  = ['Model A','Model B'];
const COL_MAP = { 'Model A':'baseline_output', 'Model B':'tuned_output' };
```

The export-CSV column count adjusts automatically.

## Customizing the coding scheme

The default coding manual is rendered from three constants near the top of each HTML:

| Constant | Purpose |
|---|---|
| `ATTRIBUTES` | Highlightable codes annotators apply to specific spans of model text (SEN, AUR, FIX, etc.). Each entry is `{code, name, color, bg}`. |
| `LLM_SCORES` | Per-response Likert/categorical scores (Sensitivity, Tentativeness/Directness, Fixation, etc.). |
| `USER_SCORES` / `YN_SCORES` | Ratings on the user prompt itself, plus binary yes/no flags. |

To use your own coding manual, edit these arrays. The export-CSV function reads from them dynamically — no other code change needed.

## Output format

`Export CSV` produces one wide row per (annotator, prompt-or-conversation-turn) with columns for every score and every attribute count, per model. Filenames look like:

```
ratings_evaluator_01_<datasetname>.csv
```

`Save JSON` produces a session snapshot keyed to your annotator ID and the dataset name, suitable for resuming or for sending raw ratings to a coordinator without re-exporting CSV.

## CSV details (what the parser handles)

The inline CSV parser is RFC 4180-style: standard comma separator, double-quote escaping (`""` inside a quoted field), embedded newlines inside quoted fields are fine, and a UTF-8 BOM at the start is silently stripped. If you saved your CSV from Excel or pandas with `quoting=QUOTE_ALL` it will work. If you have multi-character separators, semicolons, or tab-separated files, convert to standard CSV first.

## Privacy and data handling

Everything runs locally in the browser. The page makes one outbound request to load Google Fonts and otherwise never contacts the network. No data leaves your machine unless you explicitly download the CSV/JSON. `localStorage` holds in-progress ratings keyed by your annotator ID — you can clear it via the **change ID** button in the topbar.

## Browser support

Chrome / Edge / Firefox: full support, both `file://` and via local server.
Safari: works, but its default `file://` permissions can block CSV loads. If you hit a "CSV parse error" on Safari, run a local server instead:

```bash
cd annotation_ui
python -m http.server 8000
# then open http://localhost:8000/single_turn_evaluator.html
```

