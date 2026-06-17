# Cognitive Atrophy Benchmark — Code Repository


![Cognitive Atrophy attribute schema: 5 user-context attributes (U), 10 LLM-response behaviour attributes (R), and 5 binary risk flags (F).](figures/cara_schema.png)

This repository contains all executable artifacts that accompany the **Cognitive Atrophy Benchmark** — a multi-attribute evaluation framework for
LLM behavior in mental-health support contexts. It pairs with the public dataset hosted on Hugging Face:

🤗 **Dataset:** [https://huggingface.co/datasets/CABenchmark/Cognitive_Atrophy_Benchmark](https://huggingface.co/datasets/abadawi/Cognitive_Atrophy_Benchmark)

## What's in here

```
github_code/
├── llm_generation/                  Generate the 5-LLM responses on a new prompt set
│   ├── carebench_LLM_generator.py
│   ├── counselchat_LLM_generator.py
│   ├── hope_LLM_generator.py
│   └── pair_LLM_generator.py
│
├── annotation_ui/                   Browser-only annotation tool (no server)
│   ├── single_turn_evaluator.html      blank, drop in your own CSV
│   ├── multi_turn_evaluator.html       blank, drop in your own CSV
│   ├── csv_to_json.py                  optional helper
│   ├── examples/with_data/             4 ready-to-open demos with our datasets baked in
│   ├── README.md
│   └── LICENSE
│
├── analysis/                        Compute every metric from the paper
│   ├── data/                           4 anonymized human-evaluation CSVs (R1-R6 + gold_standard)
│   ├── metrics_single_turn/            5 scripts (UIRI, ARI, attributes, correlations, highlights)
│   ├── metrics_multi_turn/             6 scripts (same + per-conversation, flags)
│   ├── requirements.txt
│   └── README.md
│
└── README.md                        you are here
```

The three folders correspond to the three stages of the benchmark pipeline:

1. **Generate** — `llm_generation/` produces five model responses per prompt across four mental-health corpora (CounselChat, PAIR, HOPE, CareBench). The five LLM responses for our prompt set are released on Hugging Face under the `*_responses` configs.
2. **Annotate** — `annotation_ui/` is the actual tool clinician-trained raters used to score the responses against a 20-code coding manual. It is included so reviewers can see how the human-evaluation data was collected, and so other researchers can reuse it on their own response sets.
3. **Analyze** — `analysis/` implements every metric the paper reports (UIRI, ARI, per-attribute marginals, correlations with FDR correction, span/token highlights, multi-turn turn-level dynamics, binary-flag firing rates). Bundled with the four anonymized human-eval CSVs from Hugging Face so the scripts run end-to-end with no download step.

## Quickstart for reviewers

The fastest way to confirm reproducibility is to re-run the analysis pipeline:

```bash
cd analysis
pip install -r requirements.txt

# single-turn metrics
python metrics_single_turn/compute_uiri.py
python metrics_single_turn/compute_ari.py
python metrics_single_turn/compute_attributes.py
python metrics_single_turn/compute_correlations.py
python metrics_single_turn/compute_highlights.py

# multi-turn metrics  (compute_response_multiturn must run before compute_highlights_multiturn)
python metrics_multi_turn/compute_response_multiturn.py
python metrics_multi_turn/compute_uiri_multiturn.py
python metrics_multi_turn/compute_correlations_multiturn.py
python metrics_multi_turn/compute_per_conversation_correlations.py
python metrics_multi_turn/compute_flags_multiturn.py
python metrics_multi_turn/compute_highlights_multiturn.py
```

Each script reads from `analysis/data/`, writes its outputs alongside, and prints a summary to stdout. All eleven scripts have been verified to run cleanly on a stock install of pandas / numpy / scipy / tiktoken.

To explore the annotation tool that produced the human ratings:

```bash
open annotation_ui/examples/with_data/demo_pair.html
# or any of the four demos: demo_counselchat, demo_carebench, demo_hope
```

These open in any modern browser — no server, no install. They load with the corresponding subset of the released dataset already embedded so you can click through the rating UI as the original annotators experienced it.

## Reproducing LLM generation

The four generator scripts in `llm_generation/` produce the model responses
that populate the `*_responses` configs on the Hugging Face dataset. They are
released as-is so reviewers can verify the prompts, inference settings, and
parallelization strategy. Running them requires:

- API access to all five providers (Hugging Face Inference Providers, OpenAI, Anthropic, Google AI Studio) plus a Qwen endpoint (we used a vec-inf cluster; any OpenAI-compatible endpoint works).
- API keys exposed via standard environment variables (`HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).
- Input CSVs/XLSX placed at the path specified by the `INPUT_DIR` / `INPUT_FILE` constant near the top of each script — placeholders in the released code; edit to point at your data.

The scripts use identical inference settings across providers for fair cross-model comparison: `temperature=1.0`, `top_p=1.0`, `max_tokens=2048`, output trimmed to 300–350 words at the nearest sentence boundary. Multi-turn scripts pass the last 10 turns of conversation history as messages.

## Companion paper

This code accompanies a NeurIPS 2026 Evaluations & Datasets Track submission. The paper describes the benchmark's motivation (cognitive atrophy in LLM-mediated mental-health support), the coding manual, the inter-rater reliability protocol, and the headline findings across the five evaluated LLMs.

## License

- `annotation_ui/`: MIT (see `annotation_ui/LICENSE`).
- `llm_generation/` and `analysis/`: research-use license, to be finalized at camera-ready. Code is released as-is for reproducibility of the paper's results.
- The released **dataset** (model responses + human ratings) is on Hugging Face under **CC BY-NC 4.0**. See its data card.

## Citation

```bibtex
@article{badawi2026cognitiveatrophy,
  title   = {Towards Understanding and Measuring Cognitive Atrophy in LLM Behaviour},
  author  = {Badawi, Abeer and Olatosi, Moyosoreoluwa and Baghbanzadeh, Negin and Seyyed-Kalantari, Laleh and Rudzicz, Frank and Rosenbaum, R. Shayna and Pishdadian, Sara and Dolatabadi, Elham},
  journal = {[arXiv preprint](https://arxiv.org/pdf/2606.18129)},
  year    = {2026}
}
```

## Upstream datasets

Source prompts come from four publicly released, research-licensed corpora.
Please cite them in addition to this benchmark when using any subset:

```bibtex
@inproceedings{min-etal-2022-pair,
  title     = {{PAIR}: Prompt-Aware margIn Ranking for Counselor Reflection Scoring in Motivational Interviewing},
  author    = {Min, Do June and P{\'e}rez-Rosas, Ver{\'o}nica and Resnicow, Kenneth and Mihalcea, Rada},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year      = {2022},
  pages     = {148--158},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2022.emnlp-main.11}
}

@misc{bertagnolli-2020-counselchat,
  title  = {Counsel Chat: Bootstrapping High-Quality Therapy Data},
  author = {Bertagnolli, Nicolas},
  year   = {2020},
  note   = {Released on Hugging Face: https://huggingface.co/datasets/nbertagnolli/counsel-chat}
}

@inproceedings{yuan-2026-carebench,
  title     = {Can {LLM}s Move Beyond Short Exchanges to Realistic Therapy Conversations?},
  author    = {Yuan, Zhengqing and Wu, Liang and Xu, Jian and Zhang, Zheyuan and Shi, Kaiwen and Sun, Weixiang and Sun, Lichao and Ye, Yanfang},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=3Bdl1wL1S3}
}

@inproceedings{malhotra-etal-2022-hope,
  title     = {Speaker and Time-aware Joint Contextual Learning for Dialogue-act Classification in Counselling Conversations},
  author    = {Malhotra, Gaurav and Waheed, Abdul and Srivastava, Ashutosh and Akhtar, Md Shad and Chakraborty, Tanmoy},
  booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  year      = {2022},
  pages     = {735--745},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3488560.3498509}
}
```
