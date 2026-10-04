# PVA

Code and data for the experiments on three benchmarks:

| Folder | Benchmark | Task |
|---|---|---|
| [`HumanEval/`](HumanEval) | HumanEval / HumanEval+ | code generation, 11 models |
| [`mmlu/`](mmlu) | MMLU, MMLU-Pro | multiple-choice QA |
| [`MedBench/`](MedBench) | MedCalc-Bench | numeric medical calculations |

In each benchmark, several models answer every question. Answers are grouped by similarity
(identical choice, numeric closeness, or code equivalence).

All model outputs and intermediate results are included, so **every figure can be regenerated
without calling any API**. The generation steps are included for completeness.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or later. Figures need matplotlib 3.5 or later.

Only the generation steps need an API key and network access. They use
[OpenRouter](https://openrouter.ai): replace the `"Your key here"` / `"Your API key here"`
placeholder in the generation script or notebook with your key.

Running model-generated code (HumanEval only) is done inside the
[EvalPlus](https://github.com/evalplus/evalplus) Docker image, with no network access:

```bash
docker pull ganler/evalplus:latest
```

## Regenerating the figures

Figures are written to each benchmark's `figures/` folder, which is created on the first run.

```bash
# HumanEval: every figure and the comparison table
bash HumanEval/make_figures.sh

# MedCalc-Bench
cd MedBench && python medcalc_vote.py && python medcalc_figs.py && cd ..

# MMLU, MMLU-Pro: run all cells of mmlu/gen_plots.ipynb (from inside mmlu/)
```

---

## HumanEval

The pipeline runs in four stages. Every stage's output is already in `HumanEval/data/` and
`HumanEval/results/`, so you can start from any stage. All scripts can be run from any folder.

```
generation/  ->  similarity/  ->  selection/  ->  gen_figures.py
```

### 1. Generation (`generation/`)

**Completions.** Each model samples $K = 5$ completions per task, and one is kept per task by
self-consistency on the generated tests.

```bash
# sample 5 completions per task  ->  data/SC/HE_sc5_{model}_completions.json
python HumanEval/generation/he_generation_exp.py -m z-ai/glm-4.5-air

# run the 5 candidates against data/tests.json (in Docker, from HumanEval/)
#   ->  data/json_files/{model}_matrix.json
docker run --rm --platform linux/amd64 --network none --memory 4g --pids-limit 512 \
  -v "$(pwd):/app" -w /app ganler/evalplus:latest \
  python generation/test-sc-models.py --model z-ai_glm-4.5-air

# keep one candidate per task  ->  data/completions/{name}.jsonl
python HumanEval/generation/score-sc.py --model z-ai_glm-4.5-air --name glm-4.5-air
```

| `--model` | `--name` |
|---|---|
| cohere_north-mini-code_free | cohere_north-mini-code |
| deepseek_deepseek-v3.2-exp | deepseek_deepseek-v3.2-exp |
| z-ai_glm-4.5-air | glm-4.5-air |
| google_gemma-4-26b-a4b-it | google_gemma-4-26b-a4b-it |
| openai_gpt-oss-120b | gpt-oss-120b |
| openai_gpt-oss-20b | gpt-oss-20b |
| poolside_laguna-s-2.1 | laguna-s-2.1 |
| meta-llama_llama-3.1-8b-instruct | llama-3.1-8b-instruct |
| meta-llama_llama-3.3-70b-instruct | llama-3.3-70b |
| nvidia_nemotron-3-nano-30b-a3b | nemotron-3-nano-30b-a3b |
| qwen_qwen3-coder-30b-a3b-instruct | qwen3-coder-30b-a3b-instruct |

**Scoring the completions.** Each `data/completions/{name}.jsonl` is scored with EvalPlus, which
writes `{name}_eval_results.json` next to it:

```bash
docker run --rm --platform linux/amd64 -v "$(pwd)/HumanEval:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval --samples /app/data/completions/glm-4.5-air.jsonl
```

**Generated tests.** `generation/test_generation_he.ipynb` asks a model to write assert-based
unit tests for each task and writes `data/reasoned_tests.json`. `data/tests.json` was generated
the same way with reasoning disabled (`extra_body={"reasoning": {"enabled": False}}`).
`generation/validate_gen_tests.py` is a sanity check of these tests: it runs them (in Docker)
against the canonical HumanEval solutions in `data/canonical.json`.

### 2. Similarity (`similarity/`)

```bash
# pass/fail of every completion on every generated test (in Docker, from HumanEval/).
# First merges data/completions/ into data/all_completions.jsonl and
# data/humaneval_{base,plus}_results.csv.
docker run --rm --platform linux/amd64 --network none --memory 4g --pids-limit 512 \
  -v "$(pwd):/app" -w /app ganler/evalplus:latest \
  python similarity/run_matrix.py --tests data/tests.json --out data/init_test_matrix.json
# same with --tests data/reasoned_tests.json --out data/rbased_matrix.json

python HumanEval/similarity/crosshair_similarity.py   # CrossHair equivalence -> data/crosshair_similarity_results.json
python HumanEval/similarity/compute_pwcb.py           # CodeBLEU              -> data/sim_codebleu_pairwise.json
python HumanEval/similarity/gen_adj_similarity.py     # all four, as matrices -> data/adj/{metric}_sim.npz
```

The four similarities, as they are named in the code and in the figures:

| Code | Figures | Two completions are similar if |
|---|---|---|
| `tests` | RTests | same pass/fail pattern on the tests generated with reasoning enabled|
| `init_tests` | Tests | same pass/fail pattern on the generated tests (reasoning disabled)|
| `crosshair` | CrossHair | CrossHair finds no input on which they differ |
| `codebleu` | CodeBLEU | CodeBLEU $\geq 0.9$ |

### 3. Selection (`selection/`)

```bash
bash HumanEval/make_figures.sh --select
```

This runs, with the settings used in the paper:

- `selection/vpa.py`: our method for every similarity, $\tau \in \lbrace 1, 2, 3, 6, 7 \rbrace$, both tie rules,
  with and without ranking groups by tests passed $\rightarrow$ `results/mv_out/`.
- `selection/byzantine_he.py`: the same under a copy adversary that adds f copies of a wrong
  answer $\rightarrow$ `results/adv_out/`.
- `selection/codeT.py`: the CodeT baseline under the same adversary $\rightarrow$ `results/adv_out/`.

Every selected `.jsonl` in `results/*/max_completions/` is then scored with EvalPlus, as in
stage 1. To score a whole folder:

```bash
docker run --rm --platform linux/amd64 -v "$(pwd)/HumanEval:/app" --entrypoint bash \
  ganler/evalplus:latest -c \
  'for f in /app/results/*/max_completions/*.jsonl; do
     evalplus.evaluate --dataset humaneval --samples "$f"; done'
```

### 4. Figures

```bash
bash HumanEval/make_figures.sh     # runs gen_figures.py
```

| File in `HumanEval/figures/` | Content |
|---|---|
| `HumanEval_error_vs_plurality.pdf` | error on answered tasks vs $\tau$ and vs coverage |
| `HumanEval_coverage_accuracy.pdf` | accuracy on answered tasks vs coverage |
| `HumanEval_correct_fragmentation.pdf` | how often the correct answers end up split across groups |
| `HumanEval_adversary_error_vs_tau.pdf` | error under the copy adversary ($f = 0, 1, 2$) |
| `HumanEval_rtests_vs_codet.pdf` | ours vs CodeT under the copy adversary |
| `ablation/ablation_error_vs_tau.pdf` | ranking by tests passed vs by group size only |
| `ablation/table_tests_vs_models.tex` | ours vs each base model on the same answered tasks |

---

## MMLU and MMLU-Pro (`mmlu/`)

| File | Role |
|---|---|
| `benchmarking_mmlu.ipynb` | queries the models on the MMLU questions in `data/selected_questions.json` |
| `mmlu_pro_gen.ipynb` | the same for MMLU-Pro (`data/PRO_selected_questions.json`) |
| `self-consistency-mmlu.ipynb` | several samples per model (self-consistency setting) |
| `aggregations.py` | grouping identical answers, plurality vote with abstention |
| `utils.py` | aggregation over the result tables, adversarial voters, plots |
| `gen_plots.ipynb` | all MMLU figures, from `data/mmlu_300qst_results.csv`, `data/pro_mmlu_results.csv` and `data/sc_mmlu_results.csv` |

Each result table has one row per question, one column per model with its answer letter, and
the correct answer in `correct_label`.

**Data.** The tables read by `gen_plots.ipynb` are cleaned versions of the generation outputs:
model columns renamed to short names, answers unchanged.

- `mmlu_300qst_results.csv`: the first 300 questions of `mmlu_benchmark_results.csv` (the ones
  answered by the most models), 11 models, renamed.
- `pro_mmlu_results.csv`: `MMLU_PRO_benchmark_results.csv`, renamed, without `qwen3.6-27b`.
- `mmlu_results.csv`: the 7 models of the self-consistency setting from
  `mmlu_benchmark_results.csv`, renamed (one sample per model).
- `sc_mmlu_results.csv`: `mmlu_results.csv` with the answers on the 83 questions of
  `self_consist_mmlu_results.csv` replaced by the self-consistency answers.

## MedCalc-Bench (`MedBench/`)

| File | Role |
|---|---|
| `medcalc_generation.ipynb` | queries the models on the questions in `data/MedCalc_Numeric_Questions.json` |
| `medcalc_vote.py` | plurality vote on `data/filtered_medCalc.csv`, for $\tau = 1, \dots, 8$ $\rightarrow$ `results/`. Decimal answers within 5% of each other are grouped; other answers must match exactly |
| `medcalc_figs.py` | figures and LaTeX tables from `results/` $\rightarrow$ `figures/` |

**Data.** `data/filtered_medCalc.csv` holds generations from the 9 models
used in the paper.

```bash
cd MedBench
python medcalc_vote.py
python medcalc_figs.py
```
