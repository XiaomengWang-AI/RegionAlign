# RegionAlign

RegionAlign is a self-contained Python package that tests one research hypothesis about bit-flip
attacks (BFA) on quantized language and vision-language models:

> When an attacker first obtains only **one pre-registered 4 KiB page of a layer's quantized weight
> storage**, can a **function-preserving weight re-layout chosen by the attacker** make bit flips
> inside that page more damaging than the original layout or a random legal layout?

Everything is a **matched comparison**. For the same model, task, quantization format, region,
flip budget and search budget, three layouts are attacked:

| layout | meaning |
|---|---|
| `noperm` | the original weight layout |
| `local` | an attacker-chosen, function-equivalent permutation (RegionAlign) |
| `random` | a seed-determined random legal permutation the attacker cannot choose |

The threat model has a single constraint: the region.
- **Region:** an aligned 4 KiB page of one decoder `Linear` weight's INT8/INT4 codes, registered before search.
- **Flips:** any stored bit inside the page may flip, in either direction.
- **Numeric rule (in-range):** a flipped weight must stay finite and within the row's clean max-abs value.

**Rearranging comes first:** a re-arranged model must first pass a *clean gate*, meaning it behaves like the original model before any fault. Fault damage is then measured against the same re-arranged model without faults.

This is an **offline software fault simulation**. It is not a physical DRAM/Rowhammer
measurement, and a finite search that finds no attack does not prove a model is safe.

## What the package does

1. **Quantize** a Hugging Face checkpoint so that *all* language-model parameters are INT8 or INT4. This includes embeddings, `lm_head`, norm gains and biases.
2. **Freeze a plan** for an experiment matrix:
   - models × formats × tasks;
   - 12 registered regions per condition, spread over early, middle and late layers and over different projection kinds;
   - 3 repeat seeds, 3 layouts, search budget 300, flip budgets K = 1–16.

   The plan snapshots the source code (sha256), inputs and panels, and refuses to run if any of them changes.
3. **Search** (`algorithm/search.py`): for each region and layout, find the most damaging in-range flip set at every K, on a frozen *search* panel.
4. **Evaluate** independently on a separate frozen *dev* panel. The evaluator alone decides legality (`harness/region.py: check_region_attack`). Results are written for K = 1, 2, 4, 8, 16. The success rules are:
   - **MMLU:** accuracy drops by ≥ 50 % relative to the same layout without faults;
   - **WikiText:** perplexity at least doubles;
   - **OK-VQA / TextVQA:** VQA score drops by ≥ 50 %.
5. **Injection ASR** (optional): on frozen layouts, measure how often *random* single-region faults succeed (uniform or high-bit-biased fault law), with per-trial ΔNLL traces.
6. **Report**: cost, damage and matched-comparison summaries (`reporting/`).

Registered models:
- **LLMs:** Llama-3.2-1B, Qwen2.5-1.5B, StableLM-2-1.6B, SmolLM2-1.7B, Gemma-2-2B, Gemma-3-1B (plus the larger Llama-2-7B, Llama-3.1-8B, Qwen2.5-7B).
- **VLMs:** SmolVLM-Instruct, Qwen2-VL-2B, Qwen2.5-VL-3B (plus LLaVA-1.5-7B, Qwen2-VL-7B).

Tasks: MMLU, WikiText-2, OK-VQA and TextVQA.

## 1. Install

Requirements:
- Linux with an NVIDIA GPU (search and evaluation need CUDA; tests run on CPU);
- Python 3.10;
- about 60 GB of disk for the small models and datasets.

```bash
mkdir work && cd work
# The package directory must be named "regionalign"; all commands run from its parent.
git clone https://github.com/XiaomengWang-AI/RegionAlign.git regionalign

# Environment A: Llama, Qwen2.5, StableLM-2, SmolLM2, Gemma-2, LLaVA, Qwen2-VL
python3.10 -m venv envA && source envA/bin/activate
pip install -r regionalign/requirements.txt          # install torch==2.4.1 with the CUDA build for your GPU
deactivate

# Environment B: also Gemma-3, SmolVLM and Qwen2.5-VL (newer Transformers); runs every model
python3.10 -m venv envB && source envB/bin/activate
pip install -r regionalign/requirements-tf451.txt
```

Every plan records the library versions it was made with and refuses to run under others. Use one environment per plan.

Point the package at a model and data cache. The walkthrough below uses this variable in its commands, so export it in every shell
(it overrides the `"hf_home"` key in `regionalign/runtime.json`):

```bash
export REGIONALIGN_HF_HOME=/path/to/huggingface_cache
```

At run time the package sets Hugging Face to offline mode, so download all inputs first (step 3).

## 2. Check the installation (CPU, about 15 min)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s regionalign/tests -t .
```

**Expected:** `OK`, with about 377 tests and no failures.
The tests use tiny random models. They cover several properties:
- flips are reversible;
- permutations preserve both function and the stored codes;
- quantization is complete and the in-range rule holds;
- plans are frozen and sharding is matched;
- the package is self-contained.

## 3. Download models and datasets

```bash
# LLM checkpoints: llama3.2-1b, qwen2.5-1.5b, stablelm2-1.6b, smollm2-1.7b, gemma2-2b, gemma3-1b, ...
python -m regionalign.experiments.input_audit download --model qwen2.5-1.5b      # repeat per model
python -m regionalign.experiments.input_audit download --dataset text-vqa          # VLM runs only

# MMLU and WikiText-2 need a configuration name, which the helper does not pass; download them directly:
python -c "import os; from datasets import load_dataset; d = os.environ['REGIONALIGN_HF_HOME'] + '/datasets'; \
load_dataset('cais/mmlu', 'all', cache_dir=d); load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=d)"

# The small VLMs are not in the helper's registry; download them with the Hub CLI:
for repo in HuggingFaceTB/SmolVLM-Instruct Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen2.5-VL-3B-Instruct; do
  huggingface-cli download "$repo" --cache-dir "$REGIONALIGN_HF_HOME/hub"
done

python -m regionalign.experiments.input_audit audit                                # writes an input audit
```

`--revision` (both tools) pins a Hub commit. To reproduce this project exactly, use the
checkpoint revisions and dataset files recorded in the `models` and `datasets` fields of its plans.

OK-VQA has no training split on the Hub. Build the audited official train inputs once:

```bash
python -m regionalign.experiments.official_okvqa prepare-inputs --output regionalign/results/experiments/okvqa_official_inputs
```

## 4. Run a matched LLM experiment

Each step writes under `regionalign/results/` (append-only; existing runs are never overwritten).

```bash
# 4a. Freeze a plan. Name each model with key=/path/to/checkpoint (a Hugging Face snapshot dir).
python -m regionalign.experiments.llm plan \
  --model qwen2.5-1.5b=$REGIONALIGN_HF_HOME/hub/models--Qwen--Qwen2.5-1.5B/snapshots/<rev> \
  --wikitext-train      $REGIONALIGN_HF_HOME/datasets/wikitext/.../wikitext-train.arrow \
  --wikitext-validation $REGIONALIGN_HF_HOME/datasets/wikitext/.../wikitext-validation.arrow \
  --output regionalign/results/experiments/my_run/plans/llm
# -> prints the run directory; plan.json lists the conditions, e.g. qwen2.5-1.5b__int8__mmlu

RUN=regionalign/results/experiments/my_run/plans/llm
COND=qwen2.5-1.5b__int8__mmlu

# 4b. Search, then independent evaluation. Split the work into 36 shards (one per region × repeat;
#     each shard holds all three layouts); run shards in parallel, e.g. as a Slurm array.
for s in $(seq 0 35); do
  python -m regionalign.experiments.llm search   --run $RUN --condition $COND --shard $s/36
  python -m regionalign.experiments.llm evaluate --run $RUN --condition $COND --shard $s/36
done

# 4c. Progress
python -m regionalign.experiments.llm status --run $RUN
```

**Expected output per condition:**
- `artifacts/<condition>/`: 108 search artifacts (12 regions × 3 repeats × 3 layouts), each holding the flip set found for K = 1–16 and the permutation used.
- `eval/<condition>/`: 108 evaluation reports.

Each evaluation report contains:
- `rearranged.gate`: whether the re-arranged layout passed the clean gate;
- `results[K]`, for K = 1, 2, 4, 8, 16, with fields `valid`, `delta_nll`, `relative_score_drop` (or `ppl_ratio`) and `success`.

A small 1–2 B model takes about 20–60 GPU-minutes per shard on an A100/H100.

**VLMs** follow the same steps with `regionalign.experiments.vqa`, plus a mandatory real-GPU
`preflight --run $RUN --condition $COND` before `search`/`evaluate`. The plan takes:
- `--model smolvlm-instruct=/path`;
- `--textvqa-train <arrow files>`;
- `--okvqa-official regionalign/results/experiments/okvqa_official_inputs`.

Use environment B for SmolVLM and Qwen2.5-VL.

## 5. Optional: injection ASR on frozen layouts

```bash
python -m regionalign.experiments.llm_injection_trace prepare \
  --source-run $RUN --law uniform \
  --regions '[{"condition": "qwen2.5-1.5b__int8__mmlu", "region": "L08.v_proj.p7"}]' \
  --output regionalign/results/experiments/my_run/inject_uniform
python -m regionalign.experiments.llm_injection_trace run \
  --run regionalign/results/experiments/my_run/inject_uniform --condition qwen2.5-1.5b__int8__mmlu
```

`--law highbias` weights each bit by 2^index (sign and high bits first). The default is 200 trials per layout at K = 1 and 16.

**Expected:** in `inject/<condition>/`, one JSON per layout item of the region and per requested budget set. That is 7 files: 1 `noperm`, 3 `local` and 3 `random` repeats, each covering both K values. Pass `--item` and `--ks` to split the work finer.

For each K, a file holds:
- the ASR with a 95 % Wilson interval;
- outcome counts;
- the per-trial ΔNLL trace.

## 6. Summaries

`reporting/campaign_costs.py` builds per-arm and paired (local vs noperm, local vs random)
summaries, damage-by-K tables and coverage from a finished campaign (`--base`, `--output`).
Clean-baseline tables come from `reporting/mmlu.py` and `reporting/clean_matrix.py`.

## Results of this project

The experiment outputs are not in this repository: `results/` is append-only and not version-controlled.
All results live in `results/experiments/`, organized by experiment code, then plan, then condition; `results/campaigns/` and the other old top-level names are read-only symlink aliases kept so that paths recorded in frozen plans and logs still resolve.
`RESULTS.md` indexes every experiment and plan. For each one it lists:
- what it ran and its status;
- where its check reports are;
- which git commit reproduces its frozen sources byte for byte.

The Gemma and VLM plans load on the current branch; to load any other old plan, check out its commit as a worktree (see `RESULTS.md`).

## Repository layout

| path | purpose |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | Operating contract for contributors and coding agents (hypothesis, what may change, self-check) |
| `RESULTS.md` | Index of all experiment campaigns, their status and reproducing commits |
| `config.py`, `runtime.json` | Paths, offline settings, append-only output rules; `hf_home` cache location |
| `requirements.txt`, `requirements-tf451.txt` | The two pinned environments |
| `harness/core.py` | Kernel: quantization, legal permutations, bit flips, in-range check, base runtime |
| `harness/region.py` | Region geometry, the three layouts, region attack check, compute-budget context |
| `harness/evidence.py` | JSON sanitising helpers |
| `algorithm/search.py` | The shared region attack search |
| `runtimes/text.py` | Frozen MMLU / WikiText token panels and objectives |
| `runtimes/vqa.py`, `runtimes/vqa_score.py` | VLM panels, multimodal passes (full model or cached image features), official VQA scoring |
| `experiments/llm.py` | LLM matched matrix: `plan`, `search`, `evaluate`, `status` |
| `experiments/vqa.py` | VLM matched matrix: `plan`, `preflight` (required), `search`, `evaluate`, `status` |
| `experiments/vqa_reeval.py` | Re-evaluate VQA reports whose summary crashed when the clean score is 0 (reuses search artifacts) |
| `experiments/official_okvqa.py` | Official OK-VQA train inputs and their audit |
| `experiments/llm_injection.py`, `llm_injection_biased.py`, `llm_injection_trace.py` | Random-fault injection ASR: uniform law, high-bit-biased law, per-trial trace (use the trace module) |
| `experiments/input_audit.py` | Download and audit models and datasets |
| `experiments/clean_mmlu.py`, `clean_wikitext.py`, `clean_benchmarks.py`, `clean_memory.py` | Fault-free baselines, model loading, input and source audit |
| `experiments/campaign_protocol.py` | New plans that inherit panels byte for byte, with explicit resource policy and independent compute budget |
| `experiments/campaign_variants.py` | Frozen ablation variants (`equal_exposure`, `distinct_weights`, budget variants), reference baselines |
| `experiments/campaign_stage.py` | Stage coverage verification and completion markers |
| `experiments/campaign_memory.py`, `bounded_checksum.py` | Chunked integer-checksum implementation with its own plan identity (`campaign_memory prepare` forks a frozen plan) |
| `experiments/campaign_evaluation.py` | Final and cross-task evaluation after all dev artifacts are sealed (the only path that may open test panels) |
| `experiments/nohup_campaign.py`, `campaign_watch.py`, `timeout_recovery.py` | Queue scheduling with dependencies, Slurm allocation, waiting for external evidence, bounded resumption after Slurm time limits |
| `experiments/layout_validation.py` | Fault-free layout-equivalence and gradient-resource checks on real weights |
| `experiments/task_transfer.py`, `checkpoint_transfer.py` | Cross-task and cross-checkpoint transfer evaluation |
| `reporting/` | Tables and figures from finished campaigns |
| `tests/helpers.py` | Tiny random models and runtime fixtures shared by the tests |
| `tests/test_core.py`, `test_decoder_families.py` | Kernel: reversible flips, function- and code-preserving permutations (incl. LayerNorm and Gemma decoders), complete quantization, in-range rule |
| `tests/test_region_core.py` | Regions, legal layouts, search interface, bit-identical reuse, package self-containment |
| `tests/test_llm_region_study.py`, `test_vqa_region_study.py` | LLM / VLM matrices: matched repeats, isolation, source snapshots, input identity, evaluation rules |
| `tests/test_vqa_reeval.py` | None-safe summary equals the original one; only summary crashes are selected |
| `tests/test_matrix_vqa_runtime.py`, `test_vqa_cached_runtime.py` | Multimodal runtime, GQA and vision-channel sync; cached-feature runtime parity with the full model |
| `tests/test_official_okvqa.py` | Official OK-VQA selection equivalence, source binding, no replacement of failed images |
| `tests/test_llm_injection.py`, `test_llm_injection_biased.py`, `test_llm_injection_trace.py` | Fault sampling, success rule, plan hashing, biased law, per-trial trace consistency |
| `tests/test_mmlu_clean.py`, `test_wikitext_clean_dev.py`, `test_clean_benchmarks.py`, `test_clean_memory.py` | Clean baselines: scoring, quantization, panel isolation, memory policy |
| `tests/test_campaign_*.py`, `test_bounded_checksum.py`, `test_nohup_campaign.py`, `test_timeout_recovery.py`, `test_layout_validation.py` | Campaign tooling: inheritance, variants, stages, checksums, queueing, recovery, final-access gate, costs |
| `tests/test_task_transfer.py`, `test_checkpoint_transfer.py` | Transfer evaluations on tiny real runtimes |
| `tests/test_config.py` | Output boundaries, offline environment, refusal to overwrite, download scope |
| `results/` | Outputs (append-only, not in git); data in `results/experiments/`, old paths are aliases |

Module docstrings are the authoritative description of each implementation.
