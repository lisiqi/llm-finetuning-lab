# LLM Fine-Tuning Lab

A hands-on laboratory for understanding **LoRA and QLoRA**, designed for an M1 Mac with 16 GB unified memory. The concrete task is passage-grounded answering and abstention. The completed experiment compares regular LoRA on an unquantized BF16 base with LoRA on an MLX 4-bit base, using matched training settings and four-way held-out evaluation.

Both 200-step training runs and all four 200-example evaluations completed. Quantized LoRA reduced peak training allocation by 36.54%, with roughly 14% longer training time; both adapters over-abstained and degraded answering performance. This is a completed learning experiment, not a production model.

## Learning guide

Read [LoRA and QLoRA](docs/lora-and-qlora.md) for a concept map, transfer-learning and fine-tuning background, a worked low-rank example, quantization mechanics, and connections to this experiment.

## Experiment question

Can a small instruction model learn to return concise passage-supported answers and `INSUFFICIENT_EVIDENCE` when a passage cannot answer a question, without sacrificing answer quality?

The task isolates generation with supplied context. It is **not a retrieval benchmark** and does not inject new facts into model weights intentionally. Retrieval is outside the scope of this experiment.

## Setup

Run all commands from this repository. Python 3.11+ is required. The dataset and metric commands use only Python's standard library. Model commands require Apple Silicon/macOS and MLX.

```bash
uv sync --locked --extra mlx --no-editable
uv run --locked --extra mlx --no-editable python -m unittest discover -s tests -v
```

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`. Python 3.13 is selected by `.python-version`; uv manages the project `.venv` automatically. Commands use `--no-editable` to install a built package consistently, including in cloud-synced directories. After source edits, rerun `uv sync --locked --extra mlx --no-editable --reinstall-package llm-finetuning-lab`. Commit `uv.lock` and use `--locked` to prevent unintended dependency changes. To intentionally update dependencies, run `uv lock --upgrade`, review the changes, and sync again. Install uv using its official instructions: https://docs.astral.sh/uv/getting-started/installation/ . The lab records actual package versions in each run.

## 1. Download and prepare public data

```bash
uv run --locked --extra mlx --no-editable finetuning-lab download
uv run --locked --extra mlx --no-editable finetuning-lab prepare
```

Downloads the canonical SQuAD 2.0 training and development JSON files, validates schema, and records SHA-256 hashes. Preparation uses 500 train / 100 validation / 200 test examples, approximately half unanswerable in each split. Train and validation use disjoint training articles; test comes from development articles absent from training. Shared exact passages are removed across pools. Seed: 42.

Training targets use the first reference answer. Evaluation accepts all available reference answers. No evaluation answer is included in the inference prompt. Plausible distractor answers for unanswerable examples are not used as targets.

Contexts exceeding 1,600 characters are excluded before sampling. This changes the distribution toward shorter passages. Actual token lengths are checked separately with the model tokenizer, including chat template and target. Never truncate the target to fit a budget. If the check fails, prepare a new directory with a lower character cap, or increase `max_seq_length` cautiously; update both the config and evaluation reference path accordingly.

`data/prepared/` contains MLX chat JSONL files, separate reference JSONL files, and a provenance manifest. Generated data is ignored by Git; publish conversion code rather than copies of Wikipedia passages.

## Implementation and evaluation

The lab implements data preparation, experiment orchestration, resource measurement, and evaluation. LoRA layers and the training loop are supplied by MLX-LM, not implemented from scratch. Both methods call `train_model` in `src/finetuning_lab/training_worker.py`; `setup_models` in `benchmark.py` determines whether the frozen base is quantized.

| File | Responsibility |
|---|---|
| `src/finetuning_lab/core.py` | Article-level splits, prompts, sampling, and scoring |
| `src/finetuning_lab/cli.py` | Data download, training launch, generation, and evaluation |
| `src/finetuning_lab/training_worker.py` | MLX-LM adapter training, callbacks, memory/time measurement |
| `src/finetuning_lab/benchmark.py` | Matched model variants, four-way experiment, comparison checks |
| `configs/m1.json` | Shared training settings |
| `tests/` | Metric edge cases, reference coverage, deterministic splits, model-source/tokenizer validation |

Evaluation reloads saved adapters in fresh processes. It uses greedy generation with a maximum of 64 new tokens. Reference answers are not supplied in the inference prompt. Answer metrics use SQuAD-style normalization; abstention requires exactly `INSUFFICIENT_EVIDENCE`. Undefined precision/recall is reported as `null`, not zero. This custom subset/protocol is not the official SQuAD leaderboard evaluation.

Actual complete-example token lengths, including template and target, were checked before training: maximum 482 training / 392 validation / 428 test tokens, all within the 512-token budget. The lab refuses over-budget examples rather than silently truncating answers.

The five unit tests cover scoring, deterministic data splits, and model-source/tokenizer validation. Integration checks verified both methods, saved-adapter reloads, report generation, and rejection of mismatched training-step settings. The full-run manifests and metrics record model/data hashes, package versions, and training settings. The experiment findings and verification details are documented here.

## Reviewing the code and results

- Read `src/finetuning_lab/core.py` for split construction and metrics.
- Read `src/finetuning_lab/cli.py` for network access, training, and inference.
- Inspect a few answerable/unanswerable examples in each reference file.
- Compare baseline/adapter failures, false abstentions, and unsupported answers.
- Record training config, package versions, model revision, memory, time, and data hashes.
- Explain why prompting alone might be sufficient and whether adaptation actually helped.

## Method terminology

LoRA trains low-rank updates while base parameters remain frozen. MLX calls LoRA over a quantized base **QLoRA**. Its quantization implementation is not necessarily the original bitsandbytes NF4/double-quantization/paged-optimizer setup. Describe the actual toolchain: **quantized LoRA using MLX on Apple Silicon**.

## Data and model attribution

SQuAD 2.0: Rajpurkar, Jia, and Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD* (2018). Dataset project: https://rajpurkar.github.io/SQuAD-explorer/ . Observe dataset/Wikipedia attribution and applicable licences before redistribution.

Qwen model card/licence: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct . Model and dataset licences are separate from this project's code. No automatic upload or publication is implemented.

MLX training documentation: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md

## 2. Download and prepare both model variants

Both experiments have been completed: regular LoRA uses the unquantized BF16 model, and quantized LoRA uses the MLX 4-bit model. Each was trained for 200 steps and evaluated against its own unadapted baseline on the same 200 held-out examples. The preparation commands below reproduce the completed setup.

The shared training worker measures both methods in fresh processes. Both use the same pinned Qwen revision, tokenizer, article splits, adapter configuration, completion-only objective, step count, seed, and sequence budget. Regular LoRA uses the unquantized BF16 base; quantized LoRA uses MLX 4-bit weights. This is not a bitsandbytes NF4 benchmark.

To reproduce in a fresh checkout, prepare both variants from pinned Qwen revision `7ae557604adf67be50417f59c2c2f167def9a775`:

```bash
uv run --locked --extra mlx --no-editable finetuning-lab setup-benchmark-models
```

Run a small integration check in a **fresh** output directory:

```bash
uv run --locked --extra mlx --no-editable finetuning-lab benchmark --output runs/benchmark-smoke-new --steps 5 --eval-limit 4
```

Reproduce the full experiment in a fresh output directory:

```bash
uv run --locked --extra mlx --no-editable finetuning-lab benchmark --output runs/benchmark-reproduction
```

In a fresh clone, run data and model preparation first. Skip preparation only when those local artifacts already exist. The recorded experiment used `runs/benchmark-full/`; generated datasets, weights, and runtime outputs are excluded from Git.

This executes four evaluations (unquantized baseline, LoRA adapter, quantized baseline, quantized adapter), and two training runs sequentially. It writes per-method configs, adapter weights, structured training metrics, generation details, manifests, and `report.md` / `summary.json`. No existing output directory is overwritten. There is no auto-resume; choose a fresh directory after failure. Model setup similarly refuses existing model directories.

Recreate the report without rerunning training:

```bash
uv run --locked --extra mlx --no-editable finetuning-lab benchmark-report --output runs/benchmark-full
```

### Measurement definitions

- **Loaded base active bytes:** materialized model allocation after load and device synchronization, before adapter construction.
- **Training peak allocated bytes:** reset peak after base load and dataset setup; measure adapter/optimizer initialization, training, validation, and checkpoint saving. Both methods use the same boundary. This is total peak within the region, not incremental peak above base memory.
- **Training wall seconds:** the same region, including compilation/warm-up, validation, and saves. Model loading/token checks are excluded.
- **Loss tokens per compute second:** MLX trainer completion-loss token counts divided by accumulated training-step time. Prompt tokens also consume computation but are not counted in this throughput numerator; this is not total context throughput. Reporting is per step to avoid incomplete reporting-window ambiguity.
- **Storage:** base safetensors total and final adapter weight size, excluding intermediate adapter checkpoints.

The runner verifies the model source and tokenizer, actual trainable parameter shapes, matching settings/data hashes/environment, model-file fingerprints, and identical held-out evaluation inputs. The quantized/unquantized model hashes intentionally differ. Within each method, baseline and adapter use exactly the same frozen base files.

The default order is LoRA then quantized LoRA. Thermal throttling, compilation, and warm OS caches can affect speed. Use `--seed 43 --order qlora-first` with a fresh output directory to repeat with a different seed and reverse method order before claiming general performance. Five-step runs are labelled smoke tests; an apparent quality gain is not evidence of successful adaptation.

### Where to inspect results

For the recorded experiment, read [the included report](results/benchmark-2026-10-12.md). After a local run, start with its `report.md` for the memory/speed and four-way quality tables. `summary.json` contains structured training metrics; `benchmark.json` records shared settings and model sources.

Each method has its own `baseline/`, `adapted/`, and `adapter/` directories. Inspect `baseline/details.jsonl` and `adapted/details.jsonl` for reference answers and predictions; `adapter/training.log` for training/validation losses; and `adapter/training_metrics.json` for allocation, speed, and parameter shapes. `quality-comparison.md` compares each adapter with its own baseline.

Compare each adapter to **its own** baseline to assess adaptation, and the two baselines to assess quantization. Resource savings and answer quality are separate findings; neither implies the other.

## Full experiment findings — 2026-10-12

The learning goal was to understand LoRA and quantized LoRA through implementation and measure their memory, speed, and task-quality tradeoffs. The full run completed; it demonstrates this workflow, but does not establish successful task adaptation.

### Setup and evidence

A [recorded benchmark report](results/benchmark-2026-10-12.md) is included in the repository. Its source is the local `runs/benchmark-full/report.md`, `benchmark.json`, and per-method training/evaluation metrics. Full runtime artifacts are ignored by Git; the report and findings below preserve the main results.

Both methods used Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`, the same SQuAD 2.0 subset (500 training, 100 validation, 200 test examples), and seed 42. Test data included 100 answerable and 100 unanswerable questions. Each method trained for 200 steps with batch size 1, rank 8, scale 16, four adapted layers, learning rate 1e-4, a 512-token budget, completion-only loss, and gradient checkpointing. Regular LoRA used an unquantized BF16 base; quantized LoRA used an MLX 4-bit base. Both trained 733,184 adapter parameters.

### Memory and speed

| Measure | LoRA | MLX quantized LoRA |
|---|---:|---:|
| Loaded base allocation (decimal GB) | 0.988 | 0.281 |
| Peak training allocation (decimal GB) | 1.710 | 1.085 |
| Training wall time (seconds) | 83.663 | 95.392 |
| Completion-loss tokens / compute second | 23.457 | 20.517 |
| Trainable parameters | 733,184 | 733,184 |
| Base weight storage (decimal GB) | 0.988 | 0.278 |
| Final adapter weight storage (decimal MB) | 2.939 | 2.939 |

**Finding:** Quantized LoRA reduced peak training allocation by **36.54%**, while training wall time increased by approximately **14.02%** and completion-token throughput decreased by approximately **12.53%**. Loaded base allocation and weight storage fell by roughly 72%. Adapter size and parameter count remained unchanged. Training peak savings are smaller than base-weight savings because activations, optimizer state, gradients, and temporary buffers also consume memory.

### Four-way task-quality evaluation

Here, baseline means the model **without any trained adapter**. The two baseline variants isolate quantization's effect; each adapted variant is compared against its own baseline to assess fine-tuning.

| Configuration | Answer exact match | Answer token F1 | Abstention precision | Abstention recall |
|---|---:|---:|---:|---:|
| Unquantized base, no adapter | 0.5300 | 0.6706 | undefined | 0.0000 |
| Unquantized base + LoRA | 0.1200 | 0.1545 | 0.5366 | 0.8800 |
| Quantized base, no adapter | 0.2400 | 0.4824 | undefined | 0.0000 |
| Quantized base + adapter | 0.0900 | 0.1030 | 0.5367 | 0.9500 |

Answer metrics apply only to answerable questions. Abstention precision measures how often a refusal is appropriate; recall measures the fraction of unanswerable questions correctly refused. Baseline precision is undefined because neither baseline emitted the exact abstention marker.

**Finding:** Both adapters learned refusal behaviour but over-abstained. LoRA correctly refused 88/100 unanswerable questions and incorrectly refused 76/100 answerable questions. Quantized LoRA correctly refused 95/100 unanswerable questions and incorrectly refused 82/100 answerable questions. Answer exact match and F1 deteriorated substantially after training. The quantized baseline also had lower answer accuracy than the unquantized baseline in this run.

Overall exact match was 50% for LoRA adapted and 52% for quantized LoRA adapted. Those aggregate scores can obscure the failure: an always-abstain model would score 50% exact match on this balanced test set. Retaining separate answer and abstention metrics is therefore essential.

### Conclusions and limitations

- The memory-efficiency hypothesis was supported **for this model, configuration, and run**. A speed penalty accompanied the memory reduction.
- Neither final adapter achieved a useful balance of answering and abstention. These results do not justify deployment or a claim of improved grounded QA.
- This is one seed and one model size, with fixed training order (LoRA, then quantized LoRA). No confidence intervals or repeated full-run estimates were produced. Thermal conditions, compilation, and caches may affect timing.
- Memory is MLX allocation, not total system RAM. Training measurements include adapter/optimizer setup, validation, and checkpoint saving within the documented boundary.
- This uses MLX quantized LoRA, not the original bitsandbytes NF4/double-quantization/paged-optimizer QLoRA implementation. Results cannot be transferred directly to that stack.
- Data is a small, balanced, short-context subset. It is not representative of arbitrary customer workloads, long-context RAG, or the original SQuAD distribution.
- Exact match/token F1 and an exact abstention marker are imperfect proxies for semantic correctness. The task supplies context directly; it does not evaluate retrieval or citation correctness.
- The final 200-step adapters were evaluated. Earlier checkpoints were not compared for validation-based selection, and no detailed failure analysis has been completed. Over-abstention is observed; its underlying training cause has not been established.

The experiment demonstrates the mechanics of adapter training and the measured resource tradeoff. Improving task quality is a separate extension. If pursued, inspect failures and select checkpoints/configurations using validation data; keep a fresh holdout for final assessment after further tuning.
