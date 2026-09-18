# Picoformer

## Quick CPU generation check

Load a saved training checkpoint and stream 512 new tokens after `Hi, `:

```bash
uv run python -m picoformer.vibe_check /path/to/epoch_0_step_499
```

Use `--prompt "Once upon a time"` or `--max-new-tokens 128` to change the
input or output length. Generation is greedy and continues through EOS until
the requested token count. The script hides GPUs, loads float32 weights on CPU,
and uses four CPU threads by default (`--num-threads` changes this), so it can
run alongside training.

Pass a finished checkpoint directory or an exported Hugging Face model directory.
NeMo shards are automatically exported to `model/consolidated` on first use.
The tokenizer defaults to the checkpoint's tokenizer if present, otherwise
`HuggingFaceTB/SmolLM2-135M`; override it with `--tokenizer`.

## Learning-rate and batch-size sweep

The sweep grid is defined in `config/sweep.yaml`. By default it runs all six
combinations of three learning rates and two local batch sizes, sequentially:

```bash
uv run picoformer-sweep --multirun
```

Hydra writes every resolved training configuration to that job's
`outputs/sweeps/.../automodel.yaml`, then launches it with two AutoModel
processes. Each job also gets a separate checkpoint directory.

Edit `hydra.sweeper.params` in `config/sweep.yaml` to make the grid persistent,
or override it for a one-off sweep:

```bash
uv run picoformer-sweep --multirun \
  optimizer.lr=5e-5,1e-4 \
  step_scheduler.local_batch_size=1,2,4
```

AdamW is used by default. Select Muon for a run, or include both optimizers in
the grid, with Hydra's `sweep_optimizer` choice:

```bash
uv run picoformer-sweep --multirun sweep_optimizer=muon
uv run picoformer-sweep --multirun sweep_optimizer=adamw,muon
```

The Muon choice uses Lion for embeddings, normalization weights, and other
non-matrix parameters; matrix parameters use Muon.

To inspect the generated YAML files without starting training, add
`sweep.dry_run=true`.

## Compute-aware hyperparameter scaling laws

`picoformer-scaling` runs a separate Optuna grid-search study for each model-size
and FLOP-target pair in `config/scaling.yaml`. The model sizes and FLOP targets
form a Cartesian sweep, and tokens are derived with `D = C / (6N)`. Every trial uses Muon and
searches the configured learning-rate and global-batch-size candidates while
keeping weight decay fixed. `n_trials` caps the total number of grid points stored
per study across all resumptions (and the grid size is a hard upper bound). The
objective is the final loss on the independent `validation-*.bin`
split, not training loss.

```bash
uv run picoformer-scaling --config config/scaling.yaml
```

To regenerate the tables, fitted laws, and plot from the existing study
databases without running any new training trials:

```bash
uv run picoformer-analyze-scaling --config config/scaling.yaml
```

Each study is resumable through its SQLite database. The output directory gets:

- `best_settings.csv`: exact best trial plus the geometric median of all trials
  within 0.25% of its validation loss;
- `best_overall.json`: the model size, token count, and Muon hyperparameters
  that achieved the lowest validation loss across all scale points;
- `power_laws.json`: coefficients, exponents, and log-space R-squared for
  `h(C) = a C^b`, fitted independently for every model size using `C = 6 N D`;
- `power_laws.png`: learning-rate and batch-size log-log fits,
  with a separate scaling law for every model size;
- per-trial generated configs, logs, checkpoints, and validation metrics.

The `num_parameters` values in `config/scaling.yaml` are the explicit `N` used
to derive token counts. Keep them synchronized with the exact trainable count
when changing the associated architecture. At least two FLOP-target points are
required for each model size; four or more spanning multiple orders of magnitude
are recommended.

## Forecast validation loss for one FLOP study

Read the saved validation curves and extrapolate each completed trial against
training tokens, using its saved WSD schedule to exclude warmup and final decay:

```bash
uv run python -m picoformer.predict_study \
  /data/picoformer/outputs/scaling/qwen3_5_135m_135136414p_1e+17flops/study.db \
  --tokens 5e8 --output-dir results/forecasts/1e17
```

The installed command is `picoformer-predict-study`. Omit `--tokens` to forecast
to twice the planned training tokens. Use `--trials 0 5` to select trials, or
`--fit-start-fraction 0.2` to also exclude the first 20% of training.
Tokens are `(step + 1) * global_batch_size * sequence_length`, since validation
steps are zero-based. The validation metric's `num_label_tokens` counts evaluation
tokens and is not used for this axis.

The fit is `loss = floor + amplitude * (tokens / reference_tokens)^(-exponent)`,
with nonnegative floor/amplitude and exponent in `[0.01, 3]`. It minimizes squared
error in validation-loss space and requires at least four distinct usable points.
Trials with missing curves, too few points, or no decreasing trend are reported
and skipped. Save periodic validation during training to make these fits possible;
a final-only evaluation cannot be extrapolated.

Outputs are `validation_forecast.png`, `.json` (coefficients, fit RMSE and R², selected
points, exclusions, and skipped trials), and `.csv` (predicted losses by token
count). Faint crosses show excluded observations and dashed lines show the
continuation from the last fitted point. The command also prints the trial with
the lowest predicted loss at the target token count and its parameter set, and
saves this summary as `best_at_target` in the JSON. Selection is among the fitted
trials with R² ≥ 0.95 (restricted by `--trials` if supplied), using predicted loss
at the target. Fits below this threshold remain plotted but cannot win parameter
selection. If no fit qualifies, the command reports this and writes
`best_at_target: null` in the JSON.
Each fit's R² is printed and shown in the legend. It is calculated in loss space
using only the included fitting points, excluding warmup and final decay.
These are estimates assuming continued
training at the plateau learning rate; they do not predict a future cooldown.
Short curves may poorly constrain the floor and long-range prediction, even with
small fit RMSE. Fits reaching an exponent bound are flagged.

## Language-model evaluation plan

Use [EleutherAI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
for reproducible English-only evaluation. Keep the harness version or commit,
task versions, tokenizer, prompt settings, and model checkpoint in every result.
Save sample generations as well as aggregate metrics, and do not combine the
different task metrics into a single score.

The evaluation is split into a base-model suite and an instruction-model suite.
The base suite must not use a chat template. The instruction suite must use the
same tokenizer and chat template that were used for supervised fine-tuning.

### Pretraining suite

| Capability | Harness tasks | Metrics to inspect |
| --- | --- | --- |
| English language modeling | `wikitext`, `lambada_openai` | Perplexity and exact match |
| Commonsense reasoning | `hellaswag`, `piqa`, `arc_easy`, `winogrande` | Accuracy, preferably `acc_norm` where available |
| Dialogue understanding | `mutual`, `coqa` | R@1/MRR and F1 |
| Direct arithmetic | `arithmetic` | Accuracy for all ten subtasks |
| Elementary word problems | `asdiv` | Accuracy |
| Stretch math | `gsm8k` | Flexible-extract exact match |

`arithmetic` and `asdiv` are the primary math signals at this model size.
Treat GSM8K as a stretch diagnostic: a result close to zero is plausible for a
135M-parameter model and should not by itself determine training decisions.
MuTual and CoQA measure the language understanding needed for conversation, but
do not demonstrate that the base model can behave as an assistant.

After installing and pinning the harness, evaluate a consolidated Hugging Face
checkpoint with a command of this form:

```bash
uv run lm-eval run \
  --model hf \
  --model_args pretrained=/path/to/hf-checkpoint,tokenizer=HuggingFaceTB/SmolLM2-135M,dtype=bfloat16 \
  --tasks wikitext lambada_openai hellaswag piqa arc_easy winogrande mutual coqa arithmetic asdiv gsm8k \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/pretrain \
  --log_samples
```

### Instruction-tuned suite

| Capability | Harness tasks | Metrics to inspect |
| --- | --- | --- |
| Instruction compliance | `ifeval` | Prompt- and instruction-level strict and loose accuracy |
| Social and emotional dialogue understanding | `eq_bench` | EQ score and `percent_parseable` |
| Conversational question answering | `coqa` | F1 |
| Multi-turn dialogue reasoning | `mutual_plus` | R@1/MRR |
| Generated elementary math | `asdiv_cot_llama` | Flexible-extract exact match |
| Stretch mathematical reasoning | `gsm8k` | Flexible-extract exact match |

Run these generative and conversational tasks through the post-SFT tokenizer's
chat template:

```bash
uv run lm-eval run \
  --model hf \
  --model_args pretrained=/path/to/instruct-hf-checkpoint,dtype=bfloat16 \
  --tasks ifeval eq_bench coqa mutual_plus asdiv_cot_llama gsm8k \
  --apply_chat_template \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/instruct \
  --log_samples
```

Also rerun a small base-format retention suite without a chat template. This
separates capability regressions introduced by instruction tuning from changes
caused only by prompt formatting:

```bash
uv run lm-eval run \
  --model hf \
  --model_args pretrained=/path/to/instruct-hf-checkpoint,dtype=bfloat16 \
  --tasks hellaswag piqa arc_easy winogrande arithmetic \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/instruct-retention \
  --log_samples
```

### Evaluation procedure

1. Pin LM Evaluation Harness rather than silently following its main branch.
2. Export the final NeMo AutoModel checkpoint as consolidated Hugging Face
   safetensors. Verify that `AutoModelForCausalLM` can load it before running the
   harness. Ensure that the checkpoint also contains the SmolLM2 tokenizer and,
   for the instruction model, the exact SFT chat template.
3. First run every generative task with `--limit 20 --write_out --log_samples`.
   Inspect prompt formatting, stopping behavior, answer extraction, and malformed
   responses before launching the full evaluation.
4. Use deterministic decoding for reported benchmark results and retain the raw
   generations for error analysis.
5. Record results separately for each task and checkpoint. Compare pretraining
   checkpoints using the pretraining suite, and compare SFT checkpoints using
   both the instruction and retention suites.

The harness tasks are only proxies for natural conversation. Before final model
selection, add a private held-out evaluation containing approximately 50 short,
scripted multi-turn English conversations and 200 generated-answer arithmetic
questions. Keep it out of all training data. Initially skip MATH, AIME, GPQA,
MMLU-Pro, and BBH because they are likely to produce uninformative floor scores
at this scale.

### Evaluating saved runs

Install the pinned harness and its task dependencies once:

```bash
uv sync --extra eval
```

The evaluator recursively finds every `epoch_<N>_step_<N>` directory, exports
intermediate NeMo shards to Hugging Face format when necessary, runs each task,
keeps raw samples and a reproducibility manifest, and updates one plot per
task/metric. Checkpoint filenames use zero-based steps, so `step_99` is plotted
at 100 completed optimizer steps.

Run the required 20-example prompt inspection first:

```bash
uv run picoformer-evaluate-checkpoints \
  /data/picoformer/outputs/sweeps/2026-08-11_14-52-23 \
  --suite pretrain --phase smoke
```

After inspecting `results/evaluations/**/smoke/lm-eval.log` and the saved sample
JSON, run the complete suite. Completed checkpoint/suite pairs are skipped when
the command is resumed.

```bash
uv run picoformer-evaluate-checkpoints \
  /data/picoformer/outputs/sweeps/2026-08-11_14-52-23 \
  --suite pretrain --phase full
```

For an SFT run, `--suite instruct` automatically runs both the chat-templated
instruction suite and the non-chat retention suite. The exact SFT tokenizer is
required:

```bash
uv run picoformer-evaluate-checkpoints /path/to/sft-run \
  --suite instruct --phase full --tokenizer /path/to/sft-tokenizer
```

Use `--dry-run` to inspect commands, or `--steps 100,500` to select completed
optimizer steps. The evaluator streams harness progress and prints a heartbeat
after 15 seconds without output; change that interval with
`--heartbeat-seconds`. At startup it also prints the raw Hugging Face download
cache and the prepared benchmark cache. Results default to
`results/evaluations`; regenerate their CSV and plots without evaluating again
with:

```bash
uv run picoformer-plot-evaluations results/evaluations
```
